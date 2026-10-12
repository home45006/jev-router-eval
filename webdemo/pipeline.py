"""第 14 版前置分流的全链路实现，每一步都记录耗时和判断依据。

    会话检查 → Jev 第一级（is_fast、needs_rewrite、纠错题，一次请求）→ 分流决策
      → 快思考：（需要时）DeepSeek 改写 → 快思考模型 → 拒识或低置信时转大脑
      → Agent 大脑：标签模式（一次调用输出能力标签）或编排模式（调用能力工具，多轮推理）

每一步通过 emit 回调实时推给前端。
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

REPO = Path(__file__).resolve().parents[1]
CRITERIA = json.loads((REPO / "criteria_v2.json").read_text(encoding="utf-8"))

REWRITE_PROMPT = ("你是车载语音助手的改写模块。用户的话里如果有俗称、外号、简称、昵称指代某个地点、人物或作品，"
                  "把它换成唯一的标准名称；其余字词一个都不要改。没有需要还原的，原样输出。只输出改写后的一句话。")

# 大脑下挂的能力。label 是输出的 Agent 标签，tool 是给模型的工具名。
ABILITIES = [
    ("快思考", "fast_think", "执行说清楚的车控、媒体点播、导航去某地、打电话等直接指令（快思考模型）。"),
    ("任务大师", "task_master", "建立长期有效、反复触发的规则：每天定时、每周提醒、每当上车或进入某种模式时自动执行。"),
    ("条件任务", "conditional_task", "一次性的定时或延时动作和提醒，如后天两点导航到火车站、十分钟后关车窗、5月30日提醒我开会；查询删除日程。"),
    ("联网搜索", "web_search", "新闻资讯、热点、需要实时信息的问题。"),
    ("用车手册", "car_manual", "本车功能是什么、怎么用、在哪设置，故障报警原因，某个部件当前状态是否正常。"),
    ("导航", "navigation", "靠描述找地点、规划路线、多地点行程、两地距离和路况、顺路找加油站餐馆，会多轮确认。"),
    ("空调", "climate", "根据感受和环境（热、冷、闷、起雾）推断空调、座椅通风加热、车窗怎么调。"),
    ("多媒体", "media", "按心情、场景、歌词、剧情找歌找片，推荐音乐和影视，问演员歌手作品。"),
    ("美团", "meituan", "找餐厅、点外卖、下单，需要多轮确认菜品和地址。"),
    ("记忆", "memory", "记住用户的个人信息、喜好、常去的地点。"),
    ("闲聊", "chat", "开放问答、百科、写诗、建议、日期时间、陪伴聊天。"),
]
LABEL_OF_TOOL = {t: l for l, t, _ in ABILITIES}
LABELS = [l for l, _, _ in ABILITIES]


def ability_list() -> str:
    return "\n".join(f"- {l}（{t}）：{d}" for l, t, d in ABILITIES)


BRAIN_RULES = f"""你是智能座舱的 Agent 大脑。快思考模型处理不了的话都交给你。你下挂这些能力：
{ability_list()}

先判断用户这句话要办什么事、需要哪些能力，一件事需要几个能力配合时按执行顺序列出。
用户原话后面如果附有“前置分流提示”，优先考虑提示里的能力，但以用户原话为准。
如果有进行中的任务（见对话历史），而用户这句话与它无关（例如点外卖途中说“把空调调低”），status 填 suspend。"""

LABEL_MODE_PROMPT = BRAIN_RULES + """
只输出一个 JSON 对象，不要输出其他内容：
{"labels": [按顺序列出需要的能力标签，只能用上面括号前的中文名], "plan": "一句话说明打算怎么做",
 "status": "continue（任务还要多轮继续）| suspend（这句与进行中的任务无关）| end（本轮就能完成）", "speech": "播报给用户的一句话"}"""

ORCH_MODE_PROMPT = BRAIN_RULES + """
需要时调用对应的工具（参数 task 写清楚任务和必要的上下文），可以连续调用多个。
全部完成后不要再调用工具，只输出一个 JSON 对象：
{"labels": [实际需要的能力标签，按顺序], "plan": "一句话说明做了什么", "status": "continue | suspend | end", "speech": "播报给用户的一句话"}"""

ABILITY_PROMPT = ("你是座舱里的“{label}”能力（{desc}）。这是演示环境，没有接入真实服务："
                  "请给出你会怎么处理和一个合理的模拟结果，两三句话，开头标注“模拟”。")

TOOLS = [{"type": "function", "function": {
    "name": t, "description": f"{l}：{d}",
    "parameters": {"type": "object", "properties": {"task": {"type": "string", "description": "要完成的任务和上下文"}},
                   "required": ["task"]}}} for l, t, d in ABILITIES]


@dataclass
class Settings:
    is_fast_t: float = 0.7
    rewrite_t: float = 0.5
    correction_t: float = 0.5
    corrections: dict[str, str] = field(default_factory=lambda: {k: v["status"] for k, v in CRITERIA["corrections"].items()})
    brain_mode: str = "label"           # label：一次调用出标签；orchestrate：调用能力工具
    simulate_fast_reject: bool = False
    stream: bool = True                 # 大模型流式输出，记录首 token 耗时
    run_search: bool = True             # 大脑判出“联网搜索”时真的去搜（DeepSeek 自带联网搜索）
    max_iters: int = 6

    @classmethod
    def from_dict(cls, d: dict | None) -> "Settings":
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k) and v is not None:
                setattr(s, k, v)
        return s


@dataclass
class Session:
    history: list[dict] = field(default_factory=list)   # 只存用户原话和大脑播报
    active: bool = False                                 # 大脑占用会话
    labels: list[str] = field(default_factory=list)
    turns: int = 0


Emit = Callable[[dict], Awaitable[None]]


class Trace:
    def __init__(self, emit: Emit) -> None:
        self.t0 = time.perf_counter()
        self.emit = emit
        self.steps: list[dict] = []

    def now(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)

    def streamer(self, s: dict, key: str = "stream_text"):
        """返回 on_delta 回调：把流式增量拼进 s["detail"][key]，并且最多每 80ms 推一次给前端。"""
        s["detail"][key] = ""
        last = [0.0]

        async def on_delta(text: str) -> None:
            if s["detail"].get("first_token_at") is None:
                s["detail"]["first_token_at"] = self.now()       # 相对本轮开始的时刻，画瀑布图用
            s["detail"][key] += text
            if time.perf_counter() - last[0] > 0.08:
                last[0] = time.perf_counter()
                await self.emit({"type": "step", "step": s})
        return on_delta

    @asynccontextmanager
    async def step(self, name: str, kind: str, **detail):
        s = {"id": len(self.steps), "name": name, "kind": kind, "start": self.now(), "end": None,
             "status": "running", "summary": "", "detail": dict(detail)}
        self.steps.append(s)
        await self.emit({"type": "step", "step": s})
        try:
            yield s
            s["status"] = s["status"] if s["status"] != "running" else "ok"
        except Exception as e:  # 记录后继续抛出
            s["status"], s["summary"] = "error", f"{type(e).__name__}: {e}"[:300]
            raise
        finally:
            s["end"] = self.now()
            await self.emit({"type": "step", "step": s})


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _clean_labels(labels) -> list[str]:
    out = []
    for x in labels or []:
        x = str(x).strip()
        x = LABEL_OF_TOOL.get(x, x)
        if x in LABELS and x not in out:
            out.append(x)
    return out


def _llm_detail(r) -> dict:
    return {"model": r.model, "ms": round(r.ms, 1), "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "reasoning_tokens": r.reasoning_tokens, "cost": r.cost,
            "ttft_ms": round(r.ttft_ms, 1) if r.ttft_ms is not None else None}


class Pipeline:
    def __init__(self, jev, llm, fast, search=None) -> None:
        self.jev, self.llm, self.fast, self.search = jev, llm, fast, search
        self.sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------ #
    async def run(self, query: str, session_id: str, settings: Settings, emit: Emit,
                  previous_turn: dict | None = None) -> dict:
        tr = Trace(emit)
        sess = self.sessions.setdefault(session_id, Session())
        if previous_turn is None and sess.history:
            last_user = next((h["content"] for h in reversed(sess.history) if h["role"] == "user"), None)
            if last_user:
                previous_turn = {"query": last_user}
        result = {"query": query, "session_id": session_id, "destination": None, "labels": [], "path": []}

        # 1. 会话检查
        async with tr.step("会话检查", "code") as s:
            s["detail"].update(active=sess.active, turns=sess.turns, labels=sess.labels)
            s["summary"] = f"大脑占用会话中（{'、'.join(sess.labels) or '进行中的任务'}）" if sess.active else "没有进行中的会话"
        if sess.active:
            result["path"].append("会话占用 → 大脑")
            brain = await self._brain(tr, query, sess, settings, hint=None, occupied=True)
            if brain["status"] != "suspend":
                return self._finish(tr, result, "Agent大脑", brain, sess)
            result["path"].append("大脑：不归当前任务，挂起后走正常分流")
            result["suspended"] = True

        # 2. Jev 第一级
        questions = dict(CRITERIA["stage1"])
        for k, st in settings.corrections.items():
            if st != "off" and k in CRITERIA["corrections"]:
                questions[k] = CRITERIA["corrections"][k]
        state = {"query": query}
        if previous_turn:
            state["previous_turn"] = previous_turn
        async with tr.step("Jev 第一级", "jev", questions=list(questions)) as s:
            jr = await self.jev.ask(state, questions)
            s["detail"].update(probabilities=jr.probabilities, input_tokens=jr.input_tokens,
                               model=jr.model, api_ms=round(jr.ms, 1), state=state)
            p = jr.probabilities
            s["summary"] = f"is_fast {p['is_fast']:.2f} · needs_rewrite {p['needs_rewrite']:.2f}"
        result["jev"] = jr.probabilities

        # 3. 分流决策
        async with tr.step("分流决策", "code") as s:
            reasons, dest, hint, correction = [], None, None, None
            hits = sorted(((p[k], k) for k in questions if k in CRITERIA["corrections"] and p[k] >= settings.correction_t),
                          reverse=True)
            for prob, k in hits:
                c, st = CRITERIA["corrections"][k], settings.corrections.get(k, "off")
                if st == "on" and dest is None:
                    dest, hint, correction = c["route"], c.get("hint"), k
                    reasons.append(f"纠错题 {k} 命中（{prob:.2f}），按它的去向：{c['route']}" + (f"，提示 {hint}" if hint else ""))
                else:
                    reasons.append(f"纠错题 {k} 命中（{prob:.2f}）但处于 {st}，只记录")
            if dest is None:
                dest = "快思考" if p["is_fast"] >= settings.is_fast_t else "Agent大脑"
                reasons.append(f"is_fast {p['is_fast']:.2f} {'≥' if dest == '快思考' else '<'} 阈值 {settings.is_fast_t} → {dest}")
            rewrite = dest == "快思考" and p["needs_rewrite"] >= settings.rewrite_t
            if dest == "快思考":
                reasons.append(f"needs_rewrite {p['needs_rewrite']:.2f} {'≥' if rewrite else '<'} 阈值 {settings.rewrite_t}"
                               + (" → 先改写" if rewrite else " → 不改写"))
            s["detail"].update(reasons=reasons, destination=dest, rewrite=rewrite, correction=correction, hint=hint)
            s["summary"] = dest + ("（先改写）" if rewrite else "") + (f"，纠错题 {correction}" if correction else "")
        result["path"].append(f"Jev → {dest}")
        result["correction"] = correction

        # 4. 快思考
        if dest == "快思考":
            text = query
            if rewrite:
                async with tr.step("大模型改写", "llm") as s:
                    r = await self.llm.chat([{"role": "system", "content": REWRITE_PROMPT},
                                             {"role": "user", "content": query}], role="rewrite",
                                            stream=settings.stream, on_delta=tr.streamer(s))
                    text = r.content.strip() or query
                    s["detail"].update(before=query, after=text, **_llm_detail(r))
                    s["summary"] = f"{query} → {text}"
                result["rewritten"] = text
            async with tr.step("快思考模型", "fast") as s:
                fr = await self.fast.run(text, previous_turn, settings.simulate_fast_reject)
                s["detail"].update(input=text, connected=fr.connected, rejected=fr.rejected,
                                   confidence=fr.confidence, response=fr.detail)
                s["summary"] = ("拒识" if fr.rejected else "理解成功") + ("" if fr.connected else "（服务未接入）")
            result["fast"] = {"input": text, "connected": fr.connected, "rejected": fr.rejected, "detail": fr.detail}
            if not fr.rejected and (fr.confidence is None or fr.confidence >= 0.5):
                result["path"].append("快思考执行")
                result.update(destination="快思考", labels=["快思考"])
                return self._finish(tr, result, None, None, sess)
            result["path"].append("快思考拒识 → 大脑")

        # 5. Agent 大脑
        brain = await self._brain(tr, query, sess, settings, hint=hint, occupied=False)
        return self._finish(tr, result, "Agent大脑", brain, sess)

    # ------------------------------------------------------------------ #
    async def _brain(self, tr: Trace, query: str, sess: Session, settings: Settings,
                     hint: str | None, occupied: bool) -> dict:
        content = query + (f"\n（前置分流提示：可能需要调用{hint}）" if hint else "")
        history = sess.history[-8:]
        if settings.brain_mode == "orchestrate":
            return await self._brain_orchestrate(tr, content, history, settings)
        msgs = [{"role": "system", "content": LABEL_MODE_PROMPT}, *history, {"role": "user", "content": content}]
        async with tr.step("Agent 大脑 · 判断能力", "llm", mode="label", occupied=occupied) as s:
            r = await self.llm.chat(msgs, json_mode=True, role="brain", stream=settings.stream, on_delta=tr.streamer(s))
            out = _parse_json(r.content)
            labels = _clean_labels(out.get("labels"))
            if not labels and occupied and sess.labels:      # 会话中的确认、补充信息：沿用当前任务的标签
                labels = list(sess.labels)
            s["detail"].update(raw=r.content, **_llm_detail(r))
            s["summary"] = "、".join(labels) or "未给出有效标签"
        brain = {"labels": labels, "plan": out.get("plan", ""), "status": out.get("status", "end"),
                 "speech": out.get("speech", ""), "tools_called": [], "mode": "label", "raw": r.content}
        if "联网搜索" in labels and self._search_on(settings):
            answer = await self._web_search(tr, query, settings)
            brain["tools_called"].append({"label": "联网搜索", "task": query, "result": answer})
            brain["speech"] = answer
        return brain

    async def _brain_orchestrate(self, tr: Trace, content: str, history: list[dict], settings: Settings) -> dict:
        msgs = [{"role": "system", "content": ORCH_MODE_PROMPT}, *history, {"role": "user", "content": content}]
        called: list[dict] = []
        out: dict = {}
        for i in range(1, settings.max_iters + 1):
            async with tr.step(f"Agent 大脑 · 第 {i} 次推理", "llm", mode="orchestrate") as s:
                r = await self.llm.chat(msgs, tools=TOOLS, role="brain", stream=settings.stream, on_delta=tr.streamer(s))
                s["detail"].update(content=r.content, tool_calls=[{"name": c.name, "args": c.arguments} for c in r.tool_calls],
                                   **_llm_detail(r))
                s["summary"] = ("调用 " + "、".join(LABEL_OF_TOOL.get(c.name, c.name) for c in r.tool_calls)) if r.tool_calls else "给出结论"
            if not r.tool_calls:
                out = _parse_json(r.content)
                break
            msgs.append(r.raw_message or {"role": "assistant", "content": r.content})
            results = await asyncio.gather(*(self._ability(tr, c, settings) for c in r.tool_calls))
            for c, text in zip(r.tool_calls, results):
                called.append({"label": LABEL_OF_TOOL.get(c.name, c.name), "task": c.arguments.get("task", ""), "result": text})
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": text})
        else:
            out = {"plan": f"达到推理轮数上限 {settings.max_iters}", "status": "end"}
        labels = _clean_labels(out.get("labels")) or _clean_labels([c["label"] for c in called])
        return {"labels": labels, "plan": out.get("plan", ""), "status": out.get("status", "end"),
                "speech": out.get("speech", ""), "tools_called": called, "mode": "orchestrate"}

    def _search_on(self, settings: Settings) -> bool:
        return bool(self.search and self.search.kind in ("real", "mock") and settings.run_search)

    async def _web_search(self, tr: Trace, task: str, settings: Settings) -> str:
        """DeepSeek 自带联网搜索：一次流式请求完成搜索和写播报，答案边生成边推给前端。"""
        try:
            async with tr.step("联网搜索 · DeepSeek", "search", task=task) as s:
                sr = await self.search.search(task, on_delta=tr.streamer(s))
                s["detail"].update(answer=sr.answer, items=sr.items, queries=sr.queries, usage=sr.usage,
                                   model=sr.model, url=sr.url, call_ms=round(sr.ms, 1), cost=sr.cost,
                                   ttft_ms=sr.ttft_ms, first_event_ms=sr.first_event_ms,
                                   search_done_ms=sr.search_done_ms, answer_first_ms=sr.answer_first_ms)
                s["summary"] = f"搜索 {len(sr.queries)} 次，来源 {len(sr.items)} 条：{sr.answer[:30]}"
        except Exception as e:          # 搜索失败不影响本轮：记录错误，播报说明没查到
            return f"联网搜索暂时不可用（{str(e)[:120]}）"
        return sr.answer or "没有查到相关信息。"

    async def _ability(self, tr: Trace, call, settings: Settings | None = None) -> str:
        label = LABEL_OF_TOOL.get(call.name, call.name)
        task = call.arguments.get("task", "")
        if call.name == "web_search" and settings is not None and self._search_on(settings):
            return await self._web_search(tr, task, settings)
        if call.name == "fast_think":
            async with tr.step("能力 · 快思考", "fast", task=task) as s:
                fr = await self.fast.run(task, None, False)
                s["summary"] = "理解成功" + ("" if fr.connected else "（服务未接入）")
                s["detail"].update(response=fr.detail)
            return json.dumps({"status": "done", **fr.detail}, ensure_ascii=False)
        desc = next((d for l, t, d in ABILITIES if t == call.name), "")
        async with tr.step(f"能力 · {label}", "agent", task=task) as s:
            r = await self.llm.chat([{"role": "system", "content": ABILITY_PROMPT.format(label=label, desc=desc)},
                                     {"role": "user", "content": task}], temperature=0.3, role=f"ability:{label}",
                                    stream=settings.stream if settings else True, on_delta=tr.streamer(s))
            s["detail"].update(reply=r.content, **_llm_detail(r))
            s["summary"] = r.content[:40]
        return r.content

    # ------------------------------------------------------------------ #
    def _finish(self, tr: Trace, result: dict, dest: str | None, brain: dict | None, sess: Session) -> dict:
        if brain is not None:
            result["destination"] = dest
            result["labels"] = brain["labels"]
            result["brain"] = brain
            st = brain["status"]
            if st == "continue":
                sess.active, sess.labels = True, brain["labels"]
            elif st == "end":
                sess.active, sess.labels = False, []
            # suspend：会话保留，状态不变
        sess.history += [{"role": "user", "content": result["query"]},
                         {"role": "assistant", "content": (brain or {}).get("speech") or f"[快思考已执行] {result['query']}"}]
        sess.turns += 1
        result["session"] = {"active": sess.active, "labels": sess.labels, "turns": sess.turns}
        steps = tr.steps
        def total(kind):
            return round(sum(s["end"] - s["start"] for s in steps if s["kind"] == kind and s["end"] is not None), 1)
        result["timing"] = {"total_ms": tr.now(), "jev_ms": total("jev"), "llm_ms": total("llm"),
                            "agent_ms": total("agent"), "fast_ms": total("fast"), "search_ms": total("search"),
                            "llm_calls": sum(s["kind"] in ("llm", "agent", "search") for s in steps)}
        costs = [s["detail"]["cost"] for s in steps if (s.get("detail") or {}).get("cost")]
        result["cost"] = {"yuan": round(sum(c["yuan"] for c in costs), 6), "calls": len(costs),
                          "period": costs[0]["period"] if costs else None}
        result["steps"] = steps
        return result

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
