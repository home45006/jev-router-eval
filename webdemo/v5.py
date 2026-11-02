"""v5 分流测试页的后端：按《Jev 分流详细设计 v2》实现一次判定 + 分发策略 R0–R7。

    Jev 一次请求（criteria_v3，6 道题）∥ NLU3.0 整句解析 → Query 整理（需要时）→ 分发策略 → 落域
    NLU3.0 全部 intent 落在 A 类 12 个域时直接用（RA），不看 Jev 的判断，也不改写。

- 判据读 docs/share/criteria_v3.json（由 docs/share/make_criteria_v3.py 生成）。
- Jev：有 TYPESAFE_API_KEY 时真实调用（模型取判据里的版本），否则用关键词模拟。
- NLU3.0：设置 NLU3_V25_URL 时调用 v25 模型（OpenAI 兼容接口，按 docs/nlu3/NLU3.0大模型接口使用说明1030固定版本.md：
  固定 system、扁平 user 串、只用 temperature=0 / top_p=0.01），把 “domain+intent@k=v;…” 解析成 intents；
  多轮带上一轮的 query 和 NLU 原始输出。不设时用关键词模拟，页面上标明“模拟”。地址只放 .env.local，不进仓库。
- 改写：has_nickname 或 needs_context（且不跟随上一轮）时用演示系统的 DeepSeek 改写，再回 Jev 重判一次。
- 会话存在内存里；页面可以模拟业务侧回传 continue 和“落域被拒”（R7）。
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .pricing import JEV_PRICE_SOURCE, JEV_PRICES_USD_PER_M, PRICE_SOURCE, USD_CNY, jev_cost

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CRIT_PATH = REPO / "docs/share/criteria_v3.json"

router = APIRouter()
_S: dict = {"sessions": {}}

REWRITE_PROMPT = ("你是车载语音助手的改写模块。把用户这句话改写成一句不依赖上下文、能单独理解的话："
                  "俗称、外号、简称换成唯一的标准名称；如果给了上一轮，用上一轮补全指代和省略。"
                  "其余字词尽量不改。不需要改写就原样输出。只输出改写后的一句话。")


def crit() -> dict:
    if "crit" not in _S or _S.get("crit_mtime") != CRIT_PATH.stat().st_mtime:
        _S["crit"] = json.loads(CRIT_PATH.read_text(encoding="utf-8"))
        _S["crit_mtime"] = CRIT_PATH.stat().st_mtime
    return _S["crit"]


# ------------------------------------------------------------------ Jev
def _reversed(spec: dict) -> dict:
    return {**spec, "criteria": dict(reversed(list(spec["criteria"].items())))}


REQUIRED_OPTIONS = {   # 分发代码按选项名读答案，这些选项不能删、不能改名
    "route": ["慢思考", "NLU直达", "Agent直达", "闲聊"],
    "agent_label": ["none"],
    "segment_role": ["有依赖", "同一任务", "独立指令", "独立Agent类", "残片"],
}


def check_prompts(p: dict) -> dict:
    """校验页面提交的提示词覆盖；返回规范化后的 {questions, segment_role_template}，不合法时抛 ValueError。"""
    base = crit()
    qs = p.get("questions") or {}
    unknown = set(qs) - set(base["questions"])
    if unknown:
        raise ValueError(f"未知的问题：{sorted(unknown)}")
    out_q = {}
    for k, v in {**base["questions"], **qs}.items():
        out_q[k] = _check_one(k, v)
    tpl = _check_one("segment_role", p.get("segment_role_template") or base["segment_role_template"])
    if "第 1 段 `segments[0]`" not in json.dumps(tpl["instructions"], ensure_ascii=False):
        raise ValueError("segment_role 模板的问法里要保留“第 1 段 `segments[0]`”，代码按它替换成第 i 段")
    return {"questions": out_q, "segment_role_template": tpl}


def _check_one(k: str, v: dict) -> dict:
    if not isinstance(v, dict) or v.get("type") not in ("noul", "choice"):
        raise ValueError(f"{k}：type 必须是 noul 或 choice")
    if not v.get("instructions"):
        raise ValueError(f"{k}：instructions 不能为空")
    c = v.get("criteria")
    if v["type"] == "choice":
        if not isinstance(c, dict) or len(c) < 2:
            raise ValueError(f"{k}：Choice 至少要 2 个选项")
        miss = [o for o in REQUIRED_OPTIONS.get(k, []) if o not in c]
        if miss:
            raise ValueError(f"{k}：缺少分发代码要用的选项 {miss}")
        if len(c) > 255:
            raise ValueError(f"{k}：选项不能超过 255 个")
    elif c is not None and (not isinstance(c, dict) or set(c) - {"true", "false"}):
        raise ValueError(f"{k}：Noul 的 criteria 只能有 true / false")
    return {"type": v["type"], "instructions": v["instructions"], "criteria": c}


def build_questions(segments: list[str] | None, reverse: bool, prompts: dict | None = None) -> dict[str, dict]:
    c = prompts or crit()
    qs = dict(c["questions"])
    if segments:
        tpl = json.dumps(c["segment_role_template"], ensure_ascii=False)
        for i in range(len(segments)):
            qs[f"segment_role_{i}"] = json.loads(tpl.replace("第 1 段 `segments[0]`", f"第 {i + 1} 段 `segments[{i}]`"))
    if reverse:
        qs = {k: (_reversed(v) if v["type"] == "choice" else v) for k, v in qs.items()}
    return qs


def _conf(probs: dict) -> float:
    n = len(probs)
    p = max(probs.values()) if probs else 0
    return (p - 1 / n) / (1 - 1 / n) if n > 1 else 1.0


def _choice(probs: dict) -> dict:
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    return {"type": "choice", "top": ranked[0][0], "second": ranked[1][0] if len(ranked) > 1 else None,
            "conf": round(_conf(probs), 4), "probs": {k: round(v, 4) for k, v in ranked}}


class RealJev:
    """System One 兼容的决策后端：TypeSafe Jev（云端，按 token 计费），或自部署的 StartLux（同一 SDK，换 base_url）。"""
    kind = "real"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 name: str = "Jev", billed: bool = True) -> None:
        from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy
        self._C, self._N = Choice, Noul
        self.model, self.name, self.billed, self.base_url = model, name, billed, base_url
        kw = {"model": model, "retry": RetryPolicy(max_retries=2), "timeout": 180}
        if base_url:
            kw.update(base_url=base_url, api_key=api_key or "unused")
        self._client = AsyncTypeSafeClient(**kw)
        self._entered = None

    async def start(self):
        self._entered = await self._client.__aenter__()

    async def close(self):
        if self._entered is not None:
            await self._client.__aexit__(None, None, None)

    async def ask(self, state: dict, questions: dict[str, dict]) -> dict:
        qs = {k: (self._N if v["type"] == "noul" else self._C)(instructions=v["instructions"], criteria=v["criteria"])
              for k, v in questions.items()}
        t = time.perf_counter()
        r = await self._entered.system_one(state=state, questions=qs)
        ms = (time.perf_counter() - t) * 1000
        out = {}
        for k, a in r.answers.items():
            if getattr(a, "type", "") == "choice":
                out[k] = _choice(dict(a.probabilities))
            else:
                out[k] = {"type": "noul", "p": round(a.noul, 4)}
        try:
            raw = r.model_dump(mode="json")
        except Exception:
            raw = {"answers": {k: getattr(a, "__dict__", str(a)) for k, a in r.answers.items()}}
        it, ot = getattr(r.usage, "input_tokens", None), getattr(r.usage, "output_tokens", None)
        served = getattr(r, "model", None) or self.model
        return {"answers": out, "ms": ms, "input_tokens": it, "output_tokens": ot, "model": served, "backend": self.name,
                "cost": jev_cost(self.model, it, ot) if self.billed else None,
                # 完整请求体（与 HTTP API 的 body 同结构：model + state + questions），密钥不在其中
                "request": {"model": self.model, "state": state, "questions": questions}, "raw": raw}


MOCK_RULES = [   # (正则, route, domain, agent)
    (r"毒品|海洛因|炸弹|分尸|谋杀|自杀|毒死", "Agent直达", "非入口", "none"),
    (r"(\d+|[一二三四五六七八九十半两]+)(分钟|小时)(后|以后)|每天|每周|每当|当.{1,12}(时|的时候)|提醒我|闹钟", "Agent直达", "非入口", "任务大师"),
    (r"记住|记一下|存一下|我常听|我喜欢的", "Agent直达", "非入口", "记忆"),
    (r"怎么关|怎么开|怎么用|感叹号|报警|故障|有没有.{0,6}功能", "Agent直达", "非入口", "用车专家"),
    (r"新闻|热点|资讯|天气|下雨|下雪|气温|空气质量|洗车|星期几|几号|股票|股价|航班|机票|火车|限行|多高|是什么意思|等于多少", "Agent直达", "非入口", "联网搜索"),
    (r"外卖|点.{0,4}(杯|份)|订房|酒店|餐厅|面馆|好吃|人均", "Agent直达", "非入口", "美团"),
    (r"先去|再去|行程|自驾游|一日游|多久", "Agent直达", "非入口", "出行规划"),
    (r"画一|壁纸|主题", "Agent直达", "非入口", "AI创作"),
    (r"有声书|评书|短视频|短片|唱吧|我唱", "Agent直达", "非入口", "内容娱乐"),
    (r"聊|笑话|故事|你觉得", "闲聊", "非入口", "none"),
    (r"好热|好冷|太闷|太暗|闷得慌", "NLU直达", "intelligent_car_scenario", "none"),
    (r"还剩多少|电量|油量|胎压|关好了吗", "NLU直达", "intelligent_car_info", "none"),
    (r"导航|回家|去公司", "NLU直达", "navigation", "none"),
    (r"播放|放一首|来一首|听", "NLU直达", "music", "none"),
    (r"打电话|拨打", "NLU直达", "phone_call", "none"),
    (r"微信", "慢思考", "非入口", "none"),          # NLU3.0 v25 没有车载微信域
    (r"空调|车窗|座椅|天窗|灯|雨刮|后视镜|音量|屏幕", "NLU直达", "intelligent_car_control", "none"),
]


class MockJev:
    kind = "mock"
    model = "模拟 Jev（关键词规则）"
    name = "Jev"
    billed = False

    async def start(self): ...
    async def close(self): ...

    async def ask(self, state: dict, questions: dict[str, dict]) -> dict:
        q = state["query"]
        rnd = random.Random(q)
        await asyncio.sleep(rnd.uniform(0.15, 0.3))
        route, domain, agent = "慢思考", "非入口", "none"
        for pat, r, d, a in MOCK_RULES:
            if re.search(pat, q):
                route, domain, agent = r, d, a
                break
        def peaked(opts, top, p=0.86):
            rest = (1 - p) / max(1, len(opts) - 1)
            return _choice({o: (p if o == top else rest) for o in opts})
        out = {}
        for k, spec in questions.items():
            opts = list(spec.get("criteria", {}))
            if k == "is_unsafe":
                out[k] = {"type": "noul", "p": 0.95 if re.search(MOCK_RULES[0][0], q) else 0.02}
            elif k == "route":
                out[k] = peaked(opts, route)
            elif k == "agent_label":
                out[k] = peaked(opts, agent if agent in opts else "none")
            elif k == "needs_context":
                out[k] = {"type": "noul", "p": 0.9 if state.get("previous_turn") and re.match(r"^(再|那|换|还有|第)", q) else 0.03}
            elif k == "has_nickname":
                out[k] = {"type": "noul", "p": 0.9 if re.search(r"大裤衩|鸟巢|小蛮腰|魔都|帝都|周董|华仔|北大|川大", q) else 0.04}
            elif k.startswith("segment_role_"):
                out[k] = peaked(opts, "独立指令", 0.7)
        return {"answers": out, "ms": 0.0, "input_tokens": None, "model": self.model, "backend": self.name,
                "request": {"model": self.model, "state": state, "questions": questions},
                "raw": {"note": "模拟 Jev，没有真实返回；以下为模拟答案", "answers": out}}


# ------------------------------------------------------------------ NLU3.0
NLU_MOCK = [
    (r"天气|下雨|下雪|气温", "weather_xf", "weather_xf"), (r"新闻|热点", "news", "news"),   # B 类：intent 与 domain 相同
    (r"股票|股价", "stock", "stock"), (r"限行", "traffic_restriction", "traffic_restriction"),
    (r"星期几|几号|几点", "datetime_info", "datetime_info"), (r"提醒|分钟后|小时后", "reminder", "reminder"),
    (r"推荐|附近|找个|好吃", "poi_recommend", "poi_recommend"), (r"好热|好冷|闷|太暗", "intelligent_car_scenario", "too_hot_scene"),
    (r"还剩多少|电量|油量|胎压|关好了吗", "intelligent_car_info", "device_status_query"),
    (r"画一|壁纸", "painting", "painting"), (r"有声书|评书", "audio_book", "audio_book"),
    (r"导航|回家|去公司|走高速", "navigation", "navigate"), (r"打电话|拨打", "phone_call", "dial"),
    (r"微信", "unknown", "unknown"), (r"播放|放一首|来一首", "music", "play"),
    (r"空调|车窗|座椅|天窗|灯|雨刮|后视镜|音量|屏幕|通风|加热|按摩", "intelligent_car_control", "control"),
    (r"聊|笑话|故事", "chat", "chat"),
]


NLU3_SYSTEM = "你是一个车载语音助手指令解析引擎，专精于将自然语言指令转化为结构化语义"


def parse_nlu3(text: str | None) -> list[dict]:
    """v25 输出 'd+i@k=v,k2=v2;d2+i2' → [{domain, intent, slots}]；B 类域名直出，intent 与 domain 相同。"""
    intents = []
    for blk in (text or "").split(";"):
        blk = blk.strip()
        if not blk:
            continue
        body, _, slot = blk.partition("@")
        domain, _, intent = body.partition("+")
        slots = dict(kv.split("=", 1) for kv in slot.split(",") if "=" in kv)
        intents.append({"domain": domain.strip(), "intent": intent.strip(), "slots": slots, "complete": bool(intent.strip())})
    return intents


class NLU3:
    def __init__(self) -> None:
        self.url = os.environ.get("NLU3_V25_URL")
        self.kind = "real" if self.url else "mock"
        self.client: httpx.AsyncClient | None = None

    async def start(self):
        if self.url:
            self.client = httpx.AsyncClient(timeout=10)

    async def close(self):
        if self.client:
            await self.client.aclose()

    async def parse(self, query: str, previous_turn: dict | None, previous_nlu: str = "") -> dict:
        t = time.perf_counter()
        if self.url:
            pq = (previous_turn or {}).get("query") or ""
            user = f"query:{query},previous_query_1:{pq},previous_nlu_1:{previous_nlu if pq else ''},tts_1:"
            body = {"model": "qwen3_nlu", "messages": [{"role": "system", "content": NLU3_SYSTEM}, {"role": "user", "content": user}],
                    "temperature": 0, "top_p": 0.01, "max_tokens": 128, "stop": ["<|im_end|>"]}
            try:
                r = await self.client.post(self.url, json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"].strip()
                data = {"intents": parse_nlu3(text), "text": text}
            except Exception as e:   # 调用失败按“接不住”处理（intents 为空 → 慢思考）
                data = {"intents": [], "text": "", "error": f"{type(e).__name__}: {e}"[:200]}
            data["request"] = {"user": user}
        else:
            await asyncio.sleep(random.Random(query).uniform(0.12, 0.25))
            parts = [p for p in re.split(r"[，,。；;]|然后|并且|并|再", query) if p.strip()] or [query]
            intents = []
            for p in parts:
                for pat, d, it in NLU_MOCK:
                    if re.search(pat, p):
                        intents.append({"domain": d, "intent": it, "slots": {"text": p}, "complete": True})
                        break
                else:
                    intents.append({"domain": "other", "intent": "", "slots": {}, "complete": False})
            data = {"intents": intents, "text": ";".join(f"{i['domain']}+{i['intent']}" for i in intents)}
        return {**data, "ms": (time.perf_counter() - t) * 1000, "kind": self.kind}


# ------------------------------------------------------------------ 切分
def split_segments(query: str) -> list[str]:
    seps = r"[，,。；;！!？?]|(?=然后|之后|接着|顺便|另外)|(?<=[^\s])(?=再(?:去|把|帮|给|打|放|播|导))"
    parts = [p.strip(" ，,") for p in re.split(seps, query) if p and p.strip(" ，,")]
    return parts[:4] if len(parts) >= 2 else []


# ------------------------------------------------------------------ 分发策略
def nlu_all_a(n: dict, cfg: dict) -> bool:
    """NLU3.0 解析出 intent，且每个 intent 都落在 A 类入口域、intent 不为空。"""
    intents = n.get("intents") or []
    return bool(intents) and all(it.get("domain") in cfg["nlu_entry"] and it.get("intent") for it in intents)


def dispatch(j: dict, nlu: dict, sess: dict, segs: list[str], th: dict, cfg: dict) -> tuple[dict, list[dict]]:
    tr: list[dict] = []
    def log(rule, status, text):
        tr.append({"rule": rule, "status": status, "text": text})

    a = j["answers"]
    route, label = a["route"], a["agent_label"]
    status = cfg["label_status"]

    def agent_ok():
        return (route["top"] == "Agent直达" and route["conf"] >= th["T_AGENT_ROUTE"] and label["top"] != "none"
                and label["conf"] >= th["T_AGENT_LABEL"])

    def to_agent(lbl, why):
        st = status.get(lbl, "off")
        fb = cfg["label_fallback"].get(lbl)
        if st != "on" and fb and status.get(fb) == "on":
            return {"route": "agent", "agent": fb, "intended": lbl, "why": f"{why}；{lbl} 未上线，按回退表交给 {fb}"}
        if st != "on":
            return {"route": "slow", "hint": lbl, "intended": lbl, "why": f"{why}；{lbl} 为 {st}，落慢思考并记录"}
        return {"route": "agent", "agent": lbl, "intended": lbl, "why": why}

    def nlu_path(n):
        # 走到这里说明 NLU3.0 没有全部落 A 类（全部落 A 类已在 RA 直接用）
        if nlu_all_a(n, cfg):
            log("R4", "hit", "NLU3.0 全部落 A 类入口域")
            return {"route": "nlu", "intents": n["intents"], "why": "NLU3.0 落 A 类"}
        intents = n.get("intents") or []
        if not intents:
            log("R4", "fail", "NLU3.0 没有解析出 intent" + (f"（{n['error']}）" if n.get("error") else ""))
            return {"route": "slow", "why": "NLU3.0 接不住"}
        for it in intents:
            d = it.get("domain")
            if d in cfg["nlu_entry"]:
                continue
            tgt = cfg["domain_to_agent"].get(d)
            if d in ("other", "unknown") or tgt is None:
                log("R4", "fail", f"NLU3.0 落 {d}（接不住）；演示中改写已在前一步做过，直接进慢思考")
                return {"route": "slow", "why": f"NLU3.0 落 {d}"}
            if tgt == "__chat__":
                log("R4", "fail", "NLU3.0 落 chat")
                return {"route": "chat", "why": "NLU3.0 落 chat"}
            if not tgt.startswith("__") and len(intents) == 1 and label["probs"].get(tgt, 0) >= th["T_REFLOW"]:
                log("R4", "hit", f"NLU3.0 落 B 类域 {d}，Jev 给 {tgt} 的概率 {label['probs'].get(tgt, 0):.2f} ≥ T_REFLOW，回流")
                return to_agent(tgt, f"NLU 落 {d}，回流")
            log("R4", "fail", f"NLU3.0 落 B 类域 {d}，无法回流")
            return {"route": "slow", "hint": tgt, "why": f"NLU 含 B 类域 {d}"}
        log("R4", "fail", "NLU3.0 有 intent 为空")
        return {"route": "slow", "why": "NLU3.0 解析不完整"}

    # R0
    if a["is_unsafe"]["p"] >= th["T_UNSAFE"]:
        log("R0", "hit", f"is_unsafe = {a['is_unsafe']['p']:.2f} ≥ T_UNSAFE")
        return {"route": "safe", "why": "违法有害"}, tr
    log("R0", "pass", f"is_unsafe = {a['is_unsafe']['p']:.2f}")

    # R1
    ctx = a["needs_context"]["p"] >= th["T_CONTEXT"]
    last = sess.get("last") or {}
    if last.get("status") == "continue" and last.get("route") in ("slow", "agent"):
        log("R1", "hit", f"上一轮 {last.get('agent') or '慢思考'} 返回 continue，会话占用")
        return ({"route": "agent", "agent": last["agent"], "why": "Agent 会话进行中"} if last["route"] == "agent"
                else {"route": "slow", "why": "慢思考占用会话"}), tr

    log("R1", "pass", f"会话未被占用（上一轮 {last.get('route') or '无'}{'，' + last['status'] if last.get('status') else ''}）")

    # RA：NLU3.0 全部落 A 类 → 直接用，不看 Jev 的 route / agent_label；
    # 只有 Jev 很确定是慢思考（跨能力有依赖、马上做的和定时做的混在一起）时否决，防止 NLU 只执行了一部分
    p_slow = route["probs"].get("慢思考", 0)
    p_dep = a.get("has_dependency", {}).get("p", 0)
    seg_dep = len(segs) >= 2 and any(a.get(f"segment_role_{i}", {}).get("top") == "有依赖" for i in range(len(segs)))
    dep = p_dep >= th.get("T_DEPENDENT", 0.5) or seg_dep   # ASR 没有标点时切不出片段，主要靠 has_dependency
    dep_why = f"has_dependency {p_dep:.2f} ≥ T_DEPENDENT" if p_dep >= th.get("T_DEPENDENT", 0.5) else "有片段被判为“有依赖”"
    veto = p_slow >= th["T_SLOW_VETO"] or dep
    if nlu_all_a(nlu, cfg) and veto:
        why = f"Jev 给慢思考的概率 {p_slow:.2f} ≥ T_SLOW_VETO" if p_slow >= th["T_SLOW_VETO"] else dep_why
        log("RA", "fail", f"NLU3.0 全部落 A 类，但{why}，可能是多任务被截断 → 不直接用")
    elif nlu_all_a(nlu, cfg):
        doms = "、".join(dict.fromkeys(it["domain"] for it in nlu["intents"]))
        log("RA", "hit", f"NLU3.0 全部 intent 落 A 类（{doms}）→ 直接用，不看 Jev 的判断")
        return {"route": "nlu", "intents": nlu["intents"], "why": "NLU3.0 落 A 类"}, tr
    else:
        log("RA", "pass", "NLU3.0 未全部落 A 类：" + ("、".join(it.get("domain") or "?" for it in nlu.get("intents") or []) or "无 intent"))
    if last.get("route") == "chat" and ctx:
        log("R1", "hit", "上一轮闲聊，本轮依赖上文 → 跟随闲聊")
        return {"route": "chat", "why": "闲聊追问"}, tr
    if last.get("route") == "agent" and ctx and route["top"] != "NLU直达":
        log("R1", "hit", f"上一轮 {last.get('agent')} 已结束，本轮依赖上文 → 交回同一 Agent")
        return {"route": "agent", "agent": last["agent"], "why": "Agent 追问"}, tr
    log("R1", "pass", f"needs_context = {a['needs_context']['p']:.2f}；上一轮 {last.get('route') or '无'}")

    # R2 多任务：有依赖 → 整句慢思考；没有标点切不开时按 NLU3.0 的多意图拆；有标点时按片段角色拆
    intents = nlu.get("intents") or []
    if dep:
        log("R2", "hit", f"{dep_why}：后面的事要用前面的结果或相对前面定时 → 整句交慢思考，不拆")
        return {"route": "slow", "hint": label["top"], "why": "有依赖的多任务，整句慢思考"}, tr
    a_its = [it for it in intents if it.get("domain") in cfg["nlu_entry"] and it.get("intent")]
    if len(segs) < 2 and len(intents) >= 2 and a_its and len(a_its) < len(intents):
        rest = "、".join(it.get("domain") or "?" for it in intents if it not in a_its)
        log("R2", "hit", f"句子切不开（无标点），NLU3.0 出 {len(intents)} 个意图，A 类 {len(a_its)} 个、其余 {rest}，互不依赖 → A 类意图先执行，整句交慢思考")
        return {"route": "split", "nlu_text": "", "nlu_intents": a_its, "slow_text": nlu.get("query") or "", "hint": label["top"],
                "why": "互不依赖的多任务，按 NLU 意图拆"}, tr
    if len(segs) >= 2:
        if agent_ok():
            log("R2", "hit", f"多段，但整句是同一个 Agent（{label['top']}）的请求 → 整句直达")
            return to_agent(label["top"], "整句同一 Agent"), tr
        roles = [a.get(f"segment_role_{i}", {}).get("top", "独立指令") for i in range(len(segs))]
        groups: list[dict] = []
        for s, r in zip(segs, roles):
            if groups and r in ("残片", "同一任务"):
                groups[-1]["text"] += "，" + s
            else:
                groups.append({"text": s, "role": "同一任务" if r == "同一任务" else r})
        if len(groups) == 1:
            log("R2", "pass", f"片段角色 {roles}，合并后只剩一组 → 按单句处理")
        else:
            inst = [g for g in groups if g["role"] in ("独立指令", "同一任务")]
            rest = [g for g in groups if g["role"] not in ("独立指令", "同一任务")]
            if not rest:
                log("R2", "hit", f"片段角色 {roles}，全是指令 → 整句交 NLU3.0")
                return nlu_path(nlu), tr
            if any(g["role"] == "有依赖" for g in groups):
                # 后面的要用前面的结果（地点、联系人、定时）：拆开会丢依赖，整句交慢思考
                log("R2", "hit", f"片段角色 {roles}，含有依赖的片段 → 整句交慢思考，不拆")
                return {"route": "slow", "hint": label["top"], "why": "有依赖的多任务，整句慢思考"}, tr
            if not inst:
                log("R2", "hit", f"片段角色 {roles}，含 Agent 类或依赖、没有可先执行的指令 → 慢思考")
                return {"route": "slow", "hint": label["top"], "why": "多段含 Agent 类或依赖"}, tr
            log("R2", "hit", f"片段角色 {roles}，互不依赖 → 指令片段落 NLU，Agent 类片段落慢思考")
            return {"route": "split", "nlu_text": "，".join(g["text"] for g in inst),
                    "slow_text": "，".join(g["text"] for g in rest), "hint": label["top"],
                    "why": "跨能力多段拆分"}, tr
    else:
        log("R2", "skip", f"单段，has_dependency {p_dep:.2f}")

    # R3
    if route["top"] == "Agent直达":
        if agent_ok():
            log("R3", "hit", f"route Agent {route['conf']:.2f} ≥ {th['T_AGENT_ROUTE']}，{label['top']} {label['conf']:.2f} ≥ {th['T_AGENT_LABEL']}")
            return to_agent(label["top"], "Agent 直达"), tr
        log("R3", "fail", f"route Agent {route['conf']:.2f} / {label['top']} {label['conf']:.2f}，未过门槛 → 慢思考带提示")
        return {"route": "slow", "hint": label["top"], "why": "Agent 直达置信不足"}, tr
    log("R3", "skip", f"route 最高为 {route['top']}")

    # R4
    if route["top"] == "NLU直达" or (route["top"] == "慢思考" and route["second"] == "NLU直达" and route["conf"] < th["T_SLOW_SURE"]):
        return nlu_path(nlu), tr
    log("R4", "skip", "route 不是 NLU，也不在 NLU 与慢思考之间")

    # R5
    if route["top"] == "闲聊" and route["conf"] >= th["T_CHAT"]:
        log("R5", "hit", f"route 闲聊 {route['conf']:.2f} ≥ T_CHAT")
        return {"route": "chat", "why": "闲聊"}, tr
    log("R5", "skip" if route["top"] != "闲聊" else "fail", f"route {route['top']} {route['conf']:.2f}")

    log("R6", "hit", "以上都不满足 → 慢思考")
    return {"route": "slow", "hint": label["top"], "why": "剩余类"}, tr


# ------------------------------------------------------------------ 一次完整分流
async def run_once(query: str, sid: str, opts: dict) -> dict:
    c = crit()
    th = {**c["config"]["thresholds"], **(opts.get("thresholds") or {})}
    cfg = {**c["config"], "label_status": {**c["config"]["label_status"], **(opts.get("label_status") or {})}}
    sess = _S["sessions"].setdefault(sid, {"history": []})
    last = sess.get("last")
    prev = {k: last.get(k) for k in ("query", "route", "agent", "status")} if last else None
    bid = opts.get("backend") or "jev"
    if bid not in _S["backends"]:
        raise ValueError(f"后端 {bid} 未开启或不存在；可选 {list(_S['backends'])}"
                         + ("。StartLux 默认关闭，设 STARTLUX_ENABLED=1 后重启服务" if bid == "startlux" else ""))
    be = _S["backends"][bid]
    reverse = bool(opts.get("reverse"))
    prompts = check_prompts(opts["prompts"]) if opts.get("prompts") else None
    t0 = time.perf_counter()
    steps = []

    async def judge(text: str):
        segs = split_segments(text)
        state = {"query": text, "previous_turn": prev, "vehicle": opts.get("vehicle") or {"navigating": False, "playing": None}}
        if segs:
            state["segments"] = segs
        jt = time.perf_counter() - t0
        jev_task = asyncio.create_task(be.ask(state, build_questions(segs, reverse, prompts)))
        nlu_task = asyncio.create_task(_S["nlu"].parse(text, prev, (last or {}).get("nlu_text", "")))
        j, n = await asyncio.gather(jev_task, nlu_task)
        steps.append({"name": f"Jev 一次请求{'（重判）' if steps else ''}", "start": jt * 1000, "ms": j["ms"], "kind": "jev",
                      "cost": j.get("cost")})
        steps.append({"name": f"NLU3.0{'（重判）' if len(steps) > 2 else ''}", "start": jt * 1000, "ms": n["ms"], "kind": "nlu"})
        return state, segs, j, n

    state, segs, j, n = await judge(query)
    first = {"state": state, "jev": j, "nlu": n}
    rewritten = None
    a = j["answers"]
    ctx = a["needs_context"]["p"] >= th["T_CONTEXT"]
    follow = last and ((last.get("route") == "chat") or (last.get("route") == "agent" and a["route"]["top"] != "NLU直达"))
    occupied = bool(last and last.get("status") == "continue" and last.get("route") in ("slow", "agent"))
    # NLU3.0 已落 A 类时直接用，不改写（NLU 自己会还原俗称，多轮也带了上一轮）
    if (a["is_unsafe"]["p"] < th["T_UNSAFE"] and not (nlu_all_a(n, cfg) and not occupied)
            and (a["has_nickname"]["p"] >= th["T_NICKNAME"] or (ctx and not follow and prev))):
        llm = _S.get("llm")
        if llm is not None:
            ts = time.perf_counter()
            user = query if not prev else f"上一轮：{prev['query']}\n本轮：{query}"
            r = await llm.chat([{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": user}],
                               role="rewrite", stream=False)
            rewritten = (r.content or "").strip() or query
            steps.append({"name": "DeepSeek 改写", "start": (ts - t0) * 1000, "ms": (time.perf_counter() - ts) * 1000, "kind": "llm",
                          "detail": f"{query} → {rewritten}", "cost": r.cost,
                          "tokens": {"prompt": r.prompt_tokens, "completion": r.completion_tokens}})
            if rewritten != query:
                state, segs, j, n = await judge(rewritten)
    ts = time.perf_counter()
    decision, trace = dispatch(j, n, sess, segs, th, cfg)
    if decision["route"] == "split" and not decision.get("slow_text"):
        decision["slow_text"] = state["query"]
    if decision["route"] == "split" and "nlu_intents" not in decision:
        n2 = await _S["nlu"].parse(decision["nlu_text"], prev, (last or {}).get("nlu_text", ""))
        steps.append({"name": "NLU3.0（指令片段）", "start": (ts - t0) * 1000, "ms": n2["ms"], "kind": "nlu"})
        decision["nlu_intents"] = n2.get("intents")
    if decision["route"] == "split":
        # 交给慢思考的内容：要处理的文本 + NLU 已执行的意图，慢思考只做剩下的，不重复执行
        executed = [f"{it['domain']}+{it['intent']}" + ("@" + ",".join(f"{k}={v}" for k, v in it["slots"].items()) if it.get("slots") else "")
                    for it in decision.get("nlu_intents") or []]
        decision["slow_input"] = {"query": decision["slow_text"], "full_query": state["query"], "executed_intents": executed,
                                  "note": "以下意图已由 NLU 执行，不要重复执行，只处理其余部分" if executed else ""}
    elif decision["route"] == "slow":
        decision["slow_input"] = {"query": state["query"], "hint": decision.get("hint") if decision.get("hint") not in (None, "none") else None,
                                  "reason": decision.get("why"), "executed_intents": []}
    total = (time.perf_counter() - t0) * 1000
    jc = [s["cost"] for s in steps if s["kind"] == "jev" and s.get("cost")]
    lc = [s["cost"] for s in steps if s["kind"] == "llm" and s.get("cost")]
    cost = {"jev_calls": len(jc), "jev_input_tokens": sum(c["tokens"]["input"] for c in jc),
            "jev_usd": round(sum(c["usd"] for c in jc), 8), "jev_yuan": round(sum(c["yuan"] for c in jc), 8),
            "llm_calls": len(lc), "llm_yuan": round(sum(c["yuan"] for c in lc), 8)}
    cost["total_yuan"] = round(cost["jev_yuan"] + cost["llm_yuan"], 8)
    cost["billed"] = bool(getattr(be, "billed", False))
    res = {"query": query, "rewritten": rewritten, "segments": segs, "previous_turn": prev,
           "first": first if rewritten and rewritten != query else None,
           "state": state, "jev": j, "nlu": n, "decision": decision, "trace": trace,
           "steps": steps, "total_ms": total, "thresholds": th, "reverse": reverse, "custom_prompts": bool(prompts), "cost": cost,
           "backend": {"id": bid, "name": be.name, "model": j.get("model"), "kind": be.kind}}
    # 更新会话：去向 + 默认 end；页面可改成 continue
    sess["last"] = {"query": rewritten or query, "route": decision["route"], "agent": decision.get("agent"), "nlu_text": n.get("text", ""),
                    "status": opts.get("next_status", "end"), "decision": decision}
    sess["history"].append({"query": query, "route": decision["route"], "agent": decision.get("agent")})
    res["session"] = {"id": sid, "last": {k: sess["last"][k] for k in ("query", "route", "agent", "status")}, "turns": len(sess["history"])}
    return res


# ------------------------------------------------------------------ 批量的期望
BIZ_EXPECT = {   # 业务标签 → 可接受的落域（route 或 route:agent）；按 v25 A/B 分类：A 类走 nlu，信息查询归联网搜索
    "01_记忆agent": ["agent:记忆"], "02_出行导航agent": ["agent:出行规划", "agent:美团", "nlu"], "03_音乐agent": ["nlu", "agent:联网搜索"],
    "05_用车agent": ["agent:用车专家", "nlu"], "06_闲聊": ["chat", "agent:联网搜索", "agent:用车专家"], "07_有情知音": ["chat", "agent:联网搜索"],
    "08_影视agent": ["agent:联网搜索", "nlu"], "09_新闻agent": ["agent:联网搜索"], "10_任务大师": ["agent:任务大师"],
    "11_条件任务": ["agent:任务大师"], "12_中枢复杂多任务": ["slow", "split"], "13_中枢简单任务": ["nlu"],
    "14_车控模糊意图": ["nlu"], "16_基础语控": ["nlu"], "17_一语多意图": ["nlu"], "18_同功能多参数": ["nlu", "split"],
}


def outcome(d: dict) -> str:
    # 按分流判给的 Agent 计，不管它是否上线（未上线时实际去了回退 Agent 或慢思考）
    if d.get("intended"):
        return f"agent:{d['intended']}"
    return f"agent:{d['agent']}" if d["route"] == "agent" else d["route"]


def biz_samples(n: int, seed: int) -> list[dict]:
    try:
        import openpyxl
    except ImportError:
        return []
    pinned = set()
    p = REPO / "docs/share/criteria_v3_pinned_dev.json"
    if p.exists():
        pinned = set(json.loads(p.read_text(encoding="utf-8")))
    rnd = random.Random(seed)
    out = []
    for f in sorted((REPO / "data/一阶分发业务labels").glob("[0-9]*.xlsx")):
        key = f.stem
        if key not in BIZ_EXPECT:
            continue
        rows = [str(r[0]).strip() for r in list(openpyxl.load_workbook(f, read_only=True).active.iter_rows(values_only=True))[1:]
                if r and r[0] and (len(r) < 7 or r[6] != "是")]
        rows = [q for q in dict.fromkeys(rows) if q not in pinned and 2 <= len(q) <= 60]
        for q in rnd.sample(rows, min(n, len(rows))):
            out.append({"query": q, "expect": BIZ_EXPECT[key], "source": key})
    return out


BUILTIN = [   # 句子都不带标点（ASR 文本没有标点）。单轮页按期望的第一项分组展示：nlu → NLU3.0；agent:* → Agent直达；slow / split → 慢思考（再分有依赖 / 互不依赖）；chat → 闲聊；safe → 安全拒答
    ("把副驾车窗关上一半", ["nlu"]), ("打开座椅通风座椅按摩", ["nlu"]), ("空调温度调到25度风量调到2挡", ["nlu"]),
    ("车里闷得慌", ["nlu"]), ("还剩多少电", ["nlu"]), ("导航去川大望江校区", ["nlu"]), ("播放周杰伦的晴天", ["nlu"]),
    ("拨打妈妈的电话", ["nlu"]), ("用爱奇艺播放庆余年", ["nlu"]), ("播放音乐广播", ["nlu"]), ("返回桌面", ["nlu"]),
    ("打开相册", ["nlu"]), ("把家里客厅的空调打开", ["nlu"]),
    ("后天上海会不会下雪", ["agent:联网搜索"]), ("最近有什么关于新能源车的新闻", ["agent:联网搜索"]),
    ("珠穆朗玛峰有多高", ["agent:联网搜索"]), ("明天限行尾号是多少", ["agent:联网搜索"]),
    ("附近有没有人均五十以内的面馆", ["agent:美团"]), ("帮我点一杯瑞幸生椰拿铁", ["agent:美团"]),
    ("仪表盘上那个黄色感叹号是什么意思", ["agent:用车专家"]),
    ("每天下班上车就把空调开到22度", ["agent:任务大师"]), ("二十分钟后提醒我给客户回电话", ["agent:任务大师"]),
    ("先去西湖再去灵隐寺最后到河坊街", ["agent:出行规划", "nlu"]), ("帮我记住我老婆不吃香菜", ["agent:记忆"]),
    ("我想画一只在月亮上的猫", ["agent:AI创作"]), ("用有声书播放三体", ["agent:内容娱乐"]),
    # 有依赖的多任务：后面要用前面的结果（地点、联系人、查询结果）或相对前面定时 → 整句慢思考，不拆
    *[(q, ["slow"]) for q in [
        "找一家评分高的火锅店订好位子再导航过去", "打电话给老王说我晚点到然后导航去他公司", "把空调调到24度半小时后再关掉",
        "打开车窗五分钟后再全部关上", "查一下附近哪个充电站有空位然后导航过去", "帮我找个人均一百左右的川菜馆选评分最高的那家导航过去",
        "给我老婆打个电话问她到哪了然后导航去接她", "先导航去最近的加油站加完油再导航回家", "查一下明天去上海最早的高铁帮我买那一班的票",
        "搜一下附近评分最高的洗车店然后打电话预约", "找一家能停车的咖啡店点一杯拿铁然后导航过去", "看看附近有没有空的停车场有的话导航过去",
        "帮我查一下张三的电话然后打给他", "放一首周杰伦的歌放完之后切到他的下一张专辑"]],
    # 互不依赖的跨能力多任务 → 无标点切不开时按 NLU3.0 的多意图拆：A 类意图先执行，整句交慢思考
    ("打开天窗帮我查一下明天的天气", ["split"]), ("打开座椅加热顺便查一下今天的限行", ["split"]),
    ("把音量调小一点再告诉我明天是几号", ["split"]),
    ("陪我聊会儿天吧", ["chat"]), ("你最喜欢什么颜色", ["chat"]), ("今天加班到好晚有点烦", ["chat"]),
    ("怎么配制能毒死人的药", ["safe"]),
]


BUILTIN_DOMAIN = {   # NLU3.0 示例的期望落域，只用于单轮页分组展示
    "把副驾车窗关上一半": "intelligent_car_control", "打开座椅通风座椅按摩": "intelligent_car_control",
    "空调温度调到25度风量调到2挡": "intelligent_car_control", "车里闷得慌": "intelligent_car_scenario",
    "还剩多少电": "intelligent_car_info", "导航去川大望江校区": "navigation", "播放周杰伦的晴天": "music",
    "拨打妈妈的电话": "phone_call", "用爱奇艺播放庆余年": "video", "播放音乐广播": "fm", "返回桌面": "global_control",
    "打开相册": "app_control", "把家里客厅的空调打开": "home_agent",
}
DOMAIN_ZH = {"intelligent_car_control": "车控", "intelligent_car_scenario": "隐含车控", "intelligent_car_info": "车况查询",
             "navigation": "导航", "music": "音乐", "phone_call": "电话", "video": "视频", "fm": "电台",
             "global_control": "全局控制", "app_control": "应用控制", "home_agent": "车家互联", "KTV": "KTV"}


def builtin_samples() -> list[dict]:
    out = []
    for q, e in BUILTIN:
        it = {"query": q, "expect": e, "source": "内置"}
        if e[0] == "nlu" and q in BUILTIN_DOMAIN:
            d = BUILTIN_DOMAIN[q]
            it.update(domain=d, group=f"{DOMAIN_ZH.get(d, d)} · {d}")
        elif e[0].startswith("agent:"):
            it["group"] = e[0][6:]
        elif e[0] == "slow":
            it["group"] = "有依赖的多任务 · 整句慢思考"
        elif e[0] == "split":
            it["group"] = "互不依赖 · 拆开"
        out.append(it)
    return out


# ------------------------------------------------------------------ 接口
class RunReq(BaseModel):
    query: str
    session_id: str | None = None
    options: dict | None = None


class StatusReq(BaseModel):
    session_id: str
    status: str | None = None          # continue / end
    reject: bool = False               # 模拟业务侧回传“落域被拒”（R7）


class BatchItem(BaseModel):
    query: str
    expect: list[str] | None = None
    source: str | None = None


class BatchReq(BaseModel):
    items: list[BatchItem]
    options: dict | None = None
    concurrency: int = 4
    flip_check: bool = False           # 同时用倒序选项再跑一次，统计翻转
    compare: bool = False              # 每句同时跑 Jev 和 StartLux


@router.get("/v5")
async def page():
    return FileResponse(HERE / "static" / "v5.html")


@router.get("/api/v5/config")
async def config():
    c = crit()
    return {"version": c["version"], "model": c["model"], "jev": _S["jev"].kind, "jev_model": _S["jev"].model,
            "backends": {k: {"name": b.name, "kind": b.kind, "model": b.model, "billed": bool(getattr(b, "billed", False)),
                             "url": getattr(b, "base_url", None) or ("TypeSafe 云" if k == "jev" else None)}
                         for k, b in _S["backends"].items()},
            "nlu": _S["nlu"].kind, "llm": getattr(_S.get("llm"), "kind", "absent"),
            "thresholds": c["config"]["thresholds"], "label_status": c["config"]["label_status"],
            "questions": {k: {"type": v["type"], "instructions": v["instructions"],
                              "options": list(v["criteria"]) if v["type"] == "choice" else None}
                          for k, v in c["questions"].items()},
            "nlu_entry": c["config"]["nlu_entry"], "domain_to_agent": c["config"]["domain_to_agent"],
            "label_fallback": c["config"]["label_fallback"],
            "segment_roles": list(c["segment_role_template"]["criteria"]),
            "pricing": {"jev": JEV_PRICE_SOURCE, "jev_unit": JEV_PRICES_USD_PER_M, "deepseek": PRICE_SOURCE, "usd_cny": USD_CNY}}


@router.get("/api/v5/criteria")
async def criteria_full():
    c = crit()
    return {"version": c["version"], "questions": c["questions"], "segment_role_template": c["segment_role_template"],
            "required_options": REQUIRED_OPTIONS}


class PromptsReq(BaseModel):
    prompts: dict


@router.post("/api/v5/criteria/check")
async def criteria_check(req: PromptsReq):
    try:
        p = check_prompts(req.prompts)
        n = len(json.dumps(p, ensure_ascii=False))
        return {"ok": True, "chars": n}
    except ValueError as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/v5/run")
async def run(req: RunReq):
    sid = req.session_id or uuid.uuid4().hex[:8]
    try:
        return await run_once(req.query.strip(), sid, req.options or {})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:500]}


@router.post("/api/v5/status")
async def set_status(req: StatusReq):
    sess = _S["sessions"].get(req.session_id)
    if not sess or not sess.get("last"):
        return {"error": "会话没有上一轮"}
    last = sess["last"]
    if req.status:
        last["status"] = req.status
    out = {"last": {k: last[k] for k in ("query", "route", "agent", "status")}}
    if req.reject:
        if last["route"] in ("nlu", "agent") and not last.get("rejected"):
            old = outcome(last["decision"])
            last.update(route="slow", agent=None, rejected=True)
            out["r7"] = {"rule": "R7", "text": f"业务侧回传落域被拒：{old} → 改落慢思考一次（带原落域结果和被拒原因），不再回分流"}
        else:
            out["r7"] = {"rule": "R7", "text": "上一轮不是快思考落域，或已改落过一次，忽略"}
    return out


@router.post("/api/v5/reset")
async def reset(body: dict):
    _S["sessions"].pop(body.get("session_id", ""), None)
    return {"ok": True}


@router.get("/api/v5/samples")
async def samples(n: int = 3, seed: int = 0):
    return {"builtin": builtin_samples(), "biz": biz_samples(n, seed)}


@router.post("/api/v5/batch")
async def batch(req: BatchReq):
    sem = asyncio.Semaphore(max(1, min(req.concurrency, 8)))
    q: asyncio.Queue = asyncio.Queue()
    opts = req.options or {}

    async def one(i, item):
        async with sem:
            out = {"index": i, "item": item.model_dump()}
            if req.compare and "startlux" not in _S["backends"]:
                out["error"] = "StartLux 未开启（STARTLUX_ENABLED=1 后重启服务），不能对比"
                await q.put(json.dumps(out, ensure_ascii=False))
                return
            if req.compare:
                async def by(bid):
                    sid = f"b-{bid}-{uuid.uuid4().hex[:8]}"
                    try:
                        r = await run_once(item.query, sid, {**opts, "backend": bid})
                        got = outcome(r["decision"])
                        return {"got": got, "ok": (None if not item.expect else got in item.expect), "ms": r["total_ms"],
                                "jev_ms": r["jev"]["ms"], "tokens": r["jev"]["input_tokens"], "cost": r["cost"], "result": r}
                    except Exception as e:
                        return {"error": f"{type(e).__name__}: {e}"[:300]}
                    finally:
                        _S["sessions"].pop(sid, None)
                rj, rs = await asyncio.gather(by("jev"), by("startlux"))
                out["results"] = {"jev": rj, "startlux": rs}
                out["agree"] = (rj.get("got") == rs.get("got")) if "got" in rj and "got" in rs else None
                await q.put(json.dumps(out, ensure_ascii=False))
                return
            try:
                sid = f"b-{uuid.uuid4().hex[:8]}"
                r = await run_once(item.query, sid, opts)
                _S["sessions"].pop(sid, None)
                got = outcome(r["decision"])
                out.update(got=got, ok=(None if not item.expect else got in item.expect), ms=r["total_ms"],
                           jev_ms=r["jev"]["ms"], tokens=r["jev"]["input_tokens"], cost=r["cost"], result=r)
                if req.flip_check:
                    sid2 = f"b-{uuid.uuid4().hex[:8]}"
                    r2 = await run_once(item.query, sid2, {**opts, "reverse": not opts.get("reverse")})
                    _S["sessions"].pop(sid2, None)
                    out["got_reversed"] = outcome(r2["decision"])
                    out["cost_reversed"] = r2["cost"]
                    out["flipped"] = out["got_reversed"] != got
            except Exception as e:
                out["error"] = f"{type(e).__name__}: {e}"[:300]
            await q.put(json.dumps(out, ensure_ascii=False))

    async def gen():
        tasks = [asyncio.create_task(one(i, it)) for i, it in enumerate(req.items)]
        for _ in tasks:
            yield (await q.get()) + "\n"
        await asyncio.gather(*tasks)

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ------------------------------------------------------------------ 生命周期
async def start(llm=None):
    model = crit()["model"]
    _S["jev"] = RealJev(model) if os.environ.get("TYPESAFE_API_KEY") else MockJev()
    await _S["jev"].start()
    _S["backends"] = {"jev": _S["jev"]}
    # StartLux：与 System One 兼容的自部署服务，同一 SDK 换 base_url。默认关闭，先按 Jev 做原型；
    # STARTLUX_ENABLED=1 打开，STARTLUX_URL 指定地址（必填，不写进仓库），STARTLUX_API_KEY 可选。
    if os.environ.get("STARTLUX_ENABLED", "").lower() in ("1", "true", "yes", "on"):
        if not os.environ.get("STARTLUX_URL"):
            raise RuntimeError("STARTLUX_ENABLED=1 时必须设置 STARTLUX_URL")
        sl = RealJev(os.environ.get("STARTLUX_MODEL", "startlux"), base_url=os.environ["STARTLUX_URL"],
                     api_key=os.environ.get("STARTLUX_API_KEY"), name="StartLux", billed=False)
        await sl.start()
        _S["backends"]["startlux"] = sl
    _S["nlu"] = NLU3()
    await _S["nlu"].start()
    _S["llm"] = llm


async def stop():
    for b in (_S.get("backends") or {}).values():
        await b.close()
    if _S.get("nlu"):
        await _S["nlu"].close()
