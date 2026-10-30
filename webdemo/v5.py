"""v5 分流测试页的后端：按《Jev 分流详细设计 v2》实现一次判定 + 分发策略 R0–R7。

    Jev 一次请求（criteria_v3，7 道题）∥ NLU3.0 整句解析 → Query 整理（需要时）→ 分发策略 → 落域

- 判据读 docs/share/criteria_v3.json（由 docs/share/make_criteria_v3.py 生成）。
- Jev：有 TYPESAFE_API_KEY 时真实调用（模型取判据里的版本），否则用关键词模拟。
- NLU3.0：设置 NLU3_URL 时 POST {"query", "previous_turn"}，期望返回 {"intents": [{"domain", "intent", "slots", "complete"}]}；
  不设时用关键词模拟，页面上标明“模拟”。
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
    "domain": ["非入口"],
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
    (r"新闻|热点|资讯", "Agent直达", "非入口", "新闻"),
    (r"天气|下雨|下雪|气温|空气质量|洗车", "Agent直达", "非入口", "天气"),
    (r"(\d+|[一二三四五六七八九十半两]+)(分钟|小时)(后|以后)|每天|每周|每当|当.{1,12}(时|的时候)|提醒我", "Agent直达", "非入口", "任务大师"),
    (r"记住|记一下|存一下|我常听|我喜欢的", "Agent直达", "非入口", "记忆"),
    (r"怎么|为什么|是什么|在哪|有没有|吗$", "Agent直达", "非入口", "用车专家"),
    (r"推荐|找个|附近|先去|再去|行程", "Agent直达", "非入口", "出行导航"),
    (r"好热|好冷|太闷|晕", "Agent直达", "非入口", "自适应车控"),
    (r"聊|笑话|故事|你觉得", "闲聊", "非入口", "none"),
    (r"导航|回家|去公司", "NLU直达", "navigation", "none"),
    (r"播放|放一首|来一首|听", "NLU直达", "music", "none"),
    (r"打电话|拨打", "NLU直达", "phone_call", "none"),
    (r"微信", "NLU直达", "wechat_on_car", "none"),
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
            elif k == "domain":
                out[k] = peaked(opts, domain if domain in opts else "非入口")
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
    (r"天气|下雨|下雪|气温", "weather_xf", "query_weather"), (r"新闻|热点", "news", "play_news"),
    (r"股票|股价", "stock", "query_stock"), (r"限行", "traffic_restriction", "query"),
    (r"星期几|几号|几点", "datetime_info", "query"), (r"提醒|分钟后|小时后", "reminder", "create"),
    (r"推荐|附近|找个|好吃", "poi_recommend", "search"), (r"好热|好冷|太闷", "intelligent_car_scenario", "adjust"),
    (r"导航|回家|去公司|走高速", "navigation", "navigate"), (r"打电话|拨打", "phone_call", "dial"),
    (r"微信", "wechat_on_car", "send"), (r"播放|放一首|来一首", "music", "play"),
    (r"空调|车窗|座椅|天窗|灯|雨刮|后视镜|音量|屏幕|通风|加热|按摩", "intelligent_car_control", "control"),
    (r"聊|笑话|故事", "chat", "chat"),
]


class NLU3:
    def __init__(self) -> None:
        self.url = os.environ.get("NLU3_URL")
        self.kind = "real" if self.url else "mock"

    async def parse(self, query: str, previous_turn: dict | None) -> dict:
        t = time.perf_counter()
        if self.url:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.post(self.url, json={"query": query, "previous_turn": previous_turn})
                r.raise_for_status()
                data = r.json()
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
            data = {"intents": intents}
        return {**data, "ms": (time.perf_counter() - t) * 1000, "kind": self.kind}


# ------------------------------------------------------------------ 切分
def split_segments(query: str) -> list[str]:
    seps = r"[，,。；;！!？?]|(?=然后|之后|接着|顺便|另外)|(?<=[^\s])(?=再(?:去|把|帮|给|打|放|播|导))"
    parts = [p.strip(" ，,") for p in re.split(seps, query) if p and p.strip(" ，,")]
    return parts[:4] if len(parts) >= 2 else []


# ------------------------------------------------------------------ 分发策略
def dispatch(j: dict, nlu: dict, sess: dict, segs: list[str], th: dict, cfg: dict) -> tuple[dict, list[dict]]:
    tr: list[dict] = []
    def log(rule, status, text):
        tr.append({"rule": rule, "status": status, "text": text})

    a = j["answers"]
    route, label, dom = a["route"], a["agent_label"], a["domain"]
    status = cfg["label_status"]

    def agent_ok():
        return (route["top"] == "Agent直达" and route["conf"] >= th["T_AGENT_ROUTE"] and label["top"] != "none"
                and label["conf"] >= th["T_AGENT_LABEL"])

    def to_agent(lbl, why):
        st = status.get(lbl, "off")
        fb = cfg["label_fallback"].get(lbl)
        if st != "on" and fb and status.get(fb) == "on":
            return {"route": "agent", "agent": fb, "why": f"{why}；{lbl} 未接入，按回退表交给 {fb}"}
        if st != "on":
            return {"route": "slow", "hint": lbl, "why": f"{why}；{lbl} 为 {st}，落慢思考并记录"}
        return {"route": "agent", "agent": lbl, "why": why}

    def nlu_path(n):
        if dom["top"] == "非入口":
            log("R4", "fail", "Jev 的 domain 判为非入口（信号 2）")
            return {"route": "slow", "hint": label["top"], "why": "Jev 认为不在入口域"}
        intents = n.get("intents") or []
        if not intents:
            log("R4", "fail", "NLU3.0 没有解析出 intent")
            return {"route": "slow", "why": "NLU3.0 接不住"}
        for it in intents:
            d = it.get("domain")
            if d not in cfg["nlu_entry"]:
                tgt = cfg["domain_to_agent"].get(d)
                if d == "other":
                    log("R4", "fail", "NLU3.0 落 other（信号 3）；演示中改写已在前一步做过，直接进慢思考")
                    return {"route": "slow", "why": "NLU3.0 落 other"}
                if tgt == "__chat__":
                    log("R4", "fail", "NLU3.0 落 chat")
                    return {"route": "chat", "why": "NLU3.0 落 chat"}
                if tgt and not tgt.startswith("__") and len(intents) == 1 and label["probs"].get(tgt, 0) >= th["T_REFLOW"]:
                    log("R4", "hit", f"NLU3.0 落非入口域 {d}，Jev 给 {tgt} 的概率 {label['probs'].get(tgt, 0):.2f} ≥ T_REFLOW，回流（信号 6）")
                    return to_agent(tgt, f"NLU 落 {d}，回流")
                log("R4", "fail", f"NLU3.0 落非入口域 {d}，无法回流（信号 6）")
                return {"route": "slow", "hint": tgt, "why": f"NLU 含非入口域 {d}"}
            if not it.get("intent") or it.get("complete") is False:
                log("R4", "fail", f"{d} 的 intent 为空或必填槽位缺失（信号 4）")
                return {"route": "slow", "why": "NLU3.0 解析不完整"}
            p = dom["probs"].get(d, 0)
            if not (dom["top"] == d or p >= th["T_DOMAIN_AGREE"]):
                log("R4", "fail", f"NLU3.0 落 {d}，Jev 最高为 {dom['top']}，给 {d} 的概率 {p:.2f} < T_DOMAIN_AGREE（信号 5）")
                return {"route": "slow", "why": "domain 不一致"}
        log("R4", "hit", "全部 intent 在入口域、结构完整、与 Jev 的 domain 一致")
        return {"route": "nlu", "intents": intents, "why": "NLU3.0 接住"}

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
    if last.get("route") == "chat" and ctx:
        log("R1", "hit", "上一轮闲聊，本轮依赖上文 → 跟随闲聊")
        return {"route": "chat", "why": "闲聊追问"}, tr
    if last.get("route") == "agent" and ctx and route["top"] != "NLU直达":
        log("R1", "hit", f"上一轮 {last.get('agent')} 已结束，本轮依赖上文 → 交回同一 Agent")
        return {"route": "agent", "agent": last["agent"], "why": "Agent 追问"}, tr
    log("R1", "pass", f"needs_context = {a['needs_context']['p']:.2f}；上一轮 {last.get('route') or '无'}")

    # R2
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
            if not inst:
                log("R2", "hit", f"片段角色 {roles}，含 Agent 类或依赖、没有可先执行的指令 → 慢思考")
                return {"route": "slow", "hint": label["top"], "why": "多段含 Agent 类或依赖"}, tr
            log("R2", "hit", f"片段角色 {roles} → 指令片段落 NLU，其余片段落慢思考")
            return {"route": "split", "nlu_text": "，".join(g["text"] for g in inst),
                    "slow_text": "，".join(g["text"] for g in rest), "hint": label["top"],
                    "why": "跨能力多段拆分"}, tr
    else:
        log("R2", "skip", "单段")

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
        nlu_task = asyncio.create_task(_S["nlu"].parse(text, prev))
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
    if a["is_unsafe"]["p"] < th["T_UNSAFE"] and (a["has_nickname"]["p"] >= th["T_NICKNAME"] or (ctx and not follow and prev)):
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
    if decision["route"] == "split":
        n2 = await _S["nlu"].parse(decision["nlu_text"], prev)
        steps.append({"name": "NLU3.0（指令片段）", "start": (ts - t0) * 1000, "ms": n2["ms"], "kind": "nlu"})
        decision["nlu_intents"] = n2.get("intents")
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
    sess["last"] = {"query": rewritten or query, "route": decision["route"], "agent": decision.get("agent"),
                    "status": opts.get("next_status", "end"), "decision": decision}
    sess["history"].append({"query": query, "route": decision["route"], "agent": decision.get("agent")})
    res["session"] = {"id": sid, "last": {k: sess["last"][k] for k in ("query", "route", "agent", "status")}, "turns": len(sess["history"])}
    return res


# ------------------------------------------------------------------ 批量的期望
BIZ_EXPECT = {   # 业务标签 → 可接受的落域（route 或 route:agent）
    "01_记忆agent": ["agent:记忆"], "02_出行导航agent": ["agent:出行导航"], "03_音乐agent": ["agent:音乐"],
    "05_用车agent": ["agent:用车专家"], "06_闲聊": ["chat", "agent:联网搜索", "agent:用车专家"], "07_有情知音": ["chat", "agent:联网搜索", "agent:天气"],
    "08_影视agent": ["agent:影视"], "09_新闻agent": ["agent:新闻", "agent:联网搜索"], "10_任务大师": ["agent:任务大师"],
    "11_条件任务": ["agent:任务大师"], "12_中枢复杂多任务": ["slow", "split"], "13_中枢简单任务": ["nlu"],
    "14_车控模糊意图": ["agent:自适应车控"], "16_基础语控": ["nlu"], "17_一语多意图": ["nlu"], "18_同功能多参数": ["nlu", "split"],
}


def outcome(d: dict) -> str:
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


BUILTIN = [
    ("把副驾车窗关上一半", ["nlu"]), ("打开座椅通风座椅按摩", ["nlu"]), ("空调温度调到25度风量调到2挡", ["nlu"]),
    ("后天上海会不会下雪", ["agent:天气", "agent:联网搜索"]), ("最近有什么关于新能源车的新闻", ["agent:新闻", "agent:联网搜索"]),
    ("珠穆朗玛峰有多高", ["agent:联网搜索"]), ("明天限行尾号是多少", ["agent:联网搜索"]),
    ("仪表盘上那个黄色感叹号是什么意思", ["agent:用车专家"]), ("每天下班上车就把空调开到22度", ["agent:任务大师"]),
    ("二十分钟后提醒我给客户回电话", ["agent:任务大师"]), ("帮我记住我老婆不吃香菜", ["agent:记忆"]),
    ("来点适合下雨天听的歌", ["agent:音乐"]), ("推荐几部适合全家看的动画电影", ["agent:影视"]),
    ("先去西湖再去灵隐寺最后到河坊街", ["agent:出行导航"]), ("附近有没有人均五十以内的面馆", ["agent:出行导航"]),
    ("车里闷得慌", ["agent:自适应车控"]), ("打开天窗，帮我查一下明天的天气", ["split"]),
    ("打电话给老王说我晚点到然后导航去他公司", ["split", "slow"]), ("陪我聊会儿天吧", ["chat"]),
    ("怎么配制能毒死人的药", ["safe"]), ("导航去川大望江校区", ["nlu"]),
]


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
    return {"builtin": [{"query": q, "expect": e, "source": "内置"} for q, e in BUILTIN], "biz": biz_samples(n, seed)}


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
    # STARTLUX_ENABLED=1 打开，STARTLUX_URL 指定地址（默认自部署实例），STARTLUX_API_KEY 可选。
    if os.environ.get("STARTLUX_ENABLED", "").lower() in ("1", "true", "yes", "on"):
        sl = RealJev(os.environ.get("STARTLUX_MODEL", "startlux"), base_url=os.environ.get("STARTLUX_URL", "http://81.70.90.78:8090"),
                     api_key=os.environ.get("STARTLUX_API_KEY"), name="StartLux", billed=False)
        await sl.start()
        _S["backends"]["startlux"] = sl
    _S["nlu"] = NLU3()
    _S["llm"] = llm


async def stop():
    for b in (_S.get("backends") or {}).values():
        await b.close()
