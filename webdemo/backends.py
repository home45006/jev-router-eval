"""演示系统调用的三个外部服务：Jev、DeepSeek、快思考模型。

每个服务有“真实”和“模拟”两种实现。有对应的环境变量就用真实服务，否则用模拟实现：
    TYPESAFE_API_KEY  → Jev（typesafe-sdk，模型 JEV_MODEL，默认 jev-latest）
    DEEPSEEK_API_KEY  → DeepSeek V4 Flash 非思考模式（DEEPSEEK_MODEL，默认 deepseek-v4-flash）
    FAST_MODEL_URL    → 快思考模型的 HTTP 服务（约定见 README）；不设时只记录“未接入”

模拟实现只用于在没有密钥的环境里看界面和流程，结果由关键词规则产生，不代表模型效果。
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Jev
# --------------------------------------------------------------------------- #
@dataclass
class JevResult:
    probabilities: dict[str, float]
    input_tokens: int | None
    ms: float
    model: str


class RealJev:
    kind = "real"

    def __init__(self) -> None:
        from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy
        self._Noul = Noul
        self.model = os.environ.get("JEV_MODEL", "jev-latest")
        self._client = AsyncTypeSafeClient(model=self.model, retry=RetryPolicy(max_retries=2))
        self._entered = None

    async def start(self) -> None:
        self._entered = await self._client.__aenter__()

    async def close(self) -> None:
        if self._entered is not None:
            await self._client.__aexit__(None, None, None)

    async def ask(self, state: dict, questions: dict[str, dict]) -> JevResult:
        qs = {k: self._Noul(instructions=q["instructions"], criteria=q["criteria"]) for k, q in questions.items()}
        t = time.perf_counter()
        r = await self._entered.system_one(state=state, questions=qs)
        ms = (time.perf_counter() - t) * 1000
        return JevResult({k: round(r.answers[k].noul, 4) for k in qs},
                         getattr(r.usage, "input_tokens", None), ms, self.model)


NICKNAMES = {"大裤衩": "中央电视台总部大楼", "鸟巢": "国家体育场", "水立方": "国家游泳中心", "小蛮腰": "广州塔",
             "魔都": "上海", "帝都": "北京", "周董": "周杰伦", "华仔": "刘德华", "霉霉": "泰勒·斯威夫特",
             "发哥": "周润发", "Jay": "周杰伦"}
FAST_CUE = re.compile(r"^(主驾|副驾|左前|右前|左后|右后|前排|后排|全车)?(打开|关闭|关掉|开启|调|把|导航(到|去)|放一首|播放|拨打|打电话|音量|切换|上一|下一|暂停|继续|"
                      r"座椅|空调|车窗|天窗|回家|去公司|退出|返回)")
BRAIN_CUE = re.compile(r"怎么|如何|为什么|什么|哪|推荐|记住|提醒|每天|每周|每当|当.{1,12}(时|的时候)|模式(时|下)|然后|再|顺便|"
                       r"多远|多久|堵|状态|吗|呢|新闻|热点|资讯|热|冷|困|外卖|美团|帮我找|想听|想看|心情|规划|旅行|"
                       r"\d+(分钟|小时)后|明天|后天|今晚|待会|等会")


class MockJev:
    kind = "mock"
    model = "模拟 Jev（关键词规则）"

    async def start(self) -> None: ...
    async def close(self) -> None: ...

    async def ask(self, state: dict, questions: dict[str, dict]) -> JevResult:
        q = state["query"]
        rnd = random.Random(q)
        t = time.perf_counter()
        await asyncio.sleep(rnd.uniform(0.18, 0.32))
        fast = bool(FAST_CUE.search(q)) and not BRAIN_CUE.search(q)
        if state.get("previous_turn") and re.fullmatch(r"(再|换成|那|第).{0,8}", q):
            fast = True
        p = {"is_fast": round(rnd.uniform(0.82, 0.97) if fast else rnd.uniform(0.04, 0.35), 4),
             "needs_rewrite": round(0.9 if any(k in q for k in NICKNAMES) else rnd.uniform(0.01, 0.08), 4)}
        for k in questions:
            if k == "fix_mode_rule":
                p[k] = round(0.8 if re.search(r"(模式|挡|档)(时|下)", q) else rnd.uniform(0.01, 0.1), 4)
            elif k == "fix_setting_toggle":
                p[k] = round(0.85 if re.fullmatch(r"(打开|关闭)[^，,然后再并和]{2,8}", q) else rnd.uniform(0.01, 0.1), 4)
            elif k not in p:
                p[k] = round(rnd.uniform(0.01, 0.1), 4)
        return JevResult(p, None, (time.perf_counter() - t) * 1000, self.model)


# --------------------------------------------------------------------------- #
# DeepSeek
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    content: str
    tool_calls: list[ToolCall]
    ms: float
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    raw_message: dict = field(default_factory=dict)   # 原样放回对话历史用


class RealLLM:
    kind = "real"

    def __init__(self) -> None:
        from openai import AsyncOpenAI
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self._client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                                   base_url=os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
                                   timeout=30)

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False,
                   temperature: float = 0.0, role: str = "") -> LLMResult:
        kw = {"model": self.model, "messages": messages, "temperature": temperature,
              "extra_body": {"thinking": {"type": "disabled"}}}       # V4 Flash 非思考模式
        if tools:
            kw["tools"] = tools
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        t = time.perf_counter()
        res = await self._client.chat.completions.create(**kw)
        ms = (time.perf_counter() - t) * 1000
        msg = res.choices[0].message
        calls = []
        for c in msg.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": c.function.arguments}
            calls.append(ToolCall(c.id, c.function.name, args))
        u = res.usage
        det = getattr(u, "completion_tokens_details", None)
        return LLMResult(msg.content or "", calls, ms, res.model,
                         getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None),
                         (getattr(det, "reasoning_tokens", None) or 0) if det else 0,
                         msg.model_dump(exclude_none=True))


# 模拟大脑：按关键词给出能力标签
MOCK_RULES = [
    ("美团", r"外卖|美团|点餐|点份|饿了"), ("任务大师", r"每天|每周|每当|当.{1,12}(时|的时候)|(模式|挡|档)(时|下)|上车(时|后)|下班回家时"),
    ("条件任务", r"\d+(分钟|小时)后|明天|后天|今晚|待会|等会|点钟|提醒我|日程"), ("联网搜索", r"新闻|热点|资讯|最新进展"),
    ("用车手册", r"怎么(用|打开|关闭|设置)|如何|是什么|什么意思|状态|开着|开启的吗|关了吗|故障|异响|报警"),
    ("导航", r"导航|去哪|路线|规划|多远|多久|堵|加油站|充电站|旅行|行程|景点"),
    ("空调", r"热|冷|闷|空调|通风|加热|除雾"), ("多媒体", r"歌|音乐|电影|电视剧|综艺|听|看"),
    ("记忆", r"记住|记一下|帮我存"), ("闲聊", r"你好|讲个|笑话|诗|几点|哪天|天气|为什么|是谁"),
]


class MockLLM:
    kind = "mock"
    model = "模拟大模型（关键词规则）"

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False,
                   temperature: float = 0.0, role: str = "") -> LLMResult:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        rnd = random.Random(user + role + str(len(messages)))
        t = time.perf_counter()
        await asyncio.sleep(rnd.uniform(0.35, 0.9))
        ms = (time.perf_counter() - t) * 1000
        if role == "rewrite":
            out = user
            for k, v in NICKNAMES.items():
                out = out.replace(k, v)
            return LLMResult(out, [], ms, self.model, 120, 8, 0)
        if role.startswith("ability:"):
            return LLMResult(f"（模拟）{role[8:]}已处理：{user[:40]}", [], ms, self.model, 200, 20, 0)
        query = user.split("\n")[0]
        labels = [name for name, pat in MOCK_RULES if re.search(pat, query)] or ["闲聊"]
        status = "continue" if "美团" in labels else "end"
        ordering = any(m["role"] == "user" and re.search(MOCK_RULES[0][1], m["content"]) for m in messages[:-1])
        if ordering and "美团" not in labels:          # 点外卖途中
            if re.search(r"第.|家|辣|下单|就这样|好的|地址", query):
                labels, status = ["美团"], ("end" if re.search(r"下单|就这样", query) else "continue")
            else:
                status = "suspend"
        final = {"labels": labels, "plan": "按关键词规则模拟：" + "、".join(labels), "status": status,
                 "speech": f"（模拟）好的，我来处理：{query[:20]}"}
        if tools and not any(m["role"] == "tool" for m in messages):
            name_of = {"快思考": "fast_think", "任务大师": "task_master", "条件任务": "conditional_task",
                       "联网搜索": "web_search", "用车手册": "car_manual", "导航": "navigation", "空调": "climate",
                       "多媒体": "media", "美团": "meituan", "记忆": "memory", "闲聊": "chat"}
            calls = [ToolCall(f"call_{i}", name_of[l], {"task": query}) for i, l in enumerate(labels)]
            return LLMResult("", calls, ms, self.model, 900, 40, 0,
                             {"role": "assistant", "content": "", "tool_calls": [
                                 {"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                                 for c in calls]})
        return LLMResult(json.dumps(final, ensure_ascii=False), [], ms, self.model, 1000, 60, 0)


# --------------------------------------------------------------------------- #
# 快思考模型
# --------------------------------------------------------------------------- #
@dataclass
class FastResult:
    connected: bool
    rejected: bool
    confidence: float | None
    detail: dict
    ms: float


class FastModel:
    """快思考模型。FAST_MODEL_URL 约定：POST {"query", "previous_turn"}，
    返回 {"domain", "intent", "slots", "confidence", "rejected"}。"""

    def __init__(self) -> None:
        self.url = os.environ.get("FAST_MODEL_URL")
        self.kind = "real" if self.url else "absent"

    async def run(self, query: str, previous_turn: dict | None, simulate_reject: bool) -> FastResult:
        t = time.perf_counter()
        if not self.url:
            await asyncio.sleep(0.02)
            return FastResult(False, simulate_reject, None,
                              {"note": "快思考服务未接入（设置 FAST_MODEL_URL 后调用真实服务）",
                               "模拟拒识": simulate_reject}, (time.perf_counter() - t) * 1000)
        import httpx
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(self.url, json={"query": query, "previous_turn": previous_turn})
            r.raise_for_status()
            d = r.json()
        return FastResult(True, bool(d.get("rejected")), d.get("confidence"), d, (time.perf_counter() - t) * 1000)


def make_backends():
    jev = RealJev() if os.environ.get("TYPESAFE_API_KEY") else MockJev()
    llm = RealLLM() if os.environ.get("DEEPSEEK_API_KEY") else MockLLM()
    return jev, llm, FastModel()
