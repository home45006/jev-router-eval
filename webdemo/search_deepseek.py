"""联网搜索：用 DeepSeek 自带的联网搜索（服务端执行的 web_search 工具），流式接收。

DeepSeek 的 Anthropic 兼容接口（{DEEPSEEK_BASE_URL}/anthropic/v1/messages）支持服务端联网搜索：
请求里带 {"type": "web_search_20250305", "name": "web_search"}，模型自己决定搜什么、搜几次，
一次请求里完成“搜索 + 写答案”，返回 text 块和 web_search_tool_result 块（来源）。
OpenAI 兼容接口（/chat/completions）没有这个能力。

用 stream=true 接收 SSE 事件，记录三个时间点（相对请求发出）：
    first_event_ms   收到第一个事件（连接建立、开始处理）
    search_done_ms   最后一次搜索结果（web_search_tool_result）到达
    ttft_ms          答案首字：搜索结果之后的第一个字，语音播报能开始的时刻
模型偶尔会在搜索前先说一句（实测出现过英文的“I'll search for ...”），这段记为 preamble，
不进播报；preamble_ms 是它的首字时间。没有触发搜索时，第一个字就是答案首字。

环境变量：DEEPSEEK_API_KEY；可选 DEEPSEEK_SEARCH_MODEL（默认同 DEEPSEEK_MODEL，再默认 deepseek-flash）、
DEEPSEEK_BASE_URL、DEEPSEEK_SEARCH_MAX_USES（默认 2）。没有 DEEPSEEK_API_KEY 时用 MockSearch 演示流程。

    python -m webdemo.search_deepseek "今天有什么热点新闻"     # 探测：打印时间点、来源和答案
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx

from .pricing import cost, cost_anthropic

ANSWER_SYSTEM = ("你是车载语音助手的联网搜索能力。需要最新信息时先联网搜索，搜索前不要输出任何文字；"
                 "拿到结果后用中文、适合语音播报的两三句话回答，只说搜到的信息，不要编造；没查到就直说。不要输出链接和 Markdown。")
OnDelta = Callable[[str], Awaitable[None]] | None


@dataclass
class SearchResult:
    query: str
    answer: str                                        # 模型根据搜索结果写好的播报
    items: list[dict] = field(default_factory=list)    # 来源：title / url / snippet
    queries: list[str] = field(default_factory=list)   # 模型实际用的搜索词
    usage: dict = field(default_factory=dict)
    model: str = ""
    url: str = ""
    ms: float = 0.0
    cost: dict | None = None
    first_event_ms: float | None = None
    search_done_ms: float | None = None
    ttft_ms: float | None = None
    answer_first_ms: float | None = None
    preamble: str = ""                                 # 搜索前模型说的话，不进播报
    preamble_ms: float | None = None


def _items_of(content) -> list[dict]:
    out = []
    if isinstance(content, list):
        for x in content:
            if isinstance(x, dict) and (x.get("title") or x.get("url")):
                out.append({"title": str(x.get("title") or x.get("url"))[:120], "url": x.get("url"),
                            "snippet": str(x.get("page_age") or "")})
    return out


class DeepSeekSearch:
    def __init__(self) -> None:
        self.kind = "real" if os.environ.get("DEEPSEEK_API_KEY") else "absent"
        base = (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.url = base + "/anthropic/v1/messages"
        self.model = os.environ.get("DEEPSEEK_SEARCH_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-flash"
        self.max_uses = int(os.environ.get("DEEPSEEK_SEARCH_MAX_USES", "2"))

    def _body(self, query: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": 1024,
            "stream": True,
            "system": f"今天是 {datetime.date.today():%Y年%m月%d日}。" + ANSWER_SYSTEM,     # 不给日期时模型会按训练年份搜
            "messages": [{"role": "user", "content": query}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": self.max_uses}],
            "thinking": {"type": "disabled"},          # 非思考模式（官方文档：思考模式默认打开）
        }

    async def search(self, query: str, on_delta: OnDelta = None) -> SearchResult:
        headers = {"x-api-key": os.environ["DEEPSEEK_API_KEY"], "anthropic-version": "2023-06-01",
                   "content-type": "application/json", "accept": "text/event-stream"}
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731
        res = SearchResult(query=query, answer="", model=self.model, url=self.url)
        blocks: dict[int, dict] = {}              # index → {type, json}
        usage: dict = {}
        texts: list[str] = []
        res._pre = []                             # 搜索结果到达前的文字
        async with httpx.AsyncClient(timeout=60) as c:
            async with c.stream("POST", self.url, headers=headers, json=self._body(query)) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode("utf-8", "replace")
                    raise RuntimeError(f"DeepSeek 联网搜索 HTTP {r.status_code}: {body[:300]}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if res.first_event_ms is None:
                        res.first_event_ms = ms()
                    et = ev.get("type")
                    if et == "message_start":
                        msg = ev.get("message") or {}
                        res.model = msg.get("model") or res.model
                        usage.update(msg.get("usage") or {})
                    elif et == "content_block_start":
                        b = ev.get("content_block") or {}
                        blocks[ev.get("index", len(blocks))] = {"type": b.get("type"), "json": ""}
                        if b.get("type") == "web_search_tool_result":
                            res.items += _items_of(b.get("content"))
                            if isinstance(b.get("content"), dict) and b["content"].get("error_code"):
                                texts.append(f"（搜索出错：{b['content']['error_code']}）")
                            res.search_done_ms = ms()
                        elif b.get("type") == "server_tool_use" and (b.get("input") or {}).get("query"):
                            res.queries.append(b["input"]["query"])
                        elif b.get("type") == "text" and b.get("text"):
                            await self._text(res, texts, b["text"], ms, on_delta)
                    elif et == "content_block_delta":
                        d = ev.get("delta") or {}
                        if d.get("type") == "text_delta" and d.get("text"):
                            await self._text(res, texts, d["text"], ms, on_delta)
                        elif d.get("type") == "input_json_delta":
                            blocks.setdefault(ev.get("index", -1), {"type": None, "json": ""})["json"] += d.get("partial_json", "")
                    elif et == "content_block_stop":
                        b = blocks.get(ev.get("index"))
                        if b and b["type"] == "server_tool_use" and b["json"]:
                            try:
                                q = json.loads(b["json"]).get("query")
                                if q and q not in res.queries:
                                    res.queries.append(q)
                            except json.JSONDecodeError:
                                pass
                    elif et == "message_delta":
                        usage.update(ev.get("usage") or {})
                    elif et == "error":
                        raise RuntimeError(f"DeepSeek 联网搜索出错：{json.dumps(ev.get('error'), ensure_ascii=False)[:300]}")
        res.ms = ms()
        res.preamble = "".join(res._pre).strip()
        res.answer = "".join(texts).strip()
        if not res.answer and res.preamble and res.search_done_ms is None:     # 没搜索就直接回答
            res.answer, res.preamble, res.ttft_ms, res.preamble_ms = res.preamble, "", res.preamble_ms, None
        res.items = res.items[:10]
        res.usage = usage
        res.cost = cost_anthropic(res.model, usage)
        return res

    @staticmethod
    async def _text(res: SearchResult, texts: list[str], text: str, ms, on_delta: OnDelta) -> None:
        now = ms()
        if res.search_done_ms is None:            # 还没拿到搜索结果：可能是搜索前的开场白，也可能不搜直接答
            if res.preamble_ms is None:
                res.preamble_ms = now
            res._pre.append(text)
            return
        if res.ttft_ms is None:
            res.ttft_ms = res.answer_first_ms = now
        texts.append(text)
        if on_delta:
            await on_delta(text)


class MockSearch:
    """没有 DEEPSEEK_API_KEY 时演示流程：模拟搜索约 1.2 秒，然后逐段吐出答案。结果不是真实搜索。"""
    kind = "mock"
    url = "模拟"

    async def search(self, query: str, on_delta: OnDelta = None) -> SearchResult:
        rnd = random.Random(query)
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731
        res = SearchResult(query=query, answer="", model="模拟大模型", url=self.url, queries=[f"{query} 最新"])
        await asyncio.sleep(0.15)
        res.first_event_ms = ms()
        await asyncio.sleep(rnd.uniform(0.9, 1.4))
        res.search_done_ms = ms()
        res.items = [{"title": f"（模拟）关于“{query}”的报道 {i + 1}", "url": None, "snippet": ""} for i in range(3)]
        await asyncio.sleep(rnd.uniform(0.25, 0.45))
        chunks = ["（模拟）", "根据搜索结果，", f"关于{query[:12]}，", "这里是一段示意答案，", "真实效果请设置 DEEPSEEK_API_KEY。"]
        for ch in chunks:
            if res.ttft_ms is None:
                res.ttft_ms = res.answer_first_ms = ms()
            res.answer += ch
            if on_delta:
                await on_delta(ch)
            await asyncio.sleep(0.12)
        res.ms = ms()
        res.usage = {"input_tokens": 9000, "cache_read_input_tokens": 512, "output_tokens": 80}
        res.cost = cost("deepseek-flash", 512, 9000, 80)
        return res


async def _probe(query: str) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("需要 DEEPSEEK_API_KEY")
    s = DeepSeekSearch()
    print(f"## DeepSeek 联网搜索探测（流式）\n- 地址 {s.url}，模型 {s.model}，max_uses {s.max_uses}\n")
    for q in [query, "今天有什么热点新闻", "比亚迪最近发布了什么新车"]:
        try:
            r = await s.search(q)
            print(f"### {q}\n- 首个事件 {r.first_event_ms} ms，搜索结果到达 {r.search_done_ms} ms，答案首字 {r.ttft_ms} ms，结束 {r.ms} ms"
                  + (f"；搜索前开场白（不进播报）{r.preamble_ms} ms：{r.preamble}" if r.preamble else ""))
            print(f"- 模型 {r.model}，搜索词 {r.queries}，来源 {len(r.items)} 条，用量 {json.dumps(r.usage, ensure_ascii=False)}，"
                  f"费用 ¥{(r.cost or {}).get('yuan')}")
            for it in r.items[:5]:
                print(f"  - {it['title']} | {it.get('url')}")
            print(f"- 播报：{r.answer[:300]}\n")
        except Exception as e:
            print(f"### {q}\n- 失败：{type(e).__name__}: {str(e)[:400]}\n")


if __name__ == "__main__":
    asyncio.run(_probe(sys.argv[1] if len(sys.argv) > 1 else "智能座舱最新进展"))
