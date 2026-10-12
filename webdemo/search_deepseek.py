"""联网搜索：用 DeepSeek 自带的联网搜索（服务端执行的 web_search 工具）。

DeepSeek 的 Anthropic 兼容接口（{DEEPSEEK_BASE_URL}/anthropic/v1/messages）支持服务端联网搜索：
请求里带 {"type": "web_search_20250305", "name": "web_search"}，模型自己决定搜什么、搜几次，
一次请求里完成“搜索 + 写答案”，返回 text 块和 web_search_tool_result 块（来源）。
OpenAI 兼容接口（/chat/completions）没有这个能力。

环境变量：DEEPSEEK_API_KEY；可选 DEEPSEEK_SEARCH_MODEL（默认同 DEEPSEEK_MODEL，再默认 deepseek-v4-flash）、
DEEPSEEK_BASE_URL、DEEPSEEK_SEARCH_MAX_USES（默认 2）。

    python -m webdemo.search_deepseek "今天有什么热点新闻"     # 探测：打印原始返回结构
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import httpx

from .search_mcp import SearchResult

ANSWER_SYSTEM = ("你是车载语音助手的联网搜索能力。需要最新信息时先联网搜索，然后用适合语音播报的两三句话回答，"
                 "只说搜到的信息，不要编造；没查到就直说。不要输出链接和 Markdown。")


class DeepSeekSearch:
    def __init__(self) -> None:
        self.kind = "real" if os.environ.get("DEEPSEEK_API_KEY") else "absent"
        self.provider = "deepseek"
        base = (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.url = base + "/anthropic/v1/messages"
        self.model = os.environ.get("DEEPSEEK_SEARCH_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash"
        self.max_uses = int(os.environ.get("DEEPSEEK_SEARCH_MAX_USES", "2"))

    def _body(self, query: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": 1024,
            "system": ANSWER_SYSTEM,
            "messages": [{"role": "user", "content": query}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": self.max_uses}],
            "thinking": {"type": "disabled"},          # 非思考模式
        }

    async def raw(self, query: str) -> tuple[dict, float]:
        headers = {"x-api-key": os.environ["DEEPSEEK_API_KEY"], "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        t = time.perf_counter()
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(self.url, headers=headers, json=self._body(query))
        ms = (time.perf_counter() - t) * 1000
        if r.status_code >= 400:
            raise RuntimeError(f"DeepSeek 联网搜索 HTTP {r.status_code}: {r.text[:300]}")
        return r.json(), ms

    async def search(self, query: str) -> SearchResult:
        data, ms = await self.raw(query)
        texts, items, queries = [], [], []
        for b in data.get("content", []):
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text", ""))
            elif t == "server_tool_use":
                q = (b.get("input") or {}).get("query")
                if q:
                    queries.append(q)
            elif t == "web_search_tool_result":
                content = b.get("content")
                if isinstance(content, list):
                    for x in content:
                        if isinstance(x, dict) and (x.get("title") or x.get("url")):
                            items.append({"title": str(x.get("title") or x.get("url"))[:120], "url": x.get("url"),
                                          "snippet": str(x.get("page_age") or "")})
                elif isinstance(content, dict) and content.get("error_code"):
                    texts.append(f"（搜索出错：{content.get('error_code')}）")
        usage = data.get("usage") or {}
        res = SearchResult(query=query, tool="DeepSeek web_search", url=self.url,
                           text=json.dumps({"queries": queries, "usage": usage}, ensure_ascii=False),
                           items=items[:10], connect_ms=0.0, call_ms=ms, is_error=False)
        res.answer = "".join(texts).strip()
        res.queries = queries
        res.usage = usage
        res.model = data.get("model", self.model)
        return res


async def _probe(query: str) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("需要 DEEPSEEK_API_KEY")
    s = DeepSeekSearch()
    print(f"## DeepSeek 联网搜索探测\n- 地址 {s.url}，模型 {s.model}，max_uses {s.max_uses}\n")
    for q in [query, "今天有什么热点新闻", "比亚迪最近发布了什么新车"]:
        try:
            data, ms = await s.raw(q)
            kinds = [b.get("type") for b in data.get("content", [])]
            print(f"### {q}\n- 耗时 {ms:.0f} ms，返回块：{kinds}，usage：{json.dumps(data.get('usage'), ensure_ascii=False)}")
            r = await s.search(q)
            print(f"- 模型搜索词：{r.queries}；来源 {len(r.items)} 条：")
            for it in r.items[:5]:
                print(f"  - {it['title']} | {it.get('url')}")
            print(f"- 播报：{r.answer[:300]}\n")
        except Exception as e:
            print(f"### {q}\n- 失败：{type(e).__name__}: {str(e)[:400]}\n")


if __name__ == "__main__":
    asyncio.run(_probe(sys.argv[1] if len(sys.argv) > 1 else "智能座舱最新进展"))
