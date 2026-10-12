"""联网搜索：通过 MCP 调用搜索服务（默认阿里云百炼“联网搜索”MCP）。

环境变量：
    DASHSCOPE_API_KEY   百炼的 API Key（也可用 SEARCH_MCP_KEY），以 Authorization: Bearer 传给 MCP 服务
    SEARCH_MCP_URL      MCP 服务地址；不设时依次尝试百炼的 streamable HTTP 和 SSE 地址
    SEARCH_MCP_TOOL     工具名；不设时取工具列表里名字含 search 的第一个

工具参数不写死：从工具的 inputSchema 里找第一个必填的字符串参数，把搜索词放进去。
每次搜索单独建立连接，耗时分成“建立连接”和“调用工具”两段记录。

    python -m webdemo.search_mcp "今天有什么热点新闻"     # 探测：列出工具并搜索一次
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

BAILIAN_URLS = [
    "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
    "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
]


@dataclass
class SearchResult:
    query: str
    tool: str
    url: str
    text: str                       # 工具返回的原始文本（多段拼接）
    items: list[dict] = field(default_factory=list)   # 解析出的条目：title / url / snippet
    connect_ms: float = 0.0
    call_ms: float = 0.0
    is_error: bool = False


def _key() -> str | None:
    return os.environ.get("SEARCH_MCP_KEY") or os.environ.get("DASHSCOPE_API_KEY")


def configured() -> bool:
    return bool(_key())


@asynccontextmanager
async def _session(url: str):
    from mcp import ClientSession
    from mcp.shared._httpx_utils import create_mcp_http_client
    headers = {"Authorization": f"Bearer {_key()}"}
    if url.rstrip("/").endswith("/sse"):
        from mcp.client.sse import sse_client
        async with sse_client(url, headers=headers, timeout=10) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                yield s
    else:
        from mcp.client.streamable_http import streamable_http_client
        async with create_mcp_http_client(headers=headers) as http:
            async with streamable_http_client(url, http_client=http) as streams:
                r, w = streams[0], streams[1]
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    yield s


def _attr(obj, *names):
    """mcp 1.x 用 camelCase，2.x 用 snake_case。"""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _err(e: BaseException) -> str:
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return f"{type(e).__name__}: {e}"


def _pick_tool(tools) -> tuple[str, str, dict]:
    want = os.environ.get("SEARCH_MCP_TOOL")
    tool = next((t for t in tools if t.name == want), None) if want else None
    tool = tool or next((t for t in tools if "search" in t.name.lower()), None) or (tools[0] if tools else None)
    if tool is None:
        raise RuntimeError("MCP 服务没有提供任何工具")
    schema = _attr(tool, "input_schema", "inputSchema") or {}
    props = schema.get("properties", {})
    required = schema.get("required", []) or list(props)
    arg = next((k for k in required if props.get(k, {}).get("type", "string") == "string"), None) or "query"
    extra = {}
    for k, p in props.items():                       # 常见的条数参数给个小值，控制时延
        if k in ("count", "num", "top_k", "topK", "size", "limit") and p.get("type") in ("integer", "number"):
            extra[k] = 5
    return tool.name, arg, extra


def _text_of(result) -> tuple[str, bool]:
    parts = []
    for c in getattr(result, "content", []) or []:
        parts.append(getattr(c, "text", None) or json.dumps(c.model_dump(), ensure_ascii=False))
    sc = _attr(result, "structured_content", "structuredContent")
    if not parts and sc:
        parts.append(json.dumps(sc, ensure_ascii=False))
    return "\n".join(parts), bool(_attr(result, "is_error", "isError"))


def _items(text: str) -> list[dict]:
    """尽量从返回里抽出 title / url / snippet；格式不认识就返回空，前端显示原文。"""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    stack, out = [data], []
    while stack and len(out) < 10:
        x = stack.pop(0)
        if isinstance(x, dict):
            title = x.get("title") or x.get("name")
            url = x.get("url") or x.get("link")
            snip = x.get("snippet") or x.get("summary") or x.get("content") or x.get("description")
            if title and (url or snip):
                out.append({"title": str(title)[:120], "url": url, "snippet": str(snip or "")[:300]})
                continue
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return out


class SearchMCP:
    def __init__(self) -> None:
        self.kind = "real" if configured() else "absent"
        env_url = os.environ.get("SEARCH_MCP_URL")
        self.urls = [env_url] if env_url else BAILIAN_URLS
        self.url: str | None = None        # 第一次成功后记住可用的地址
        self._tool: tuple[str, str, dict] | None = None

    async def search(self, query: str) -> SearchResult:
        errors = []
        for url in ([self.url] if self.url else self.urls):
            t0 = time.perf_counter()
            try:
                async with _session(url) as s:
                    t1 = time.perf_counter()
                    if self._tool is None:
                        self._tool = _pick_tool((await s.list_tools()).tools)
                    name, arg, extra = self._tool
                    res = await s.call_tool(name, {arg: query, **extra}, read_timeout_seconds=20)
                    t2 = time.perf_counter()
                text, err = _text_of(res)
                self.url = url
                return SearchResult(query, name, url, text, _items(text), (t1 - t0) * 1000, (t2 - t1) * 1000, err)
            except Exception as e:  # 换下一个候选地址
                errors.append(f"{url}: {_err(e)}"[:300])
        raise RuntimeError("联网搜索 MCP 调用失败：" + " | ".join(errors))


async def _probe(query: str) -> None:
    if not configured():
        sys.exit("需要 DASHSCOPE_API_KEY（或 SEARCH_MCP_KEY）")
    sm = SearchMCP()
    for url in sm.urls:
        t0 = time.perf_counter()
        try:
            async with _session(url) as s:
                tools = (await s.list_tools()).tools
            print(f"## {url}\n- 连接成功，{(time.perf_counter() - t0) * 1000:.0f} ms，工具：")
            for t in tools:
                print(f"  - {t.name}：{(t.description or '')[:120]}\n    参数：{json.dumps(_attr(t, "input_schema", "inputSchema"), ensure_ascii=False)[:400]}")
        except Exception as e:
            print(f"## {url}\n- 失败：{_err(e)[:300]}")
    for q in [query, "今天有什么热点新闻", "比亚迪最新车型是什么"]:
        try:
            r = await sm.search(q)
            print(f"\n### 搜索：{q}\n- 地址 {r.url}，工具 {r.tool}，建立连接 {r.connect_ms:.0f} ms，调用 {r.call_ms:.0f} ms，"
                  f"isError={r.is_error}，解析出 {len(r.items)} 条")
            for it in r.items[:5]:
                print(f"  - {it['title']} | {it.get('url')} | {it['snippet'][:80]}")
            print("- 原文前 600 字：" + r.text[:600].replace("\n", " "))
        except Exception as e:
            print(f"\n### 搜索：{q}\n- 失败：{e}")


if __name__ == "__main__":
    asyncio.run(_probe(sys.argv[1] if len(sys.argv) > 1 else "智能座舱最新进展"))
