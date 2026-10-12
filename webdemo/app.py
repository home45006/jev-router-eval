"""前置分流演示系统的 Web 服务。

    pip install -r webdemo/requirements.txt
    export TYPESAFE_API_KEY=...  DEEPSEEK_API_KEY=...     # 不设则对应部分用模拟实现
    uvicorn webdemo.app:app --port 8000                    # 在仓库根目录运行
    浏览器打开 http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .backends import make_backends
from .pricing import PRICE_SOURCE, PRICES
from .pricing import period as price_period
from .search_deepseek import DeepSeekSearch
from .pipeline import ABILITIES, CRITERIA, LABELS, Pipeline, Settings

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# 评测集原标签 → 应去向和大脑能力标签
FAST_DOMAINS = {"carControl", "windowControl", "systemControl", "music", "mediaControl", "generalControl", "video",
                "unicast", "naviControl", "navi", "autoPilot", "relaxCapsule", "carInfo", "phone", "memoryRecord",
                "中枢简单任务"}
BRAIN_LABEL_OF = {"用车agent": "用车手册", "复杂导航agent": "导航", "音乐agent": "多媒体", "影视agent": "多媒体",
                  "记忆agent": "记忆", "任务大师": "任务大师", "条件任务": "条件任务", "新闻agent": "联网搜索",
                  "闲聊": "闲聊", "车控agent模糊意图": "空调", "中枢复杂多任务": "多能力",
                  "scheduleReminder": "条件任务"}

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    jev, llm, fast = make_backends()
    await jev.start()
    search = DeepSeekSearch()
    state["pipeline"] = Pipeline(jev, llm, fast, search)
    state["kinds"] = {"jev": jev.kind, "llm": llm.kind, "fast": fast.kind, "search": search.kind,
                      "jev_model": jev.model, "llm_model": llm.model}
    yield
    await jev.close()


app = FastAPI(title="座舱前置分流演示", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


class RunReq(BaseModel):
    query: str
    session_id: str | None = None
    settings: dict | None = None


class BatchItem(BaseModel):
    query: str
    dest: str | None = None
    labels: list[str] | None = None
    source: str | None = None


class BatchReq(BaseModel):
    items: list[BatchItem]
    settings: dict | None = None
    concurrency: int = 4


@app.get("/")
async def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/config")
async def config():
    s = Settings()
    return {"backends": state["kinds"], "criteria_version": CRITERIA["version"],
            "defaults": {k: getattr(s, k) for k in ("is_fast_t", "rewrite_t", "correction_t", "corrections",
                                                    "brain_mode", "simulate_fast_reject", "max_iters", "run_search")},
            "corrections": {k: {"why": v["why"], "route": v["route"], "hint": v.get("hint"),
                                "instructions": v["instructions"], "status": v["status"]}
                            for k, v in CRITERIA["corrections"].items()},
            "stage1": {k: v["instructions"] for k, v in CRITERIA["stage1"].items()},
            "abilities": [{"label": l, "tool": t, "desc": d} for l, t, d in ABILITIES],
            "pricing": {"source": PRICE_SOURCE, "prices": PRICES, "period_now": price_period()}}


def _ndjson(gen):
    return StreamingResponse(gen, media_type="application/x-ndjson")


@app.post("/api/run")
async def run(req: RunReq):
    q: asyncio.Queue = asyncio.Queue()
    sid = req.session_id or uuid.uuid4().hex[:8]

    async def emit(ev):
        await q.put(json.dumps(ev, ensure_ascii=False))

    async def work():
        try:
            res = await state["pipeline"].run(req.query.strip(), sid, Settings.from_dict(req.settings), emit)
            await q.put(json.dumps({"type": "result", "result": res}, ensure_ascii=False))
        except Exception as e:
            await q.put(json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"[:500]}, ensure_ascii=False))
        await q.put(None)

    async def gen():
        task = asyncio.create_task(work())
        while (line := await q.get()) is not None:
            yield line + "\n"
        await task

    return _ndjson(gen())


@app.post("/api/reset")
async def reset(body: dict):
    state["pipeline"].reset(body.get("session_id", ""))
    return {"ok": True}


def judge(item: BatchItem, res: dict) -> dict:
    ok_dest = None if not item.dest else res.get("destination") == item.dest
    ok_label = ok_any = None
    if item.dest == "Agent大脑" and item.labels and res.get("destination") == "Agent大脑":
        got = res.get("labels") or []
        if item.labels == ["多能力"]:
            ok_label = ok_any = len(got) >= 2
        else:
            ok_label = bool(got) and got[0] in item.labels          # 首选标签（第一个）是否正确
            ok_any = bool(set(item.labels) & set(got))              # 标签里是否包含正确答案
    return {"dest": ok_dest, "label": ok_label, "label_any": ok_any}


@app.post("/api/batch")
async def batch(req: BatchReq):
    settings = Settings.from_dict(req.settings)
    sem = asyncio.Semaphore(max(1, min(req.concurrency, 8)))
    q: asyncio.Queue = asyncio.Queue()

    async def noop(_):
        return None

    async def one(i, item):
        async with sem:
            sid = f"batch-{uuid.uuid4().hex[:8]}"
            try:
                res = await state["pipeline"].run(item.query, sid, settings, noop)
                compact = {k: res.get(k) for k in ("destination", "labels", "jev", "path", "rewritten", "correction", "timing", "cost")}
                compact["brain"] = {k: (res.get("brain") or {}).get(k) for k in ("plan", "status", "speech")}
                compact["steps"] = [{**{k: s[k] for k in ("name", "kind", "start", "end", "status", "summary")},
                                     "detail": {"cost": (s.get("detail") or {}).get("cost")}} for s in res["steps"]]
                out = {"type": "item", "index": i, "item": item.model_dump(), "result": compact, "correct": judge(item, res)}
            except Exception as e:
                out = {"type": "item", "index": i, "item": item.model_dump(), "error": f"{type(e).__name__}: {e}"[:300]}
            finally:
                state["pipeline"].reset(sid)
            await q.put(json.dumps(out, ensure_ascii=False))

    async def gen():
        tasks = [asyncio.create_task(one(i, it)) for i, it in enumerate(req.items)]
        for _ in tasks:
            yield (await q.get()) + "\n"
        await asyncio.gather(*tasks)

    return _ndjson(gen())


def _eval_set_samples(n_per_label: int = 6) -> list[dict]:
    path = Path(os.environ.get("EVAL_SET", REPO / "eval_set.jsonl"))
    if not path.exists():
        return []
    by = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        it = json.loads(line)
        if it.get("split") != "dev" or it.get("turn", 1) > 1:
            continue
        g = it["gold"]
        fast = any(x in FAST_DOMAINS for x in g)
        brain = [BRAIN_LABEL_OF[x] for x in g if x in BRAIN_LABEL_OF]
        if fast and brain:
            continue                                   # 两种都算对的句子不放进演示
        by.setdefault(it["primary"], []).append(
            {"query": it["query"], "dest": "快思考" if fast else "Agent大脑",
             "labels": ["快思考"] if fast else sorted(set(brain)), "source": f"dev · {it['primary']}"})
    rnd = random.Random(0)
    out = []
    for k in sorted(by):
        out += rnd.sample(by[k], min(n_per_label, len(by[k])))
    return out


@app.get("/api/samples")
async def samples():
    d = json.loads((HERE / "samples.json").read_text(encoding="utf-8"))
    return {"builtin": d["items"], "dialogs": d["dialogs"], "eval_set": _eval_set_samples(), "labels": LABELS}
