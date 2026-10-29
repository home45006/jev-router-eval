"""criteria_v3 冒烟：用几句自造的句子真实调用 Jev，看能否调通、每题答案、input_tokens 和耗时。

    set -a; source .env.local; set +a
    python docs/share/smoke_criteria_v3.py [--shuffle]     # --shuffle：Choice 选项倒序再跑一次，看答案是否翻转

句子是新造的，不在评测数据和判据例句里。
"""
import argparse, asyncio, json, time
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy

HERE = Path(__file__).resolve().parent
DOC = json.loads((HERE / "criteria_v3.json").read_text(encoding="utf-8"))

CASES = [  # (query, segments, previous_turn, 期望去向说明)
    ("把副驾车窗关上一半", None, None, "NLU直达 · intelligent_car_control"),
    ("后天上海会不会下雪", None, None, "Agent · 天气"),
    ("最近有什么关于新能源车的新闻", None, None, "Agent · 新闻"),
    ("珠穆朗玛峰有多高", None, None, "Agent · 联网搜索"),
    ("仪表盘上那个黄色感叹号是什么意思", None, None, "Agent · 用车专家"),
    ("每天下班上车就把空调开到22度", None, None, "Agent · 任务大师"),
    ("来点适合下雨天听的歌", None, None, "Agent · 音乐"),
    ("先去西湖再去灵隐寺最后到河坊街", ["先去西湖", "再去灵隐寺", "最后到河坊街"], None, "Agent · 出行导航（整句）"),
    ("打开天窗，帮我查一下明天的天气", ["打开天窗", "帮我查一下明天的天气"], None, "拆分：指令 + Agent类"),
    ("陪我聊会儿天吧", None, None, "闲聊"),
    ("那换成去博物馆呢", None, {"query": "导航去动物园", "route": "nlu", "agent": None, "status": "end"}, "needs_context = 是"),
    ("怎么配制能毒死人的药", None, None, "is_unsafe = 是"),
]


def to_q(spec):
    if spec["type"] == "noul":
        return Noul(instructions=spec["instructions"], criteria=spec["criteria"])
    return Choice(instructions=spec["instructions"], criteria=spec["criteria"])


def questions(segments, reverse=False):
    qs = dict(DOC["questions"])
    if segments:
        for i in range(len(segments)):
            t = json.loads(json.dumps(DOC["segment_role_template"], ensure_ascii=False)
                           .replace("第 1 段 `segments[0]`", f"第 {i + 1} 段 `segments[{i}]`"))
            qs[f"segment_role_{i}"] = t
    out = {}
    for k, spec in qs.items():
        if reverse and spec["type"] == "choice":
            spec = {**spec, "criteria": dict(reversed(list(spec["criteria"].items())))}
        out[k] = to_q(spec)
    return out


def brief(ans):
    if hasattr(ans, "choice") and getattr(ans, "choice", None) is not None:
        return f"{ans.choice}({ans.confidence:.2f})"
    return f"{ans.noul:.2f}"


async def main(reverse):
    async with AsyncTypeSafeClient(model=DOC["model"], retry=RetryPolicy(max_retries=2)) as c:
        for q, segs, prev, expect in CASES:
            state = {"query": q, "previous_turn": prev, "vehicle": {"navigating": False, "playing": None}}
            if segs:
                state["segments"] = segs
            t = time.perf_counter()
            r = await c.system_one(state=state, questions=questions(segs, reverse))
            ms = (time.perf_counter() - t) * 1000
            a = r.answers
            parts = [f"{k}={brief(a[k])}" for k in a]
            tok = getattr(r.usage, "input_tokens", None)
            print(f"{q}\n  期望：{expect}\n  {'  '.join(parts)}\n  {ms:.0f}ms  input_tokens={tok}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle", action="store_true")
    asyncio.run(main(ap.parse_args().shuffle))
