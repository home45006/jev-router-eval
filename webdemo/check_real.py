"""在有真实密钥的环境里（GitHub Actions）把演示系统完整跑一遍，打印每一步的判断和耗时。

    python -m webdemo.check_real            # 需要 TYPESAFE_API_KEY、DEEPSEEK_API_KEY
结果写到 runs/webdemo/report.md，同时打印到日志。
"""
import json
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from .app import app

SINGLE = [
    ("导航到大裤衩", "label"),
    ("右后座椅通风调成1", "label"),
    ("热得浑身不自在", "label"),
    ("每周二9点提醒我听新闻", "label"),
    ("家到公司堵不堵", "label"),
    ("把空调调到二十六度然后导航去我常去的那家商场", "orchestrate"),
    ("打个电话给小李说聚餐地点定在渔人码头然后导航过去", "orchestrate"),
    ("今天有什么热点新闻", "label"),
    ("比亚迪最近发布了什么新车", "orchestrate"),
]
DIALOG = ["帮我点份外卖送到公司", "先把空调调低一点", "第二家，要中辣的", "好的就这样下单吧"]


def run(c, q, sid, mode):
    lines = c.post("/api/run", json={"query": q, "session_id": sid, "settings": {"brain_mode": mode}}).text.splitlines()
    last = json.loads(lines[-1])
    if last["type"] == "error":
        raise SystemExit(f"{q}: {last['error']}")
    return last["result"]


def fmt(r):
    t = r["timing"]
    steps = "；".join(f"{s['name']} {s['end'] - s['start']:.0f}ms" for s in r["steps"])
    return (f"| {r['query']} | {r['destination']} | {'、'.join(r['labels'])} | {(r.get('brain') or {}).get('status', '-')} | "
            f"{(r.get('jev') or {}).get('is_fast', float('nan')):.2f} | {t['jev_ms']:.0f} | {t['llm_ms'] + t['agent_ms']:.0f} | {t['total_ms']:.0f} | {steps} |")


def main():
    if not (os.environ.get("TYPESAFE_API_KEY") and os.environ.get("DEEPSEEK_API_KEY")):
        sys.exit("需要 TYPESAFE_API_KEY 和 DEEPSEEK_API_KEY")
    L = ["# 演示系统真实调用检查\n"]
    head = "| 句子 | 去向 | 大脑标签 | 会话状态 | is_fast | Jev ms | 大模型 ms | 端到端 ms | 各步骤 |\n|---|---|---|---|---|---|---|---|---|"
    with TestClient(app) as c:
        cfg = c.get("/api/config").json()["backends"]
        L.append(f"- 服务：Jev {cfg['jev']}（{cfg['jev_model']}），大模型 {cfg['llm']}（{cfg['llm_model']}），"
                 f"快思考 {cfg['fast']}，联网搜索 {cfg.get('search')}\n")
        L += ["## 单句\n", head]
        searched = []
        for i, (q, mode) in enumerate(SINGLE):
            r = run(c, q, f"single-{i}", mode)
            L.append(fmt(r) + f" ({mode})")
            se = next((s for s in r["steps"] if s["kind"] == "search"), None)
            if se:
                searched.append((r, se))
        if searched:
            L.append(f"\n## 联网搜索（{cfg.get('search_provider')}）\n")
            for r, se in searched:
                d = se["detail"]
                if se["status"] == "error":
                    L.append(f"- {r['query']}：失败 {se['summary']}")
                    continue
                L.append(f"- {r['query']}：{d['tool']}，耗时 {se['end'] - se['start']:.0f} ms，搜索词 {d.get('queries')}，"
                         f"来源 {len(d['items'])} 条，用量 {d.get('usage')}；播报：{(r.get('brain') or {}).get('speech', '')}")
                L += [f"  - {it['title']}" for it in d["items"][:5]]
        L += ["\n## 多轮：点外卖途中调空调\n", head]
        for q in DIALOG:
            L.append(fmt(run(c, q, "dialog", "label")))
        samples = c.get("/api/samples").json()["builtin"]
        rows = [json.loads(l) for l in c.post("/api/batch", json={"items": samples, "concurrency": 4}).text.splitlines()]
        dj = [r for r in rows if r.get("correct", {}).get("dest") is not None]
        lj = [r for r in rows if r.get("correct", {}).get("label") is not None]
        L.append(f"\n## 内置样例 {len(rows)} 句（标签模式）\n")
        L.append(f"- 去向正确 {sum(r['correct']['dest'] for r in dj)}/{len(dj)}；大脑首选标签正确 {sum(r['correct']['label'] for r in lj)}/{len(lj)}，"
                 f"包含正确答案 {sum(r['correct']['label_any'] for r in lj)}/{len(lj)}；出错 {sum('error' in r for r in rows)}")
        L.append("\n| 句子 | 应去 | 实际 | 应有标签 | 大脑标签 | 端到端 ms |\n|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: r["index"]):
            res = r.get("result") or {}
            ok = "" if r.get("correct", {}).get("dest") is not False and r.get("correct", {}).get("label") is not False else " ✗"
            L.append(f"| {r['item']['query']} | {r['item']['dest']} | {res.get('destination')} | {'、'.join(r['item']['labels'] or [])} | "
                     f"{'、'.join(res.get('labels') or [])}{ok} | {res.get('timing', {}).get('total_ms', 0):.0f} |")
    text = "\n".join(L) + "\n"
    Path("runs/webdemo").mkdir(parents=True, exist_ok=True)
    Path("runs/webdemo/report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
