#!/usr/bin/env python3
"""用 criteria.json 里的例句生成冒烟测试集 fixtures/smoke_set.jsonl。

这些例句取自测试集的 dev 部分，标签就是它所在的选项，所以不需要 xlsx 也能先验证 Jev 是否连通、
请求结构是否被接受。例句同时出现在提示里，准确率会偏高，只用来冒烟，不代表真实效果；
想看例句之外的效果，用 --no-examples 跑，或者用 build_dataset.py 生成完整评测集。
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
crit = json.loads((HERE / "criteria.json").read_text(encoding="utf-8"))
routes = crit["routes"]
out = HERE / "fixtures" / "smoke_set.jsonl"
out.parent.mkdir(exist_ok=True)
seen = {}
for label, opt in crit["label"]["options"].items():
    for q in opt.get("examples", []):
        seen.setdefault(q, []).append(label)
with open(out, "w", encoding="utf-8") as f:
    for i, (q, gold) in enumerate(seen.items()):
        row = {"id": f"s{i:04d}", "query": q, "gold": gold, "primary": gold[0],
               "routes": sorted({routes[g] for g in gold}), "split": "dev", "turn": 1,
               "prev_query": None, "prev_label": None, "now_playing": None,
               "is_global": False, "categories": [], "sources": ["criteria.examples"], "review": []}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"{len(seen)} 条 -> {out.relative_to(HERE)}")
