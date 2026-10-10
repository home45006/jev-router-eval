#!/usr/bin/env python3
"""用 Jev 给评测集打标签，并输出准确率、混淆和阈值分析。

准备:
    pip install typesafe-sdk
    export TYPESAFE_API_KEY=...

常用:
    # 先小样本看一眼（每个标签最多 20 条，dev 部分）
    python eval_jev.py --split dev --per-label 20 --out runs/dev-v0.1

    # 改完 criteria.json 后在 val 上定阈值，最后在 test 上出结果
    python eval_jev.py --split val  --out runs/val-v0.1
    python eval_jev.py --split test --out runs/test-v0.1

    # 检查选项顺序是否影响结果（Jev 会偏向靠前的选项）
    python eval_jev.py --split dev --per-label 20 --shuffle-options 1 --out runs/dev-shuffle1

    # 不带例句跑一遍，看例句值不值得加
    python eval_jev.py --split dev --per-label 20 --no-examples --out runs/dev-noex

    # 没有密钥时验证流程能跑通（结果是假的）
    python eval_jev.py --split dev --per-label 5 --mock --out runs/mock

每次请求问三个问题，Jev 并行作答:
    label        Choice，28 个标签选一个（NLU 侧 16 个预期domain + Agent 侧 12 个 label）
    is_multi     Noul，是否包含多条指令（只做诊断，不改标签）
    is_followup  Noul，是否是对上一轮的追问（只在有上一轮时问）

最终标签:
    上一轮标签是“闲聊”且 is_followup >= 阈值  ->  闲聊
    否则                                      ->  label 的选择

同一个 --out 目录可以重复运行，已有结果的样本会跳过。
"""
import argparse
import asyncio
import collections
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# 输入
# ----------------------------------------------------------------------------
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select(items, split, per_label, seed):
    if split != "all":
        items = [it for it in items if it["split"] == split]
    if per_label:
        rnd = random.Random(seed)
        by = collections.defaultdict(list)
        for it in items:
            by[it["primary"]].append(it)
        items = []
        for label in sorted(by):
            pool = by[label]
            rnd.shuffle(pool)
            items.extend(pool[:per_label])
    return items


def build_state(it):
    state = {"query": it["query"]}
    if it.get("prev_label"):
        prev = {"label": it["prev_label"]}
        if it.get("prev_query"):
            prev["query"] = it["prev_query"]
        state["previous_turn"] = prev
    if it.get("now_playing"):
        state["now_playing"] = it["now_playing"]
    return state


def build_questions(crit, it, order, use_examples, Choice, Noul):
    options = {}
    for name in order:
        desc = dict(crit["label"]["options"][name])
        if not use_examples:
            desc.pop("examples", None)
        options[name] = desc
    questions = {
        "label": Choice(instructions=crit["label"]["instructions"], criteria=options),
        "is_multi": Noul(instructions=crit["is_multi"]["instructions"],
                         criteria=crit["is_multi"]["criteria"]),
    }
    if it.get("prev_query"):
        questions["is_followup"] = Noul(instructions=crit["is_followup"]["instructions"],
                                        criteria=crit["is_followup"]["criteria"])
    return questions


# ----------------------------------------------------------------------------
# 模拟客户端：只用来验证流程，不代表 Jev 的表现
# ----------------------------------------------------------------------------
class MockClient:
    def __init__(self, items, labels):
        self.gold = {json.dumps(build_state(it), ensure_ascii=False, sort_keys=True): it for it in items}
        self.labels = labels

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def system_one(self, state, questions, model=None):
        it = self.gold[json.dumps(state, ensure_ascii=False, sort_keys=True)]
        rnd = random.Random(int(hashlib.md5(it["id"].encode()).hexdigest(), 16))
        right = rnd.random() < 0.85
        pick = it["primary"] if right else rnd.choice(self.labels)
        top = rnd.uniform(0.6, 0.99) if right else rnd.uniform(0.3, 0.7)
        rest = (1 - top) / (len(self.labels) - 1)
        probs = {name: (top if name == pick else rest) for name in self.labels}
        answers = {
            "label": SimpleNamespace(choice=pick, confidence=top, probabilities=probs),
            "is_multi": SimpleNamespace(noul=0.9 if it["primary"].startswith("中枢") else 0.05),
        }
        if "is_followup" in questions:
            answers["is_followup"] = SimpleNamespace(noul=rnd.uniform(0.6, 1.0))
        return SimpleNamespace(answers=answers, model="mock",
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0))


# ----------------------------------------------------------------------------
# 调用
# ----------------------------------------------------------------------------
async def run(args, crit, items, order):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = {p["id"] for p in load_jsonl(pred_path)} if pred_path.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    if done:
        print(f"已有 {len(done)} 条结果，本次补 {len(todo)} 条")

    try:
        from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy
    except ImportError:
        if not args.mock:
            sys.exit("缺少 SDK：pip install typesafe-sdk")
        Choice = Noul = lambda **kw: kw            # 仅模拟模式下的占位

    if args.mock:
        client_cm = MockClient(items, order)
    else:
        if not os.environ.get("TYPESAFE_API_KEY"):
            sys.exit("没有找到 TYPESAFE_API_KEY。先 export TYPESAFE_API_KEY=...，或加 --mock 验证流程。")
        client_cm = AsyncTypeSafeClient(model=args.model, retry=RetryPolicy(max_retries=5))

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    counter = {"n": 0, "err": 0}

    async with client_cm as client:
        with open(pred_path, "a", encoding="utf-8") as fout:
            async def one(it):
                state = build_state(it)
                questions = build_questions(crit, it, order, not args.no_examples, Choice, Noul)
                rec = {"id": it["id"]}
                async with sem:
                    try:
                        resp = await client.system_one(state=state, questions=questions)
                        ans = resp.answers
                        rec.update(
                            choice=ans["label"].choice,
                            confidence=round(ans["label"].confidence, 4),
                            probabilities={k: round(v, 4) for k, v in ans["label"].probabilities.items()},
                            is_multi=round(ans["is_multi"].noul, 4),
                            is_followup=round(ans["is_followup"].noul, 4) if "is_followup" in ans else None,
                            input_tokens=getattr(resp.usage, "input_tokens", None),
                        )
                    except Exception as e:                    # 单条失败不影响其余样本
                        rec["error"] = f"{type(e).__name__}: {e}"[:300]
                async with lock:
                    if "error" in rec:
                        counter["err"] += 1                   # 失败的不落盘，下次运行会重试
                    else:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
                    counter["n"] += 1
                    if counter["n"] % 200 == 0 or counter["n"] == len(todo):
                        print(f"  {counter['n']}/{len(todo)}  失败 {counter['err']}", flush=True)

            await asyncio.gather(*(one(it) for it in todo))
    return counter["err"]


# ----------------------------------------------------------------------------
# 统计
# ----------------------------------------------------------------------------
def pct(a, b):
    return f"{a / b:.1%}" if b else "-"


def report(args, crit, items, order, n_err):
    out = Path(args.out)
    preds = {p["id"]: p for p in load_jsonl(out / "predictions.jsonl")}
    rows = [(it, preds[it["id"]]) for it in items if it["id"] in preds]
    if not rows:
        print("没有可统计的结果")
        return
    routes = crit["routes"]
    T = args.followup_threshold

    def final_label(it, p):
        if it.get("prev_label") == "闲聊" and p.get("is_followup") is not None and p["is_followup"] >= T:
            return "闲聊"
        return p["choice"]

    n = len(rows)
    ok_choice = sum(p["choice"] in it["gold"] for it, p in rows)
    finals = [(it, p, final_label(it, p)) for it, p in rows]
    ok_final = sum(f in it["gold"] for it, p, f in finals)
    ok_route = sum(routes[f] in it["routes"] for it, p, f in finals)
    tokens = [p["input_tokens"] for _, p in rows if p.get("input_tokens")]

    L = []
    L.append(f"# Jev 标签评测：{out.name}\n")
    L.append(f"- 判据：{crit.get('version', '')}")
    L.append(f"- 样本：{n} 条（split={args.split}"
             + (f"，每个标签最多 {args.per_label} 条" if args.per_label else "") + "）")
    L.append(f"- 模型：{'模拟（结果无意义，仅验证流程）' if args.mock else args.model}")
    L.append(f"- 选项顺序：{'打乱，种子 ' + str(args.shuffle_options) if args.shuffle_options is not None else '按 criteria.json'}"
             f"；例句：{'不带' if args.no_examples else '带'}")
    if n_err:
        L.append(f"- 本次失败 {n_err} 条，未计入统计；再运行一次同样的命令会重试")
    if tokens:
        L.append(f"- 输入 token：合计 {sum(tokens):,}，平均每次请求 {sum(tokens) // len(tokens):,}")
    L.append("\n判对的口径：预测落在样本的任一可接受标签（gold）里就算对。\n")

    L.append("## 总体\n")
    L.append("| 指标 | 结果 |\n|---|---|")
    L.append(f"| 标签准确率，只看 Choice | {pct(ok_choice, n)} ({ok_choice}/{n}) |")
    L.append(f"| 标签准确率，加上“闲聊追问跟随上一轮” | {pct(ok_final, n)} ({ok_final}/{n}) |")
    L.append(f"| 分支准确率 | {pct(ok_route, n)} ({ok_route}/{n}) |")

    # 各标签
    rec = collections.defaultdict(lambda: [0, 0])
    prec = collections.defaultdict(lambda: [0, 0])
    conf = collections.Counter()
    conf_ex = collections.defaultdict(list)
    for it, p, f in finals:
        rec[it["primary"]][1] += 1
        prec[f][1] += 1
        if f in it["gold"]:
            rec[it["primary"]][0] += 1
            prec[f][0] += 1
        else:
            conf[(it["primary"], f)] += 1
            conf_ex[(it["primary"], f)].append(it["query"])
    L.append("\n## 各标签\n")
    L.append("| 标签 | 分支 | 样本 | 召回 | 被预测次数 | 精确 |\n|---|---|---|---|---|---|")
    for name in sorted(order, key=lambda k: (rec[k][0] / rec[k][1]) if rec[k][1] else 2):
        r, pr = rec[name], prec[name]
        L.append(f"| {name} | {routes[name]} | {r[1]} | {pct(*r)} | {pr[1]} | {pct(*pr)} |")

    L.append("\n## 最常见的混淆\n")
    L.append("| 应为 | 判成 | 次数 | 例句 |\n|---|---|---|---|")
    for (g, f), c in conf.most_common(25):
        ex = "；".join(q.replace("|", "/")[:40] for q in conf_ex[(g, f)][:3])
        L.append(f"| {g} | {f} | {c} | {ex} |")

    # 置信度
    L.append("\n## 置信度阈值\n")
    L.append("达到阈值的按标签分发；达不到的看“分支概率”（同一分支下各标签概率之和）最高的分支是否正确。\n")
    L.append("| 阈值 | 达标占比 | 达标部分的标签准确率 | 未达标部分的分支准确率 |\n|---|---|---|---|")

    def best_route(p):
        s = collections.Counter()
        for name, v in p["probabilities"].items():
            s[routes[name]] += v
        return s.most_common(1)[0][0]

    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        hi = [(it, p, f) for it, p, f in finals if p["confidence"] >= th]
        lo = [(it, p, f) for it, p, f in finals if p["confidence"] < th]
        hi_ok = sum(f in it["gold"] for it, p, f in hi)
        lo_ok = sum(best_route(p) in it["routes"] for it, p, f in lo)
        L.append(f"| {th} | {pct(len(hi), n)} | {pct(hi_ok, len(hi))} | {pct(lo_ok, len(lo))} ({len(lo)} 条) |")

    # 多指令
    multi = [(it, p) for it, p in rows if it["primary"] in ("中枢简单任务", "中枢复杂多任务")]
    single = [(it, p) for it, p in rows if it["primary"] not in ("中枢简单任务", "中枢复杂多任务", "任务大师", "条件任务")]
    L.append("\n## 多指令判断 is_multi（阈值 0.5）\n")
    L.append(f"- 中枢简单任务 / 中枢复杂多任务 里判为多指令：{pct(sum(p['is_multi'] >= 0.5 for _, p in multi), len(multi))}（{len(multi)} 条）")
    L.append(f"- 其他单指令标签里误判为多指令：{pct(sum(p['is_multi'] >= 0.5 for _, p in single), len(single))}（{len(single)} 条）")

    # 追问
    fu = [(it, p) for it, p in rows if p.get("is_followup") is not None]
    if fu:
        L.append(f"\n## 追问判断 is_followup（阈值 {T}）\n")
        L.append(f"- 多轮样本 {len(fu)} 条，判为追问：{pct(sum(p['is_followup'] >= T for _, p in fu), len(fu))}")
        L.append("- 数据里的多轮样本全部是闲聊追问，这里只能测到“该跟随的有没有跟随”，测不到“闲聊中途说了一条新指令能不能跳出来”。后者需要补样本。")

    text = "\n".join(L) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    with open(out / "confusion.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["应为\\判成"] + order)
        mat = collections.Counter((it["primary"], fl) for it, p, fl in finals)
        for g in order:
            w.writerow([g] + [mat[(g, fl)] for fl in order])
    with open(out / "errors.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "上一轮", "可接受标签", "预测", "置信度", "第二候选", "待确认备注"])
        for it, p, fl in finals:
            if fl not in it["gold"]:
                second = sorted(p["probabilities"].items(), key=lambda kv: -kv[1])[1]
                w.writerow([it["id"], it["query"], it.get("prev_query") or "", " / ".join(it["gold"]), fl,
                            p["confidence"], f"{second[0]} {second[1]}", "；".join(it.get("review", []))])
    print(text)
    print(f"已写入 {out}/report.md、confusion.csv、errors.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "eval_set.jsonl"))
    ap.add_argument("--criteria", default=str(HERE / "criteria.json"))
    ap.add_argument("--out", required=True, help="本次运行的输出目录")
    ap.add_argument("--split", default="dev", choices=["dev", "val", "test", "all"])
    ap.add_argument("--per-label", type=int, default=0, help="每个标签最多取多少条，0 表示全部")
    ap.add_argument("--seed", type=int, default=0, help="抽样种子")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--shuffle-options", type=int, default=None, metavar="SEED", help="打乱选项顺序")
    ap.add_argument("--no-examples", action="store_true", help="选项里不带例句")
    ap.add_argument("--followup-threshold", type=float, default=0.5)
    ap.add_argument("--mock", action="store_true", help="不调用 Jev，用假结果验证流程")
    ap.add_argument("--report-only", action="store_true", help="不发请求，只根据已有结果重新出报告")
    args = ap.parse_args()

    crit = json.loads(Path(args.criteria).read_text(encoding="utf-8"))
    items = select(load_jsonl(args.data), args.split, args.per_label, args.seed)
    order = list(crit["label"]["options"])
    if args.shuffle_options is not None:
        random.Random(args.shuffle_options).shuffle(order)
    unknown = {g for it in items for g in it["gold"]} - set(order)
    if unknown:
        sys.exit(f"评测集里有 criteria.json 没有的标签：{unknown}")

    n_err = 0
    if not args.report_only:
        print(f"待评测 {len(items)} 条，并发 {args.concurrency}")
        n_err = asyncio.run(run(args, crit, items, order))
    report(args, crit, items, order, n_err)


if __name__ == "__main__":
    main()
