#!/usr/bin/env python3
"""前置分流评测。

第 14 版（默认，只发第一级请求）：
    第一级  is_fast、needs_rewrite 和 criteria 里的 Jev 纠错题（都是 Noul，一次请求）
    去向    纠错题命中按它的 route 走；否则 is_fast 达到阈值交给快思考，没达到交给 Agent 大脑
    报告    第一级准确率，以及每道纠错题的回放：会改变多少句的去向，其中改对、改错各多少
第 13 版（--stages s1,s2）：再对每条样本问第二级 6 选 1，报告第二级和两级组合的结果。

标准答案由评测集原来的 label 换算：
    16 个 NLU domain 和中枢简单任务 → 快思考
    用车agent → 用车手册Agent；新闻agent → 联网搜索Agent；任务大师、条件任务不变
    音乐、影视、复杂导航、记忆、车控模糊意图、中枢复杂多任务、闲聊 → Agent大脑

常用：
    python eval_two_stage.py --split dev --per-label 20 --out runs/v14-dev
    python eval_two_stage.py --split dev --per-label 20 --stages s1,s2 --out runs/2stage-dev
    python eval_two_stage.py --split dev --per-label 5 --mock --out runs/mock2
"""
import argparse, asyncio, collections, csv, hashlib, json, os, random, sys
from pathlib import Path
from types import SimpleNamespace

from eval_jev import load_jsonl, select, build_state

HERE = Path(__file__).resolve().parent
FAST = {"carControl", "windowControl", "systemControl", "music", "mediaControl", "generalControl", "video",
        "unicast", "naviControl", "navi", "autoPilot", "relaxCapsule", "carInfo", "scheduleReminder", "phone",
        "memoryRecord", "中枢简单任务"}
STAGE2_OF = {"用车agent": "用车手册Agent", "新闻agent": "联网搜索Agent", "闲聊": "Agent大脑", "任务大师": "任务大师",
             "条件任务": "条件任务", "音乐agent": "Agent大脑", "影视agent": "Agent大脑", "复杂导航agent": "Agent大脑",
             "记忆agent": "Agent大脑", "车控agent模糊意图": "Agent大脑", "中枢复杂多任务": "Agent大脑"}
BACK = "退回快思考"


def gold_of(it):
    g = it["gold"]
    fast_ok = any(x in FAST for x in g)
    s2 = {STAGE2_OF[x] for x in g if x not in FAST}
    dest = set(s2) | ({"快思考"} if fast_ok else set())
    return fast_ok, bool(s2), (s2 | ({BACK} if fast_ok else set())), dest


def active_corrections(crit):
    """回放时 shadow 和 on 的纠错题都要问；off 的已下线。"""
    return {k: v for k, v in crit.get("corrections", {}).items() if v.get("status") != "off"}


def stage1_questions(crit, it, Noul):
    c = dict(crit["stage1"], **active_corrections(crit))
    return {k: Noul(instructions=c[k]["instructions"], criteria=c[k]["criteria"]) for k in c}


def stage2_questions(crit, use_examples, Choice):
    opts = {}
    for k, v in crit["stage2"]["options"].items():
        d = dict(v)
        if not use_examples:
            d.pop("examples", None)
        opts[k] = d
    return {"label": Choice(instructions=crit["stage2"]["instructions"], criteria=opts)}


class MockClient:
    """只验证流程，结果无意义。"""
    def __init__(self, labels):
        self.labels = labels

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def system_one(self, state, questions, model=None):
        rnd = random.Random(int(hashlib.md5(state["query"].encode()).hexdigest(), 16))
        ans = {}
        for k in questions:
            if k == "label":
                pick = rnd.choice(self.labels)
                ans[k] = SimpleNamespace(choice=pick, confidence=0.8,
                                         probabilities={l: (0.8 if l == pick else 0.2 / (len(self.labels) - 1)) for l in self.labels})
            else:
                ans[k] = SimpleNamespace(noul=rnd.random())
        return SimpleNamespace(answers=ans, usage=SimpleNamespace(input_tokens=0))


async def run(args, crit, items):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "predictions.jsonl"
    done = {(p["id"], p["stage"]) for p in load_jsonl(path)} if path.exists() else set()
    stages = args.stages.split(",")
    jobs = [(it, st) for it in items for st in stages
            if (it["id"], st) not in done and not (st == "s2" and it.get("fixture"))]
    print(f"待请求 {len(jobs)} 次（已有 {len(done)} 次）")
    try:
        from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy
    except ImportError:
        if not args.mock:
            sys.exit("缺少 SDK：pip install typesafe-sdk")
        Choice = Noul = lambda **kw: kw
    if args.mock:
        cm = MockClient(list(crit["stage2"]["options"]))
    else:
        if not os.environ.get("TYPESAFE_API_KEY"):
            sys.exit("没有找到 TYPESAFE_API_KEY")
        cm = AsyncTypeSafeClient(model=args.model, retry=RetryPolicy(max_retries=5))
    sem, lock, cnt = asyncio.Semaphore(args.concurrency), asyncio.Lock(), {"n": 0, "err": 0}
    async with cm as client:
        with open(path, "a", encoding="utf-8") as fo:
            async def one(it, st):
                qs = stage1_questions(crit, it, Noul) if st == "s1" else stage2_questions(crit, not args.no_examples, Choice)
                rec = {"id": it["id"], "stage": st}
                async with sem:
                    try:
                        r = await client.system_one(state=build_state(it), questions=qs)
                        a = r.answers
                        if st == "s1":
                            for k in qs:
                                rec[k] = round(a[k].noul, 4)
                        else:
                            rec.update(choice=a["label"].choice, confidence=round(a["label"].confidence, 4),
                                       probabilities={k: round(v, 4) for k, v in a["label"].probabilities.items()})
                        rec["input_tokens"] = getattr(r.usage, "input_tokens", None)
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}: {e}"[:300]
                async with lock:
                    if "error" in rec:
                        cnt["err"] += 1
                    else:
                        fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
                    cnt["n"] += 1
                    if cnt["n"] % 200 == 0 or cnt["n"] == len(jobs):
                        print(f"  {cnt['n']}/{len(jobs)}  失败 {cnt['err']}", flush=True)
            await asyncio.gather(*(one(it, st) for it, st in jobs))
    return cnt["err"]


def pct(a, b):
    return f"{a / b:.1%}" if b else "-"


def report(args, crit, items, fixtures, n_err):
    out = Path(args.out)
    P = collections.defaultdict(dict)
    for p in load_jsonl(out / "predictions.jsonl"):
        P[p["id"]][p["stage"]] = p
    rows = [(it, P[it["id"]]["s1"], P[it["id"]].get("s2")) for it in items if "s1" in P[it["id"]]]
    has_s2 = all(s2 for _, _, s2 in rows)
    if not rows:
        print("没有可统计的结果"); return
    n = len(rows)
    L = [f"# 两级分流评测：{out.name}\n", f"- 判据：{crit.get('version', '')}",
         f"- 样本：{n} 条（split={args.split}" + (f"，按原 label 每类最多 {args.per_label} 条" if args.per_label else "") + "）",
         f"- 模型：{'模拟（结果无意义）' if args.mock else args.model}；第二级例句：{'不带' if args.no_examples else '带'}"]
    if n_err:
        L.append(f"- 本次失败 {n_err} 次请求，未计入；再跑一次同样的命令会补上")
    t1 = [s1["input_tokens"] for _, s1, _ in rows if s1.get("input_tokens")]
    t2 = [s2["input_tokens"] for _, _, s2 in rows if s2 and s2.get("input_tokens")]
    if t1:
        L.append(f"- 平均输入 token：第一级 {sum(t1) // len(t1):,}" + (f"，第二级 {sum(t2) // len(t2):,}" if t2 else ""))

    G = {it["id"]: gold_of(it) for it, _, _ in rows}
    only_fast = [(it, s1, s2) for it, s1, s2 in rows if G[it["id"]][0] and not G[it["id"]][1]]
    only_agent = [(it, s1, s2) for it, s1, s2 in rows if G[it["id"]][1] and not G[it["id"]][0]]
    both = n - len(only_fast) - len(only_agent)

    # ---------- 第一级 ----------
    L.append("\n## 第一级：是否交给快思考\n")
    L.append(f"应交给快思考 {len(only_fast)} 条，不应交给 {len(only_agent)} 条，两种都算对 {both} 条（如条件任务里的纯提醒）。\n")
    L.append("| is_fast 阈值 | 该走 Agent 却交给快思考 | 该走快思考却交给大脑 | 第一级准确率 |\n|---|---|---|---|")
    for T in (0.3, 0.5, 0.7, 0.8, 0.9):
        fp = sum(s1["is_fast"] >= T for _, s1, _ in only_agent)
        fn = sum(s1["is_fast"] < T for _, s1, _ in only_fast)
        L.append(f"| {T} | {pct(fp, len(only_agent))} ({fp}) | {pct(fn, len(only_fast))} ({fn}) | {pct(n - fp - fn, n)} |")

    for T in (0.5, 0.8):
        fp = sorted(((s1["is_fast"], it) for it, s1, _ in only_agent if s1["is_fast"] >= T), key=lambda x: -x[0])
        fn = sorted(((s1["is_fast"], it) for it, s1, _ in only_fast if s1["is_fast"] < T), key=lambda x: x[0])
        L.append(f"\n阈值 {T} 时的错误，按原 label 统计：")
        L.append("- 该走 Agent 却交给快思考：" + ("，".join(f"{k} {v}" for k, v in collections.Counter(it["primary"] for _, it in fp).most_common()) or "无"))
        L.append("- 该走快思考却交给大脑：" + ("，".join(f"{k} {v}" for k, v in collections.Counter(it["primary"] for _, it in fn).most_common()) or "无"))
    L.append("\n阈值 0.5 时该走 Agent 却交给快思考的句子：\n\n| is_fast | 原 label | 句子 |\n|---|---|---|")
    for sc, it in sorted(((s1["is_fast"], it) for it, s1, _ in only_agent if s1["is_fast"] >= 0.5), key=lambda x: -x[0])[:60]:
        L.append(f"| {sc} | {it['primary']} | {it['query'].replace('|', '/')[:60]} |")
    L.append("\n阈值 0.8 时该走快思考却交给大脑的句子：\n\n| is_fast | 原 label | 句子 |\n|---|---|---|")
    for sc, it in sorted(((s1["is_fast"], it) for it, s1, _ in only_fast if s1["is_fast"] < 0.8), key=lambda x: x[0])[:60]:
        L.append(f"| {sc} | {it['primary']} | {it['query'].replace('|', '/')[:60]} |")

    # ---------- Jev 纠错题回放 ----------
    corr = {k: v for k, v in active_corrections(crit).items() if all(k in s1 for _, s1, _ in rows)}
    if corr:
        T, CT = 0.7, 0.5
        gold14 = {i: ({"快思考"} if g[0] else set()) | ({"Agent大脑"} if g[1] else set()) for i, g in G.items()}
        base = {it["id"]: ("快思考" if s1["is_fast"] >= T else "Agent大脑") for it, s1, _ in rows}
        L.append(f"\n## Jev 纠错题回放（is_fast 阈值 {T}，纠错题阈值 {CT}）\n")
        L.append("回放时不管题目当前是 shadow 还是 on，都按命中后改变去向来算。\n")
        L.append("| 纠错题 | 状态 | 命中后去向 | 命中 | 改变去向 | 改对 | 改错 |\n|---|---|---|---|---|---|---|")
        changed_rows = {}
        for k, v in corr.items():
            hit = [(it, s1) for it, s1, _ in rows if s1[k] >= CT]
            ch = [(it, s1) for it, s1 in hit if v["route"] != base[it["id"]]]
            fixed = sum(v["route"] in gold14[it["id"]] and base[it["id"]] not in gold14[it["id"]] for it, _ in ch)
            broke = sum(base[it["id"]] in gold14[it["id"]] and v["route"] not in gold14[it["id"]] for it, _ in ch)
            L.append(f"| {k} | {v.get('status')} | {v['route']}" + (f"（提示 {v['hint']}）" if v.get("hint") else "")
                     + f" | {len(hit)} | {len(ch)} | {fixed} | {broke} |")
            changed_rows[k] = ch

        def dest14(it, s1):
            hits = sorted(((s1[k], v["route"]) for k, v in corr.items() if s1[k] >= CT), reverse=True)
            return hits[0][1] if hits else base[it["id"]]
        ok0 = sum(base[it["id"]] in gold14[it["id"]] for it, _, _ in rows)
        ok1 = sum(dest14(it, s1) in gold14[it["id"]] for it, s1, _ in rows)
        L.append(f"\n- 去向准确率：不用纠错题 {pct(ok0, n)}（{ok0}/{n}），全部启用后 {pct(ok1, n)}（{ok1}/{n}）")
        for k, ch in changed_rows.items():
            if ch:
                L.append(f"\n{k} 改变去向的句子（最多 30 条）：\n\n| 得分 | 原 label | 改对了吗 | 句子 |\n|---|---|---|---|")
                for it, s1 in sorted(ch, key=lambda x: -x[1][k])[:30]:
                    good = "对" if corr[k]["route"] in gold14[it["id"]] else "错"
                    L.append(f"| {s1[k]} | {it['primary']} | {good} | {it['query'].replace('|', '/')[:60]} |")
        t1c = [s1["input_tokens"] for _, s1, _ in rows if s1.get("input_tokens")]
        if t1c:
            L.append(f"\n- 第一级请求（含 {len(corr)} 道纠错题）平均输入 token：{sum(t1c) // len(t1c):,}")

    # ---------- 第二级（第 13 版） ----------
    s2rows = [(it, s1, s2) for it, s1, s2 in rows if G[it["id"]][1] and s2]
    ok2 = [s2["choice"] in G[it["id"]][2] for it, _, s2 in s2rows]
    if s2rows:
        L.append("\n## 第二级：单独看（只用应走 Agent 的样本）\n")
        L.append(f"- 准确率：{pct(sum(ok2), len(ok2))}（{sum(ok2)}/{len(ok2)}）")
        pred = collections.Counter(s2["choice"] for _, _, s2 in s2rows)
        L.append("- 预测分布：" + "，".join(f"{k} {v}" for k, v in pred.most_common()))
        rec = collections.defaultdict(lambda: [0, 0]); conf = collections.Counter(); ex = collections.defaultdict(list)
        for (it, _, s2), ok in zip(s2rows, ok2):
            g = sorted(G[it["id"]][2] - {BACK})[0]
            rec[g][1] += 1; rec[g][0] += ok
            if not ok:
                conf[(g, s2["choice"])] += 1; ex[(g, s2["choice"])].append(it["query"])
        L.append("\n| 应为 | 样本 | 召回 |\n|---|---|---|")
        for k, (a, b) in sorted(rec.items(), key=lambda x: x[1][0] / x[1][1]):
            L.append(f"| {k} | {b} | {pct(a, b)} |")
        L.append("\n| 应为 | 判成 | 次数 | 例句 |\n|---|---|---|---|")
        for (g, f), c in conf.most_common(20):
            L.append(f"| {g} | {f} | {c} | {'；'.join(q.replace('|', '/')[:40] for q in ex[(g, f)][:3])} |")

        # ---------- 端到端 ----------
        def dest(it, s1, s2, T):
            if s1["is_fast"] >= T:
                return "快思考"
            return "快思考" if s2["choice"] == BACK else s2["choice"]
        L.append("\n## 端到端：最终去向\n")
        L.append("| is_fast 阈值 | 去向准确率 | 去快思考 | 去 Agent 大脑 | 去其他 |\n|---|---|---|---|---|")
        for T in (0.3, 0.5, 0.7, 0.8, 0.9):
            ds = [dest(it, s1, s2, T) for it, s1, s2 in rows]
            ok = sum(d in G[it["id"]][3] for d, (it, _, _) in zip(ds, rows))
            c = collections.Counter(ds)
            L.append(f"| {T} | {pct(ok, n)} ({ok}/{n}) | {pct(c['快思考'], n)} | {pct(c['Agent大脑'], n)} | {pct(n - c['快思考'] - c['Agent大脑'], n)} |")
        L.append("\n参照：第 3 版 28 选 1 在同一批 409 条上的分支准确率是 97.8%（v0.5 判据），分支口径不同，仅作量级参考。")

    # ---------- needs_rewrite ----------
    L.append("\n## needs_rewrite：模糊指代\n")
    fx = [(f, P[f["id"]].get("s1")) for f in fixtures if P[f["id"]].get("s1")]
    if fx:
        pos = [s for f, s in fx if f["needs_rewrite"]]; neg = [s for f, s in fx if not f["needs_rewrite"]]
        L.append(f"人工构造的 {len(fx)} 条（{len(pos)} 条有模糊指代，{len(neg)} 条没有），阈值 0.5：")
        L.append(f"- 识别出模糊指代：{pct(sum(s['needs_rewrite'] >= .5 for s in pos), len(pos))}；误报：{pct(sum(s['needs_rewrite'] >= .5 for s in neg), len(neg))}")
        L.append(f"- 这些句子的 is_fast ≥ 0.5：{pct(sum(s['is_fast'] >= .5 for f, s in fx if f['is_fast']), sum(f['is_fast'] for f, _ in fx))}")
        miss = [f["query"] for f, s in fx if f["needs_rewrite"] and s["needs_rewrite"] < .5]
        if miss:
            L.append("- 漏掉的：" + "；".join(miss))
    flagged = sorted(((s1["needs_rewrite"], it["query"]) for it, s1, _ in rows if s1["needs_rewrite"] >= .5), reverse=True)
    L.append(f"- 评测集样本里被判为有模糊指代的：{len(flagged)} 条" + ("，例如：" + "；".join(q for _, q in flagged[:12]) if flagged else ""))

    text = "\n".join(L) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    with open(out / "errors.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "上一轮", "原 label", "应去", "is_fast", "第二级", "第二级置信度", "needs_rewrite"])
        for it, s1, s2 in (rows if s2rows else []):
            d = dest(it, s1, s2, 0.5)
            if d not in G[it["id"]][3]:
                w.writerow([it["id"], it["query"], it.get("prev_query") or "", "/".join(it["gold"]), "/".join(sorted(G[it["id"]][3])),
                            s1["is_fast"], s2["choice"], s2["confidence"], s1["needs_rewrite"]])
    print(text)
    print(f"已写入 {out}/report.md、errors.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "eval_set.jsonl"))
    ap.add_argument("--criteria", default=str(HERE / "criteria_v2.json"))
    ap.add_argument("--fuzzy", default=str(HERE / "fixtures" / "fuzzy_refs.jsonl"), help="模糊指代样本，空字符串表示不跑")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "val", "test", "all"])
    ap.add_argument("--per-label", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no-examples", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--stages", default="s1", help="要请求的级别：s1（第 14 版，默认）或 s1,s2（第 13 版两级）")
    args = ap.parse_args()

    crit = json.loads(Path(args.criteria).read_text(encoding="utf-8"))
    items = select(load_jsonl(args.data), args.split, args.per_label, args.seed)
    unknown = {g for it in items for g in it["gold"]} - FAST - set(STAGE2_OF)
    if unknown:
        sys.exit(f"评测集里有无法换算的标签：{unknown}")
    fixtures = load_jsonl(args.fuzzy) if args.fuzzy else []
    fx_items = [{"id": f["id"], "query": f["query"], "fixture": True} for f in fixtures]
    n_err = 0
    if not args.report_only:
        n_err = asyncio.run(run(args, crit, items + fx_items))
    report(args, crit, items, fixtures, n_err)


if __name__ == "__main__":
    main()
