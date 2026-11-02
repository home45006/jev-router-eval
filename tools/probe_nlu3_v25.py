"""NLU3.0 v25 模型（8048，OpenAI 兼容接口）落域摸底。接口和格式按 docs/nlu3/NLU3.0大模型接口使用说明1030固定版本.md。

    NLU3_V25_URL=http://<host>:<port>/v1/chat/completions python tools/probe_nlu3_v25.py [--sample 30] [--out runs/nlu3_v25]

测三部分：
  1. 45 个域各 5 句（同 tools/probe_nlu3.py 的 DOMAIN_CASES；标注有歧义的句子在 ACCEPT 里列出也算对的域）
  2. 评测数据抽样（快思考测试集·基础车控 + 一阶分发业务 labels 各类）
  3. 多轮（快思考测试集_多轮）：先单独解析上一轮得到 previous_nlu_1，再按文档拼装本轮，tts_1 写“好的”
只用 temperature=0、top_p=0.01（项目约定：相同输入必须得到相同输出，不解析置信度）。
输出 out/results.jsonl 和 out/report.md。

注：tools/probe_nlu3.py 测的是 /zp/update 对话系统接口（外面包了对话管理和拒识），不是这个模型本身。
"""
import argparse, asyncio, collections, json, os, random, statistics, sys, time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from probe_nlu3 import DOMAIN_CASES, multiturn, samples   # noqa: E402

SYSTEM = "你是一个车载语音助手指令解析引擎，专精于将自然语言指令转化为结构化语义"
# v25 训练集 nlu_train_v25_modi.jsonl 的 45 个 domain（接口说明第六节）
V25_DOMAINS = ["music", "intelligent_car_control", "phone_call", "navigation", "global_control", "fm", "unknown", "video_qa",
               "chat", "video", "other", "poi_recommend", "app_control", "app_call", "intelligent_car_info", "help", "weather_xf",
               "music_qa", "short_video", "reminder", "view_poi_recommend", "outside_qa", "home_agent", "travel_plan", "agent_task",
               "task_setting", "datetime_info", "stock", "view_poi_introduction", "news", "intelligent_car_scenario", "life_suggestion",
               "outgoing_eta", "changba", "wallpaper", "calculator", "train", "audio_book", "painting", "KTV", "similar_poi",
               "theme_gen", "flight", "traffic_restriction", "row_column_position"]

# 我们自造的同类说法里，按清单 v13 的一句话介绍看有两种合理归属的句子：列出的域也算对
ACCEPT = {
    "打开导航": {"navigation"}, "打开蓝牙": {"intelligent_car_control"}, "关闭WiFi": {"intelligent_car_control"},
    "打开收音机": {"app_control"}, "下一集": {"global_control"}, "切歌": {"music", "global_control"},
    "去公司要多久": {"navigation"}, "到机场还要多长时间": {"navigation"}, "现在去火车站堵不堵": {"navigation"},
    "这首歌叫什么名字": {"music"}, "夜曲是谁唱的": {"music"}, "下次保养是什么时候": {"help"},
    "你能做什么": {"chat"}, "你叫什么名字": {"chat"}, "我今天心情不好": {"unknown"},
    "再小一点": {"unknown"}, "那个呢": {"unknown"}, "就这个吧": {"unknown"}, "嗯嗯": {"unknown"},
    "打开唱吧": {"app_control"}, "关闭唱吧": {"app_control"}, "打开KTV": {"app_control"},
    "打开设置": {"intelligent_car_control"}, "换个壁纸": {"intelligent_car_control"},
    "附近有没有可以遛狗的公园": {"poi_recommend"}, "找个能停车的商场": {"navigation"},
}


def user_content(q: str, pq: str = "", pn: str = "", tts: str = "") -> str:
    return f"query:{q},previous_query_1:{pq},previous_nlu_1:{pn},tts_1:{tts}"


def parse_nlu(text: str | None) -> list[dict]:
    """文档 3.3 的解析：'d+i@k=v,k2=v2;d2+i2' -> [{domain, intent, slots}]"""
    blocks = []
    for b in (text or "").split(";"):
        b = b.strip()
        if not b:
            continue
        body, _, slot = b.partition("@")
        domain, _, intent = body.partition("+")
        slots = dict(item.split("=", 1) for item in slot.split(",") if "=" in item)
        blocks.append({"domain": domain.strip(), "intent": intent.strip(), "slots": slots})
    return blocks


async def ask(c, url, q, sem, pq="", pn="", tts=""):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_content(q, pq, pn, tts)}]
    async with sem:
        t = time.perf_counter()
        try:
            r = await c.post(url, json={"model": "qwen3_nlu", "messages": msgs, "temperature": 0, "top_p": 0.01,
                                        "max_tokens": 128, "stop": ["<|im_end|>"]}, timeout=30)
            ms = (time.perf_counter() - t) * 1000
            d = r.json()
            text = d["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return {"query": q, "error": f"{type(e).__name__}: {e}"[:200], "ms": (time.perf_counter() - t) * 1000}
    blocks = parse_nlu(text)
    return {"query": q, "ms": ms, "text": text, "blocks": blocks, "domains": [b["domain"] for b in blocks],
            "multi": len(blocks) > 1, "prompt_tokens": (d.get("usage") or {}).get("prompt_tokens")}


def dom_str(r):
    return "ERROR" if r.get("error") else ("+".join(r["domains"]) or "(空)")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("NLU3_V25_URL"), help="默认读环境变量 NLU3_V25_URL（地址不入库）")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default=str(REPO / "runs/nlu3_v25"))
    a = ap.parse_args()
    if not a.url:
        sys.exit("请设置 NLU3_V25_URL 或传 --url（服务地址向 NLU 团队获取，不写进仓库）")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient() as c:
        djobs = [(d, q) for d, qs in DOMAIN_CASES.items() if d in V25_DOMAINS for q in qs]   # 域清单以 v25 为准
        dres = await asyncio.gather(*[ask(c, a.url, q, sem) for _, q in djobs])
        sjobs = samples(a.sample, 0)
        sres = await asyncio.gather(*[ask(c, a.url, q, sem) for _, q in sjobs])
        mjobs = multiturn(a.sample * 2, 0)

        async def dialog(prev, cur):
            r1 = await ask(c, a.url, prev, sem)
            r2 = await ask(c, a.url, cur, sem, prev, r1.get("text", ""), "好的")
            return r1, r2
        mres = await asyncio.gather(*[dialog(p, q) for p, q, _, _ in mjobs])

    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for (d, q), r in zip(djobs, dres):
            f.write(json.dumps({"set": "domain", "expect": d, **r}, ensure_ascii=False) + "\n")
        for (src, q), r in zip(sjobs, sres):
            f.write(json.dumps({"set": "sample", "source": src, **r}, ensure_ascii=False) + "\n")
        for (p, q, pl, ql), (r1, r2) in zip(mjobs, mres):
            f.write(json.dumps({"set": "multiturn", "prev": p, "prev_label": pl, "label": ql, "turn1": r1, **r2}, ensure_ascii=False) + "\n")

    allr = dres + sres + [x for pair in mres for x in pair]
    ok = [r for r in allr if not r.get("error")]
    ms = sorted(r["ms"] for r in ok)
    L = ["# NLU3.0 v25 模型落域摸底\n",
         f"- 接口：`{a.url}`（model `qwen3_nlu`，v25，Qwen3-0.6B 全量微调），{time.strftime('%Y-%m-%d %H:%M')} 实测；请求按《NLU3.0大模型接口使用说明1030固定版本》：固定 system、扁平 user 串、temperature=0、top_p=0.01",
         f"- 请求 {len(allr)} 句，错误 {len(allr) - len(ok)} 句；单次时延 p50 {statistics.median(ms):.0f} ms，p90 {ms[int(.9 * len(ms)) - 1]:.0f} ms",
         "", "## 一、45 个域逐域测试（每域 5 句：清单例句 + 4 句同类说法）", ""]
    rows, summ = [], []
    for d in [x for x in DOMAIN_CASES if x in V25_DOMAINS]:
        rs = [r for (dd, _), r in zip(djobs, dres) if dd == d]
        strict = [d in r.get("domains", []) for r in rs]
        loose = [s or bool(set(r.get("domains", [])) & ACCEPT.get(r["query"], set())) for s, r in zip(strict, rs)]
        miss = [f"{r['query']} → {dom_str(r)}" + ("（可接受）" if l and not s else "") for r, s, l in zip(rs, strict, loose) if not s]
        summ.append((d, sum(strict), sum(loose)))
        rows.append(f"| `{d}` | {sum(strict)}/5 | {sum(loose)}/5 | {'；'.join(miss) or '—'} |")
    ts, tl = sum(s for _, s, _ in summ), sum(l for _, _, l in summ)
    full = [d for d, _, l in summ if l == 5]; weak = [f"{d}（{l}/5）" for d, _, l in summ if l <= 2]
    L += [f"- {len(djobs)} 句（v25 清单内的域）：严格命中 {ts} 句（{ts / len(djobs):.0%}）；算上歧义句的可接受归属 {tl} 句（{tl / len(djobs):.0%}）",
          f"- 可接受口径下 5 句全中 {len(full)} 个域；≤ 2/5 的域：{', '.join(weak) or '—'}", "",
          "| 域 | 严格 | 含可接受 | 未严格命中的句子 → 模型输出 |", "|---|---|---|---|", *rows,
          "", "## 二、评测数据抽样", "",
          "| 来源 | 句数 | unknown | 多意图 | 落域分布（前 6） |", "|---|---|---|---|---|"]
    by = collections.defaultdict(list)
    for (src, _), r in zip(sjobs, sres):
        by[src].append(r)
    for src, rs in by.items():
        okr = [r for r in rs if not r.get("error")]; n = max(1, len(okr))
        dist = collections.Counter(dom_str(r) for r in okr)
        L.append(f"| {src} | {len(okr)} | {sum(1 for r in okr if r['domains'] == ['unknown']) / n:.0%} | {sum(1 for r in okr if r['multi'])} | "
                 + "，".join(f"{k} {v}" for k, v in dist.most_common(6)) + " |")
    L += ["", "## 三、多轮（快思考测试集_多轮；上一轮先单独解析，结果作 previous_nlu_1，tts_1=好的）", "",
          "| 上一轮标签 → 本轮标签 | 组数 | 本轮落域分布（前 5） |", "|---|---|---|"]
    mb = collections.defaultdict(list)
    for (p, q, pl, ql), (_, r2) in zip(mjobs, mres):
        mb[f"{pl} → {ql}"].append(r2)
    for k, rs in sorted(mb.items(), key=lambda kv: -len(kv[1])):
        okr = [r for r in rs if not r.get("error")]
        L.append(f"| {k} | {len(okr)} | " + "，".join(f"{x} {y}" for x, y in collections.Counter(dom_str(r) for r in okr).most_common(5)) + " |")
    L += ["", "多轮样例：", *[f"- {p}（{r1.get('text')}）→ **{q}** ⇒ `{r2.get('text')}`" for (p, q, _, _), (r1, r2) in list(zip(mjobs, mres))[:12]]]
    names = collections.Counter(b for r in ok for b in r.get("domains", []))
    known = set(V25_DOMAINS)
    odd = {k: v for k, v in names.items() if k not in known}
    L += ["", "## 四、域清单口径", "",
          f"- 清单 v13（docs/NLU3.0大模型_domain清单.md）有、v25 训练集没有：{sorted(set(DOMAIN_CASES) - known)}——这些域模型不会输出",
          f"- v25 训练集有、清单 v13 没有：{sorted(known - set(DOMAIN_CASES))}",
          f"- 输出里出现过、但不在 v25 训练集 45 个域里的：{odd or '无'}"]
    errs = [r for r in allr if r.get("error")]
    if errs:
        L.append(f"- 错误 {len(errs)} 句，例：" + "；".join(f"{r['query']} → {r['error']}" for r in errs[:5]))
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:9]))
    print(f"\n报告：{out / 'report.md'}")


if __name__ == "__main__":
    asyncio.run(main())
