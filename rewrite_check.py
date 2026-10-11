#!/usr/bin/env python3
"""用 DeepSeek V4 Flash 非思考模式把模糊指代改写成标准名称，检查 fixtures/fuzzy_refs.jsonl 里的样本。

    export DEEPSEEK_API_KEY=...
    export DEEPSEEK_MODEL=...          # 不设时依次试 deepseek-v4-flash、deepseek-flash，用第一个能调通的
    export DEEPSEEK_BASE_URL=...       # 默认 https://api.deepseek.com
    python rewrite_check.py --out runs/rewrite

样本是人工构造的 20 条，结果只说明改写能不能用，不代表线上效果。
请求都带 thinking.type=disabled（非思考模式）；报告里列出推理 token，非思考模式下应为 0。
"""
import argparse, asyncio, json, os, time
from pathlib import Path

from openai import AsyncOpenAI

CANDIDATES = ["deepseek-v4-flash", "deepseek-flash"]
NO_THINKING = {"thinking": {"type": "disabled"}}
PROMPT = (
    "你是车载语音助手的改写模块。用户的话里如果有俗称、外号、简称、昵称指代某个地点、人物或作品，"
    "把它换成唯一的标准名称；其余字词一个都不要改。没有需要还原的，原样输出。只输出改写后的一句话。"
)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fixtures/fuzzy_refs.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                         base_url=os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com")
    models = sorted(m.id for m in (await client.models.list()).data)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]

    async def call(model, query):
        return await client.chat.completions.create(
            model=model, temperature=0, extra_body=NO_THINKING,
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": query}])

    probes, model = [], None
    for cand in ([os.environ["DEEPSEEK_MODEL"]] if os.environ.get("DEEPSEEK_MODEL") else CANDIDATES):
        try:
            res = await call(cand, rows[0]["query"])
            probes.append(f"{cand}：可用（返回的模型名 {res.model}）")
            model = model or cand
        except Exception as e:
            probes.append(f"{cand}：不可用（{type(e).__name__}: {str(e)[:120]}）")
    if not model:
        raise SystemExit("没有可用的模型：\n" + "\n".join(probes))

    async def one(r):
        t = time.perf_counter()
        res = await call(model, r["query"])
        return r, res.choices[0].message.content.strip(), time.perf_counter() - t, res.usage

    results = await asyncio.gather(*(one(r) for r in rows))
    L = ["# DeepSeek 改写检查（非思考模式）\n", f"- /models 列出的模型：{', '.join(models)}",
         *[f"- 探测 {p}" for p in probes], f"- 本次使用：{model}",
         "- 样本：人工构造 20 条（12 条有模糊指代，8 条没有）\n",
         "| 原句 | 有模糊指代 | 改写结果 | 是否改动 | 耗时 秒 |", "|---|---|---|---|---|"]
    lat = []
    for r, out, sec, _ in results:
        lat.append(sec)
        L.append(f"| {r['query']} | {'是' if r['needs_rewrite'] else '否'} | {out} | {'改了' if out != r['query'] else '未改'} | {sec:.2f} |")
    lat.sort()
    neg_changed = sum(out != r["query"] for r, out, _, _ in results if not r["needs_rewrite"])
    pos_changed = sum(out != r["query"] for r, out, _, _ in results if r["needs_rewrite"])
    L.append(f"\n- 有模糊指代的 12 条里改动了 {pos_changed} 条（改得对不对需要人工看上表）")
    L.append(f"- 没有模糊指代的 8 条里被改动 {neg_changed} 条（应为 0）")
    def reasoning(u):
        d = getattr(u, "completion_tokens_details", None)
        return (getattr(d, "reasoning_tokens", None) or 0) if d else 0
    out_tok = [u.completion_tokens for *_, u in results]
    L.append(f"- 推理 token 合计 {sum(reasoning(u) for *_, u in results)}（非思考模式应为 0）；"
             f"输出 token 平均 {sum(out_tok) / len(out_tok):.1f}")
    L.append(f"- 单次耗时：中位数 {lat[len(lat) // 2]:.2f} 秒，最慢 {lat[-1]:.2f} 秒（GitHub 美国机房测得，车端另测）")
    text = "\n".join(L) + "\n"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
