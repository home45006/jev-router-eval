"""按 DeepSeek 官方价格计算每次大模型调用的费用（人民币元）。

价格来自 https://api-docs.deepseek.com/zh-cn/quick_start/pricing（2026-10-12 抓取），单位：元 / 百万 tokens。
    高峰时段：北京时间周一至周五 9:00-12:00、14:00-18:00；其余时段（含周末）为空闲时段，价格为高峰的一半。
    中国法定节假日全天为空闲时段，这里无法判断节假日，节假日会按工作日算（偏贵）。
    扣减费用 = token 消耗量 × 模型单价；输入分缓存命中 / 未命中两档。

DEEPSEEK_PRICE_PERIOD=peak|offpeak 可以固定按某一档计算；不设时按调用时的北京时间判断。
价格有变动时改下面的表。
"""
from __future__ import annotations

import datetime
import os

# 模型 → 时段 → (缓存命中输入, 缓存未命中输入, 输出)，元 / 百万 tokens
PRICES = {
    "deepseek-flash": {"peak": (0.04, 2.0, 8.0), "offpeak": (0.02, 1.0, 4.0)},
    "deepseek-v4-pro": {"peak": (0.30, 9.0, 27.0), "offpeak": (0.15, 4.5, 13.5)},
}
PRICE_SOURCE = "DeepSeek 官方价格页（2026-10-12）"
BEIJING = datetime.timezone(datetime.timedelta(hours=8))


def price_key(model: str | None) -> str:
    """deepseek-v4-pro 按 Pro 计价；其余（deepseek-flash、旧名 deepseek-v4-flash 等）按 Flash 计价。"""
    return "deepseek-v4-pro" if model and "pro" in model else "deepseek-flash"


def period(now: datetime.datetime | None = None) -> str:
    forced = os.environ.get("DEEPSEEK_PRICE_PERIOD")
    if forced in ("peak", "offpeak"):
        return forced
    t = (now or datetime.datetime.now(datetime.timezone.utc)).astimezone(BEIJING)
    hm = t.hour * 60 + t.minute
    busy = t.weekday() < 5 and (9 * 60 <= hm < 12 * 60 or 14 * 60 <= hm < 18 * 60)
    return "peak" if busy else "offpeak"


def cost(model: str | None, cache_hit: int, cache_miss: int, output: int, when: str | None = None) -> dict:
    """返回费用明细：元，以及用到的单价，前端直接展示。"""
    when = when or period()
    key = price_key(model)
    p_hit, p_miss, p_out = PRICES[key][when]
    parts = {"cache_hit": cache_hit * p_hit / 1e6, "cache_miss": cache_miss * p_miss / 1e6, "output": output * p_out / 1e6}
    return {"yuan": round(sum(parts.values()), 6), "parts": {k: round(v, 6) for k, v in parts.items()},
            "tokens": {"cache_hit": cache_hit, "cache_miss": cache_miss, "output": output},
            "period": when, "price_model": key, "unit_prices": {"cache_hit": p_hit, "cache_miss": p_miss, "output": p_out}}


def cost_openai(model: str | None, usage) -> dict | None:
    """OpenAI 格式：prompt_tokens = 命中 + 未命中，DeepSeek 另给 prompt_cache_hit_tokens / prompt_cache_miss_tokens。"""
    if usage is None:
        return None
    get = (lambda k: usage.get(k)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, None))
    prompt = get("prompt_tokens") or 0
    hit = get("prompt_cache_hit_tokens")
    miss = get("prompt_cache_miss_tokens")
    if hit is None and miss is None:
        hit, miss = 0, prompt                 # 没有缓存信息时全部按未命中算（偏贵）
    hit, miss = hit or 0, miss if miss is not None else max(prompt - (hit or 0), 0)
    return cost(model, hit, miss, get("completion_tokens") or 0)


def cost_anthropic(model: str | None, usage: dict | None) -> dict | None:
    """Anthropic 格式：input_tokens 不含缓存读取；cache_read_input_tokens 是命中，cache_creation_input_tokens 按未命中算。"""
    if not usage:
        return None
    hit = usage.get("cache_read_input_tokens") or 0
    miss = (usage.get("input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)
    return cost(model, hit, miss, usage.get("output_tokens") or 0)


# ---------------------------------------------------------------- Jev（TypeSafe）
# 来源：https://docs.typesafe.ai/models.md（2026-10-29 读取）：只按输入 token 计费，输出 token 免费；
# jev-1.14.0、jev-1.13.0 均为 $42 / 十亿 tokens，即 $0.042 / 百万 tokens。价格变动时改这里。
JEV_PRICES_USD_PER_M = {"jev-1.14.0": 0.042, "jev-1.13.0": 0.042}
JEV_PRICE_SOURCE = "TypeSafe Models 页（2026-10-29）：只计输入 token，$0.042 / 百万 tokens，输出免费"
USD_CNY = float(os.environ.get("USD_CNY", "7.1"))   # 换算汇率，可用环境变量覆盖；仅用于展示


def jev_cost(model: str | None, input_tokens: int | None, output_tokens: int | None = None) -> dict | None:
    if input_tokens is None:
        return None
    price = JEV_PRICES_USD_PER_M.get(model or "", JEV_PRICES_USD_PER_M["jev-1.14.0"])
    usd = input_tokens * price / 1e6
    return {"usd": round(usd, 8), "yuan": round(usd * USD_CNY, 8), "fx": USD_CNY,
            "tokens": {"input": input_tokens, "output": output_tokens or 0},
            "unit_price_usd_per_m": price, "price_model": model}
