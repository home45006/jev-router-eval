#!/usr/bin/env python3
"""从测试集 xlsx 生成 Jev 标签评测集。

用法:
    python build_dataset.py <测试集.xlsx> [--out .]

输出:
    eval_set.jsonl   每行一条评测样本（已去重，已按模板分组切分 dev/val/test）
    待确认改标.csv    标注与规则冲突、需要人工确认的行

标签（2026-10-10 确认）:
    语义 sheet        输出“预期domain”列。domain 一律用首字母小写的写法（carControl、naviControl、phone），
                      原表里大小写不一致的写法按此合并。“类别”列只作为附加信息保留，不是标签。
    大模型agent sheet  输出 label 列。

规则:
    1. “全局”类别表示任何场景下都可以说，是可说范围；现在类别不再是标签，只保留 is_global 标记。
    2. 日程提醒（2026-10-11 改）：语义 sheet 里 domain 为 scheduleReminder 的行全部改标，
       周期性的（每天、每周……）归任务大师，其余归条件任务；不再有 scheduleReminder 标签，提醒不交给快思考。
       一次性任务归条件任务，周期性任务归任务大师，这两类由 Agent 大脑调度。
    3. 闲聊会话里的追问一律跟随上一轮：评测脚本用 previous_turn 实现，这里只负责带上上一轮。
    4. 两地距离（2026-10-11）：语义 sheet 里 naviControl 中说出了出发地的“甲到乙还有多远、多久、多少公里、堵不堵”
       改标为复杂导航agent，交给 Agent 大脑；只问到目的地、公司、途经点的（到公司还有多远）仍是 naviControl。
    规则 2、4 在切分之后执行，句子留在原来的 dev/val/test 里，和改标前的结果可比。
"""
import argparse, collections, csv, hashlib, json, re, sys
from pathlib import Path

import openpyxl

SEM_SHEET, AGENT_SHEET = "语义", "大模型agent"

# 标签 -> 分支
ROUTE_OF_AGENT_LABEL = {
    "闲聊": "闲聊",
    "中枢简单任务": "拆分后逐条NLU直达",
    "中枢复杂多任务": "慢思考编排",
}
def route_of(label, sem_labels):
    if label in sem_labels:
        return "NLU直达"
    return ROUTE_OF_AGENT_LABEL.get(label, "Agent直达")

# ---------- 规则 2：条件任务里的"提醒" ----------
REMIND = re.compile(r"提醒|提个醒|叫我|喊我|喊一声|喊一下|给我个提示|通知我")
CAR_ACTION = re.compile(
    r"导航|导到|导去|导个航|导一下|规划.{0,6}路线|播放|播一下|播报|放点|放首|放一|放我|放些|放下|"
    r"放.{0,6}(歌|音乐|乐|曲|电台|广播|新闻)|播.{0,6}(歌|曲|新闻|音乐|广播)|切首|"
    r"打开|关闭|关掉|关了|关上|关小|开启|启动|开一下|开下|开个|再开|开空调|开座椅|调到|调高|调低|调成|调一下|"
    r"切到|切换|切蓝牙|换成|连接|静音|暂停|停掉|来点|换一首|降下来|升起来|加热.{0,4}开|通风.{0,4}开|保存位置"
)
# ---------- 一次性任务归条件任务，周期性任务归任务大师 ----------
PERIODIC = re.compile(r"每天|每周|每次|每当|每逢|每晚|每早|天天|每月|工作日|周期")
ONE_OFF = re.compile(r"等会|待会|一会儿|明天|后天|今晚|\d+分钟后|\d+小时后|半小时后")

# ---------- 规则 4：两地距离 ----------
TWO_PLACE = re.compile(r"^(?!那)[^到去，,]{1,12}(到|去)[^到去，,]{1,15}(还有)?(多远|多久|多长时间|多少公里|堵不堵)[？?]?$")

def reminder_bucket(q):
    r, a = bool(REMIND.search(q)), bool(CAR_ACTION.search(q))
    if r and not a:
        return "只有提醒，没有车机动作"
    if not r and not a:
        return "没有明确的车机动作"
    if r and a:
        return "提醒和车机动作都出现"
    return None

# ---------- 模板归一（用于切分时把同模板的句子放在同一侧） ----------
POS = r"主驾|副驾|左前|右前|左后|右后|前排|后排|二排左|二排右|二排|三排中间|三排|最后排|左侧|右侧|全舱|全车|全部|所有|中控"
APPS = r"qq音乐|酷狗音乐|网易云音乐|汽水音乐|酷我音乐|咪咕音乐|爱奇艺|腾讯视频|哔哩哔哩|喜马拉雅"
def template_key(q):
    s = q.lower()
    s = re.sub(r"[\s，。？！、,.?!：:；;“”\"'《》]", "", s)
    s = re.sub(r"\d+(\.\d+)?", "0", s)
    s = re.sub(POS, "P", s)
    s = re.sub(APPS, "A", s)
    s = re.sub(r"打开|关闭|开启|关掉|退出|进入", "V", s)
    s = re.sub(r"调高|调低|调大|调小|增大|减小|增加|降低", "D", s)
    return s

def h(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def parse_now_playing(cells):
    for c in cells:
        if not c:
            continue
        try:
            for ev in json.loads(c):
                mps = ev.get("payload", {}).get("vehicleSignals", {}).get("multimedia_play_status", {})
                for media_type, info in mps.items():
                    return {"media_type": media_type, "app_name": info.get("app_name")}
        except (ValueError, AttributeError, TypeError):
            return {"raw": str(c)[:200]}
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--out", default=".")
    ap.add_argument("--dev", type=float, default=0.2)
    ap.add_argument("--val", type=float, default=0.2)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.load_workbook(a.xlsx, read_only=True, data_only=True)
    sem_raw = [(str(r[0]).strip(), str(r[1]).strip(), str(r[2]).strip())
               for r in wb[SEM_SHEET].iter_rows(min_row=2, values_only=True) if r[1] and r[2]]
    # domain 大小写归一：统一成首字母小写的写法（carControl、naviControl、phone）
    spell = collections.defaultdict(collections.Counter)
    for _, _, dom in sem_raw:
        spell[dom.lower()][dom] += 1
    canon = {}
    for k, v in spell.items():
        name = max(v, key=v.get)
        canon[k] = name[0].lower() + name[1:]
    sem = [(canon[dom.lower()], q, cat) for cat, q, dom in sem_raw]
    agent = [r for r in wb[AGENT_SHEET].iter_rows(min_row=2, values_only=True) if r[2]]
    sem_labels = sorted({c for c, _, _ in sem})

    # 单轮语义
    items = collections.OrderedDict()   # key -> item
    def add(key, **kw):
        it = items.get(key)
        if it is None:
            it = items[key] = dict(gold=[], sources=[], review=[], categories=[], **kw)
        return it

    for c, q, cat in sem:
        it = add(("", q, ""), query=q, prev_query=None, prev_label=None, now_playing=None, turn=1)
        if c not in it["gold"]: it["gold"].append(c)
        if cat not in it["categories"]: it["categories"].append(cat)
        if SEM_SHEET not in it["sources"]: it["sources"].append(SEM_SHEET)

    sess = collections.defaultdict(list)
    for r in agent:
        sess[str(r[0])].append(r)
    for sid, rows in sess.items():
        rows.sort(key=lambda r: int(str(r[1])) if str(r[1]).isdigit() else 0)
        prev = None
        for r in rows:
            q, label, turn = str(r[2]).strip(), str(r[3]).strip(), int(str(r[1])) if str(r[1]).isdigit() else 1
            np_ = parse_now_playing(r[4:])
            if turn > 1:
                pq = prev[0] if prev else None            # 首轮缺失时为 None
                pl = prev[1] if prev else label           # 数据里多轮会话整段同标签
                key = (pq or "?" + sid, q, "")
                it = add(key, query=q, prev_query=pq, prev_label=pl, now_playing=np_, turn=turn)
                if pq is None: it["review"].append("会话缺少上一轮")
            else:
                key = ("", q, json.dumps(np_, sort_keys=True) if np_ else "")
                it = add(key, query=q, prev_query=None, prev_label=None, now_playing=np_, turn=1)
            if label not in it["gold"]: it["gold"].append(label)
            if AGENT_SHEET not in it["sources"]: it["sources"].append(AGENT_SHEET)
            prev = (q, label)

    review_rows = []
    for it in items.values():
        g = it["gold"]
        # 规则 2
        if "条件任务" in g and "scheduleReminder" not in g:
            b = reminder_bucket(it["query"])
            if b and b != "只有提醒，没有车机动作":
                it["review"].append("规则2：" + b)
                review_rows.append([AGENT_SHEET, it["query"], "条件任务", "（请判断）", b])
        if "任务大师" in g and ONE_OFF.search(it["query"]) and not PERIODIC.search(it["query"]):
            it["review"].append("规则：任务大师里疑似一次性任务")
            review_rows.append([AGENT_SHEET, it["query"], "任务大师", "（请判断）", "周期性任务归任务大师，这句疑似一次性任务"])
        if len(it["sources"]) == 2:
            review_rows.append(["两个sheet", it["query"], " / ".join(g), "（请判断）", "同一句话在两个 sheet 都出现"])
        elif len(g) > 1 and not any(x.startswith("规则2") for x in it["review"]):
            review_rows.append([it["sources"][0], it["query"], " / ".join(g), "（依赖上下文）", "同一句话有多个标签"])
        it["is_global"] = "全局" in it["categories"]
        it["primary"] = g[0]
        it["routes"] = sorted({route_of(x, sem_labels) for x in g})

    # 切分：同一标签内按模板分组，组不拆开
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, it in items.items():
        tk = template_key(it["prev_query"] or it["query"])      # 多轮按首轮分组，保证同会话同侧
        groups[it["primary"]][tk].append(it)
    for label, g in groups.items():
        keys = sorted(g, key=h)
        n = len(keys)
        n_dev = max(1, round(n * a.dev)) if n >= 3 else 0
        n_val = max(1, round(n * a.val)) if n >= 3 else 0
        for i, k in enumerate(keys):
            split = "dev" if i < n_dev else "val" if i < n_dev + n_val else "test"
            for it in g[k]:
                it["split"] = split

    # 规则 2、4：切分之后改标
    relabeled = collections.Counter()
    for it in items.values():
        g = it["gold"]
        if "scheduleReminder" in g:
            new = "任务大师" if PERIODIC.search(it["query"]) else "条件任务"
            g[:] = list(dict.fromkeys(new if x == "scheduleReminder" else x for x in g))
            it["review"].append(f"规则2：日程提醒改标为{new}")
            relabeled[f"scheduleReminder → {new}"] += 1
        if g == ["naviControl"] and TWO_PLACE.search(it["query"]):
            g[:] = ["复杂导航agent"]
            it["review"].append("规则4：两地距离改标为复杂导航agent")
            relabeled["naviControl 两地距离 → 复杂导航agent"] += 1
        it["primary"] = g[0]
        it["routes"] = sorted({route_of(x, sem_labels) for x in g})

    with open(out / "eval_set.jsonl", "w", encoding="utf-8") as f:
        for i, it in enumerate(items.values()):
            row = {"id": f"q{i:05d}", **{k: it[k] for k in (
                "query", "gold", "primary", "routes", "split", "turn", "prev_query", "prev_label",
                "now_playing", "is_global", "categories", "sources", "review")}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out / "待确认改标.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["来源", "query", "现标签", "按规则建议", "原因", "确认后的标签（请填）"])
        for r in sorted(review_rows, key=lambda r: (r[4], r[1])):
            w.writerow(r + [""])

    # 摘要
    merged = {k: dict(v) for k, v in spell.items() if len(v) > 1}
    print("domain 写法合并：", {canon[k]: v for k, v in merged.items()})
    print("domain 列表：", sorted(set(canon.values())))
    print("改标：", dict(relabeled))
    n = len(items)
    sp = collections.Counter(it["split"] for it in items.values())
    print(f"样本 {n} 条（去重后）  dev/val/test = {sp['dev']}/{sp['val']}/{sp['test']}")
    print("多轮样本", sum(it["turn"] > 1 for it in items.values()),
          "| 带播放状态", sum(bool(it["now_playing"]) for it in items.values()),
          "| 全局", sum(it["is_global"] for it in items.values()),
          "| 多个可接受标签", sum(len(it["gold"]) > 1 for it in items.values()))
    rc = collections.Counter(r[4] for r in review_rows)
    print("待确认：", dict(rc))
    by = collections.defaultdict(collections.Counter)
    for it in items.values(): by[it["primary"]][it["split"]] += 1
    for label in sorted(by, key=lambda l: -sum(by[l].values())):
        c = by[label]; print(f"  {label:<18} dev {c['dev']:>4}  val {c['val']:>4}  test {c['test']:>4}")

if __name__ == "__main__":
    sys.exit(main())
