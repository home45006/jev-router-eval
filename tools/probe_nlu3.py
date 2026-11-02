"""NLU3.0 对话系统接口（/zp/update）落域摸底。注意：这是包了对话管理的接口，模型本身请用 tools/probe_nlu3_v25.py。

原说明：NLU3.0 测试环境落域摸底：45 个域各 5 句（清单例句 + 4 句同类说法），评测数据抽样，以及多轮。

    NLU3_DIALOG_URL=https://<host>/zp/update python tools/probe_nlu3.py [--sample 30] [--out runs/nlu3_probe]

请求格式按 docs/nlu3/Dialogue_4b接口参数说明.txt：debug=true、enable_hotfix=false、input_data=[]；多轮用同一 session_id
顺序发送（首轮 first_wakeup=true，之后 false）。落域看两层：
  - 模型分类：debug.nlu_result（"domain\tintent\tk=v,…"；多意图以 "multi:" 开头、";" 分隔）——NLU3.0 的落域判断；
  - 最终输出：Payload.Semantic.Results[].Nlu.Domain——经过对话管理和拒识之后的结果，测试环境里很多技能没接，会变成 other。
输出 out/results.jsonl（逐句原始结果）和 out/report.md。
"""
import argparse, asyncio, collections, json, os, random, statistics, time, uuid
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]

# 每个域 5 句：第 1 句是清单 v13 的例句，后 4 句是同类说法
DOMAIN_CASES = {
    "music": ["我想听一个南屏晚钟", "播放周杰伦的夜曲", "来一首陈奕迅的十年", "播放我收藏的歌单", "放点八十年代的粤语歌"],
    "intelligent_car_control": ["关闭前挡风玻璃加热", "打开空调", "主驾座椅通风调到二档", "车窗打开一半", "把氛围灯调成蓝色"],
    "phone_call": ["查看二卡的电话记录", "给张三打电话", "拨打10086", "打电话给妈妈", "接听电话"],
    "navigation": ["导航到淮安市小康城", "导航到天安门", "回家", "走高速优先", "取消导航"],
    "global_control": ["小安你好请帮我快进到四十分钟44秒钟", "返回主页", "下一页", "快退十秒", "返回上一级"],
    "fm": ["播放调频宿迁交通广播", "打开收音机", "收听FM97.4", "播放中国之声", "换一个电台"],
    "chat": ["给甜甜讲首故事", "陪我聊会儿天吧", "讲个笑话", "你叫什么名字", "我今天心情不好"],
    "video_qa": ["帮我找王祖蓝参加的剧集", "成龙演过哪些电影", "推荐几部好看的悬疑剧", "甄嬛传是谁导演的", "有什么适合全家看的电影"],
    "app_call": ["查看我订的航班信息", "打开美团看看我的订单", "用高德查一下附近的加油站", "打开微信小程序", "查看我的快递"],
    "intelligent_car_info": ["查看100公里耗油", "还剩多少油", "现在胎压是多少", "续航还有多少公里", "下次保养是什么时候"],
    "video": ["调整播放速度为2倍", "播放庆余年", "下一集", "切换到高清", "用爱奇艺播放狂飙"],
    "help": ["查询经济模式的好处", "你能做什么", "座椅通风怎么打开", "这个按钮是干嘛的", "怎么连接蓝牙"],
    "poi_recommend": ["帮我推荐个快捷酒店", "附近有什么好吃的", "推荐一家评分高的火锅店", "附近人均五十以内的面馆", "找个能停车的商场"],
    "weather_xf": ["查询明天苏州的天气", "今天天气怎么样", "明天会下雨吗", "北京现在多少度", "这周末天气如何"],
    "music_qa": ["给我推荐一些民谣呗", "夜曲是谁唱的", "来点适合下雨天听的歌", "周杰伦有哪些好听的歌", "这首歌叫什么名字"],
    "travel_plan": ["帮我规划北京四天的行程", "安排一个杭州三日游", "去成都玩两天怎么安排", "自驾川西七天的路线", "帮我规划一下周末去哪玩"],
    "home_agent": ["暂停关闭主卧的窗帘", "打开客厅的灯", "家里空调开了吗", "把家里的扫地机器人打开", "关闭卧室的灯"],
    "stock": ["查询北汽蓝谷的股价", "今天大盘怎么样", "茅台股票多少钱", "帮我查一下a股", "宁德时代涨了吗"],
    "reminder": ["设置闹钟为下午两点", "明天早上七点叫我起床", "提醒我下午三点开会", "取消所有闹钟", "我明天有什么日程"],
    "short_video": ["找一些关于聚会的短片", "播放搞笑短视频", "来点抖音热门视频", "刷一下短视频", "看看猫咪的短视频"],
    "task_setting": ["上车后自动开启空调", "每天下班上车就把空调开到22度", "当车速超过一百时提醒我", "下雨时自动关闭车窗", "到家后自动关闭大灯"],
    "wechat_on_car": ["我要打视频给小阿萨", "用微信给李四发消息", "读一下微信消息", "微信发位置给张三", "回复他我马上到"],
    "agent_task": ["能帮我点两杯瑞幸凉少糖美式吗", "帮我订一间今晚的如家酒店", "帮我点一份麦当劳外卖", "帮我买两张电影票", "帮我叫个代驾"],
    "datetime_info": ["帮我查一下今天星期几", "今天几号", "现在几点了", "今天农历是多少", "离春节还有几天"],
    "view_poi_recommend": ["我想找个地方放风筝", "附近有适合拍照的地方吗", "找个能看日落的地方", "附近有没有可以遛狗的公园", "哪里适合野餐"],
    "view_poi_introduction": ["告诉我纳帕海几月份最美我要去旅行", "介绍一下西湖", "故宫有什么好玩的", "黄山什么时候去最好", "兵马俑的历史"],
    "news": ["播放5月6日的新闻", "播放今天的新闻", "最近有什么科技新闻", "来点体育新闻", "今天有什么热点"],
    "life_suggestion": ["查询最近的洗车指数", "今天适合洗车吗", "明天适合跑步吗", "今天紫外线强吗", "穿衣指数"],
    "intelligent_car_scenario": ["太暗啦", "好热啊", "车里有点闷", "我有点冷", "雾太大看不清"],
    "outgoing_eta": ["我要去夫子庙坐公交划算还是开车更划算呢", "去公司要多久", "到机场还要多长时间", "现在去火车站堵不堵", "几点出发能赶上九点的会"],
    "outside_qa": ["帮我看下那个交通标志", "前面那个牌子写的什么", "旁边那栋楼是什么", "前面是什么路口", "路边那个是什么店"],
    "changba": ["来一首晴天我唱一下", "打开唱吧", "我想唱歌", "唱吧点一首稻香", "关闭唱吧"],
    "app_control": ["开启语音助手和车载WiFi", "打开蓝牙", "打开导航", "关闭WiFi", "打开设置"],
    "calculator": ["3×200等于多少", "一百加二百五", "五十的平方是多少", "一英里等于多少公里", "三百除以七"],
    "app_agent": ["微信扫一扫给我打开", "打开支付宝付款码", "刷一下微博热搜", "打开淘宝搜一下充电宝", "美团上找一家奶茶店"],
    "wallpaper": ["我要画一幅科比桌面", "换个壁纸", "设置一张星空壁纸", "换一张风景桌面", "壁纸换成猫咪"],
    "other": ["呀呀仪表再调亮一些", "再小一点", "那个呢", "就这个吧", "嗯嗯"],
    "flight": ["帮我查下去三亚的机票", "明天北京到上海的航班", "查一下CA1234航班", "帮我订一张去成都的机票", "最便宜的去广州的机票"],
    "painting": ["我想画一个宝马七系", "帮我画一只猫", "画一幅山水画", "生成一张赛博朋克风的图片", "画一个可爱的机器人"],
    "audio_book": ["用有声书播放鬼眼神探", "播放三体有声书", "我想听评书", "播放郭德纲的相声", "继续播放上次的有声书"],
    "KTV": ["来一首陈奕迅的浮夸", "打开KTV", "我要唱K", "切歌", "原唱伴唱切换"],
    "train": ["查询去昆明的火车票", "明天去上海的高铁", "G1234次列车几点发车", "帮我买一张去南京的火车票", "北京到西安的动车"],
    "theme_gen": ["给我弄成黑夜的主题", "换个主题", "切换成科技风主题", "主题换成粉色", "生成一个中国风主题"],
    "traffic_restriction": ["打开限行区域", "今天限行尾号是多少", "明天北京限行吗", "外地车能进五环吗", "我的车今天限行吗"],
    "similar_poi": ["帮我找个跟长城一样的景点", "有没有跟海底捞差不多的火锅店", "找个跟星巴克类似的咖啡馆", "类似迪士尼的乐园", "跟西湖差不多的地方"],
}


def body(q: str, sid: str | None = None, turn: int = 1) -> dict:
    sid = sid or f"probe-{uuid.uuid4().hex[:12]}"
    return {"query": q, "session_id": sid, "request_id": f"{sid}-turn-{turn}", "input_data": [],
            "debug": True, "first_wakeup": turn == 1, "enable_hotfix": False}


def parse_nlu_result(s: str | None) -> list[dict]:
    """debug.nlu_result -> [{domain, intent, slots}]；None / unknown -> []"""
    if not s or s == "unknown":
        return []
    s = s[6:] if s.startswith("multi:") else s
    out = []
    for part in s.split(";"):
        f = part.split("\t")
        kvs = f[2].split(",") if len(f) > 2 and f[2] else []
        out.append({"domain": f[0], "intent": f[1] if len(f) > 1 else "",
                    "slots": dict(kv.split("=", 1) for kv in kvs if "=" in kv)})
    return out


async def ask(c: httpx.AsyncClient, url: str, q: str, sem: asyncio.Semaphore, sid: str | None = None, turn: int = 1) -> dict:
    async with sem:
        t = time.perf_counter()
        try:
            r = await c.post(url, json=body(q, sid, turn), timeout=30)
            ms = (time.perf_counter() - t) * 1000
            d = r.json()
        except Exception as e:
            return {"query": q, "error": f"{type(e).__name__}: {e}"[:200], "ms": (time.perf_counter() - t) * 1000}
    hdr = d.get("Header") or {}
    sem_ = (d.get("Payload") or {}).get("Semantic") or {}
    if hdr.get("Code") not in (0, None) or not sem_:
        return {"query": q, "ms": ms, "error": f"Code={hdr.get('Code')} {hdr.get('Message')}"[:200]}
    res = [{"sub": x.get("SubQuery"), "domain": (x.get("Nlu") or {}).get("Domain"), "intent": (x.get("Nlu") or {}).get("Intent"),
            "conf": (x.get("Nlu") or {}).get("Confidence"), "rejected": x.get("IsRejected"),
            "slots": [(s.get("Name"), s.get("Value")) for s in (x.get("Nlu") or {}).get("Slots") or []]}
           for x in sem_.get("Results") or []]
    dbg = d.get("debug") or {}
    nr = dbg.get("nlu_result")
    ri = dbg.get("RejectionInfo") or {}
    return {"query": q, "ms": ms, "version": hdr.get("Version"), "results": res,
            "nlu_result": nr, "model": parse_nlu_result(nr), "multi": bool(nr and nr.startswith("multi:")),
            "dbg_rejected": dbg.get("is_rejected"),
            "segments": [(x.get("generate_text"), x.get("category")) for x in dbg.get("SegementInfo") or []],
            "rejection": {k: ri.get(k) for k in ("structure", "contextual", "completely", "meaningful")}}


def samples(n: int, seed: int) -> list[tuple[str, str]]:
    import openpyxl
    rnd = random.Random(seed)
    out = []
    fast = REPO / "data/快思考测试集/一阶分发测试集.xlsx"
    if fast.exists():
        rows = sorted({str(r[0]).strip() for r in list(openpyxl.load_workbook(fast, read_only=True).active.iter_rows(values_only=True))[1:]
                       if r and r[0] and r[2] == "基础车控"})
        out += [("快思考测试集·基础车控", q) for q in rnd.sample(rows, min(n * 3, len(rows)))]
    for f in sorted((REPO / "data/一阶分发业务labels").glob("[0-9]*.xlsx")):
        if f.stem.startswith("00_"):        # 标签索引，不是 query
            continue
        rows = sorted({str(r[0]).strip() for r in list(openpyxl.load_workbook(f, read_only=True).active.iter_rows(values_only=True))[1:]
                       if r and r[0] and (len(r) < 7 or r[6] != "是")})
        out += [(f.stem, q) for q in rnd.sample(rows, min(n, len(rows)))]
    return out


def multiturn(n: int, seed: int) -> list[tuple[str, str, str, str]]:
    """快思考测试集_多轮：(上一轮, 本轮, 上一轮标签, 本轮标签)"""
    import openpyxl
    f = REPO / "data/快思考测试集/一阶分发测试集_多轮.xlsx"
    if not f.exists():
        return []
    rows = [r for r in list(openpyxl.load_workbook(f, read_only=True).active.iter_rows(values_only=True))[1:] if r and r[0] and r[1]]
    rnd = random.Random(seed)
    return [(json.loads(r[1])[0][0], str(r[0]).strip(), r[3], r[2]) for r in rnd.sample(rows, min(n, len(rows)))]


def ftop(r: dict) -> str:
    """最终输出的落域"""
    if r.get("error"):
        return "ERROR"
    return "+".join(x["domain"] for x in r["results"]) or "(空)"


def mtop(r: dict) -> str:
    """模型分类（debug.nlu_result）的落域"""
    if r.get("error"):
        return "ERROR"
    if r.get("model"):
        return ("multi:" if r.get("multi") else "") + "+".join(x["domain"] for x in r["model"])
    seg = [c for _, c in r.get("segments") or [] if c]
    return f"(无分类·{'+'.join(seg) or '—'})"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("NLU3_DIALOG_URL"), help="默认读环境变量 NLU3_DIALOG_URL（地址不入库）")
    ap.add_argument("--sample", type=int, default=30, help="每个业务标签抽几句；基础车控 3 倍；多轮 2 倍")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "runs/nlu3_probe"))
    a = ap.parse_args()
    if not a.url:
        raise SystemExit("请设置 NLU3_DIALOG_URL 或传 --url（服务地址向 NLU 团队获取，不写进仓库）")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient() as c:
        djobs = [(d, i, q) for d, qs in DOMAIN_CASES.items() for i, q in enumerate(qs)]
        dres = await asyncio.gather(*[ask(c, a.url, q, sem) for _, _, q in djobs])
        sjobs = samples(a.sample, 0)
        sres = await asyncio.gather(*[ask(c, a.url, q, sem) for _, q in sjobs])
        mjobs = multiturn(a.sample * 2, 0)

        async def dialog(prev, cur):
            sid = f"probe-mt-{uuid.uuid4().hex[:10]}"
            return await ask(c, a.url, prev, sem, sid, 1), await ask(c, a.url, cur, sem, sid, 2)
        mres = await asyncio.gather(*[dialog(p, q) for p, q, _, _ in mjobs])

    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for (d, i, q), r in zip(djobs, dres):
            f.write(json.dumps({"set": "domain", "expect": d, "case": i, **r}, ensure_ascii=False) + "\n")
        for (src, q), r in zip(sjobs, sres):
            f.write(json.dumps({"set": "sample", "source": src, **r}, ensure_ascii=False) + "\n")
        for (p, q, pl, ql), (r1, r2) in zip(mjobs, mres):
            f.write(json.dumps({"set": "multiturn", "prev": p, "prev_label": pl, "label": ql, "turn1": r1, **r2}, ensure_ascii=False) + "\n")

    allr = dres + sres + [x for pair in mres for x in pair]
    ok_ms = [r["ms"] for r in allr if not r.get("error")]
    errs = [r for r in allr if r.get("error")]
    ver = next((r.get("version") for r in allr if r.get("version")), "?")
    L = [f"# NLU3.0 测试环境落域摸底\n",
         f"- 接口：`{a.url}`，服务版本 `{ver}`，{time.strftime('%Y-%m-%d %H:%M')} 实测；请求格式按 docs/nlu3/Dialogue_4b接口参数说明.txt（debug=true，enable_hotfix=false，多轮同一 session_id）。文档里的内网地址 <对话系统内网地址> 从本机不可达",
         f"- 请求 {len(allr)} 次，错误 {len(errs)} 次；时延 p50 {statistics.median(ok_ms):.0f} ms，p90 {sorted(ok_ms)[int(.9 * len(ok_ms)) - 1]:.0f} ms",
         "- 落域看两层：**模型分类**取 `debug.nlu_result`（NLU3.0 的落域判断，含多意图）；**最终输出**取 `Payload…Nlu.Domain`（过了对话管理和拒识，测试环境里很多技能没接，会变成 other）",
         "", "## 一、45 个域逐域测试（每域 5 句：清单例句 + 4 句同类说法）", ""]
    rows, summary = [], []
    for d in DOMAIN_CASES:
        rs = [r for (dd, _, _), r in zip(djobs, dres) if dd == d]
        mh = [any(x["domain"] == d for x in r.get("model") or []) for r in rs]
        fh = [any(x["domain"] == d for x in r.get("results") or []) for r in rs]
        miss = [f"{r['query']} → {mtop(r)}" for r, h in zip(rs, mh) if not h]
        summary.append((d, sum(mh), sum(fh)))
        rows.append(f"| `{d}` | {sum(mh)}/5 | {sum(fh)}/5 | {'；'.join(miss) or '—'} |")
    full = [d for d, m, _ in summary if m == 5]; part = [d for d, m, _ in summary if 0 < m < 5]; none = [d for d, m, _ in summary if m == 0]
    tm, tf = sum(m for _, m, _ in summary), sum(f for _, _, f in summary)
    L += [f"- **模型分类**：225 句命中 {tm} 句（{tm / 225:.0%}）；5 句全中 {len(full)} 个域，部分命中 {len(part)} 个，未命中 {len(none)} 个",
          f"  - 全中：{', '.join(full) or '—'}", f"  - 部分：{', '.join(part) or '—'}", f"  - 未命中：{', '.join(none) or '—'}",
          f"- **最终输出**：225 句命中 {tf} 句（{tf / 225:.0%}）；最终输出能落到的域（≥1 句）：{', '.join(d for d, _, f in summary if f) or '—'}", "",
          "| 域 | 模型分类命中 | 最终输出命中 | 模型分类未命中的句子 → 模型分类 |", "|---|---|---|---|", *rows,
          "", "## 二、评测数据抽样", "",
          "| 来源 | 句数 | 模型无分类 | 多意图 | 拒识 | 最终输出为 other | 模型分类分布（前 6） |", "|---|---|---|---|---|---|---|"]
    by = collections.defaultdict(list)
    for (src, _), r in zip(sjobs, sres):
        by[src].append(r)
    for src, rs in by.items():
        ok = [r for r in rs if not r.get("error")]; n = max(1, len(ok))
        dist = collections.Counter(mtop(r) for r in ok)
        L.append(f"| {src} | {len(ok)} | {sum(1 for r in ok if not r.get('model')) / n:.0%} | {sum(1 for r in ok if r.get('multi'))} | "
                 f"{sum(1 for r in ok if r.get('dbg_rejected')) / n:.0%} | {sum(1 for r in ok if all(x['domain'] == 'other' for x in r['results'])) / n:.0%} | "
                 + "，".join(f"{k} {v}" for k, v in dist.most_common(6)) + " |")
    L += ["", "## 三、多轮（快思考测试集_多轮：同一 session 先发上一轮，再发本轮）", "",
          "| 上一轮标签 → 本轮标签 | 组数 | 本轮判为“明确相关”上文 | 本轮模型分类分布（前 5） |", "|---|---|---|---|"]
    mb = collections.defaultdict(list)
    for (p, q, pl, ql), (_, r2) in zip(mjobs, mres):
        mb[f"{pl} → {ql}"].append(r2)
    for k, rs in sorted(mb.items(), key=lambda kv: -len(kv[1])):
        ok = [r for r in rs if not r.get("error")]
        ctx = sum(1 for r in ok if (r.get("rejection") or {}).get("contextual") == "明确相关")
        dist = collections.Counter(mtop(r) for r in ok)
        L.append(f"| {k} | {len(ok)} | {ctx} | " + "，".join(f"{x} {y}" for x, y in dist.most_common(5)) + " |")
    L += ["", "多轮样例：", *[f"- {p} → **{q}** ⇒ {mtop(r2)}" + (f"（{r2['nlu_result']}）" if r2.get("nlu_result") else "")
                              for (p, q, _, _), (_, r2) in list(zip(mjobs, mres))[:12]]]
    confs = collections.Counter(round(x["conf"], 3) for r in allr for x in r.get("results") or [] if x.get("conf") is not None)
    L += ["", "## 四、其他", "", "- 最终输出 Confidence 取值（次数）：" + "，".join(f"{k}（{v}）" for k, v in confs.most_common(8))]
    if errs:
        L.append(f"- 错误 {len(errs)} 条，例：" + "；".join(f"{r['query']} → {r['error']}" for r in errs[:5]))
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:12]))
    print(f"\n报告：{out / 'report.md'}")


if __name__ == "__main__":
    asyncio.run(main())
