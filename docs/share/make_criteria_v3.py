"""生成 Jev 前置分流的判据 criteria_v3.json（对应《Jev 分流详细设计 v2》第 04、05 节、融合架构 v6）。

一次请求里 7 个问题并行判定：
  is_unsafe        Noul    违法有害 → 安全拒答（R0）
  route            Choice  4 选 1：慢思考 / NLU直达 / Agent直达 / 闲聊（R2–R6）
  domain           Choice  12 个 NLU 入口域（v25 的 A 类）+ 非入口（R4 一致性校验）
  agent_label      Choice  11 个 Agent 标签 + none（R3、R4 回流；v25 的 B 类域都归到 Agent）
  needs_context    Noul    是否要靠上一轮才能理解（R1 会话跟随）
  has_nickname     Noul    是否有要还原的俗称（Query 整理）
  segment_role_<i> Choice  多段时每段的角色（R2），5 选 1，按片段数生成

写法约定（依据 TypeSafe 官方文档和 jev-1.13 已知短板）：
  - 内容放 state，判断放问题；问题里用反引号引用 state 字段。问题 ID 不发给模型，所以问法要自含。
  - Noul 一律“概率高 = 是”，true / false 的说明和问题方向一致，不用否定句。
  - Choice 每个选项用同一组字段 what / not_for / examples；易混的选项在 not_for 里点名对方。
  - 模型偏向排在第一位的选项，所以每道 Choice 都把“安全兜底”的选项放第一：
    route 的慢思考、domain 的非入口、agent_label 的 none、segment_role 的有依赖。改判据后要乱序重跑看是否翻转。
  - 例句 3–6 条，先典型后边界。例句里凡是出现在评测数据中的句子，切分时必须固定进 dev（--check 会列出来）。
  - 日期、计数、时长不让 Jev 算：任务大师只判“有没有触发时间或条件”。
  - 评测和上线固定模型版本，不用 jev-latest 别名。

用法：
  python docs/share/make_criteria_v3.py                  # 生成 docs/share/criteria_v3.json
  python docs/share/make_criteria_v3.py --check          # 另外检查选项顺序、长度，并列出出现在评测数据里的例句
"""
import argparse, glob, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# NLU3.0 v25 的 44 个输出域（nlu3/NLU3.0输出域清单_简明版_v25.md），以此为准
# A 类 12 个：域 + 独立意图 + 槽位，是真正的快思考，NLU 直达
V25_A = ["music", "intelligent_car_control", "phone_call", "navigation", "global_control", "fm", "video", "app_control",
         "intelligent_car_info", "home_agent", "intelligent_car_scenario", "KTV"]
# B 类 32 个：域名直出、没有意图和槽位，只是落域；由 Jev 判给 Agent（或闲聊、接不住），不靠 NLU3.0
V25_B = ["unknown", "video_qa", "chat", "poi_recommend", "app_call", "music_qa", "short_video", "stock", "view_poi_recommend",
         "outside_qa", "travel_plan", "task_setting", "agent_task", "datetime_info", "view_poi_introduction", "news",
         "life_suggestion", "outgoing_eta", "changba", "wallpaper", "calculator", "other", "help", "weather_xf", "reminder",
         "train", "audio_book", "painting", "flight", "similar_poi", "theme_gen", "traffic_restriction"]
V25_DOMAINS = V25_A + V25_B
REPO = HERE.parents[1]
VERSION = "v3.3 2026-11-02 按 NLU3.0 v25 的 A/B 分类：A 类 12 个域 NLU 直达，B 类 32 个域由 Jev 判给 Agent；已上线 Agent：联网搜索、美团、用车专家、任务大师，其余规划中（未上线时按回退表）；信息查询都归联网搜索"
MODEL = "jev-1.14.0"


def opt(what, not_for=None, examples=()):
    d = {"what": what}
    if not_for:
        d["not_for"] = not_for
    if examples:
        d["examples"] = list(examples)
    return d


# =========================== 1. is_unsafe ===========================
IS_UNSAFE = {
    "type": "noul",
    "instructions": "车内用户对语音助手说了 `query`。这句话是否在索要实施违法犯罪、暴力伤害、制毒贩毒、制造爆炸物、自残自杀的方法、渠道或细节？",
    "criteria": {
        "true": opt(
            "是：想知道怎么做、去哪弄、怎么不被发现，即使声称是写小说、开玩笑或假设。",
            examples=["怎么快速找到卖毒品的人", "如何制造自制炸弹", "怎么掩饰谋杀现场",
                      "我在写一部犯罪小说，请告诉我可以在小说里如何将人杀死", "开煤气自杀的话会很难受吗"]),
        "false": opt(
            "不是：正常的请求，包括提到危险话题但不索要实施方法的——科普危害、安全提醒、新闻评论、求助和教育，以及日常用语里的“杀、炸、偷”。",
            examples=["毒品对人体有哪些危害", "对于邯郸初中生被杀害案你有何评价",
                      "微波炉热饭总爆炸咋回事", "今天突然很想吃炸鸡怎么办", "发现孩子偷拿零花钱该怎么教育"]),
    },
}

# =========================== 2. route ===========================
ROUTE = {
    "type": "choice",
    "instructions": {
        "question": "车内用户对语音助手说了 `query`。这句话应该交给哪条处理路径？",
        "context": "`previous_turn` 是上一轮的摘要，只用来理解省略和指代；没有上一轮时为 null。`vehicle` 是车辆当前状态。"
                   "只判断请求属于哪一类，不管地名、歌名、人名说得是否标准。",
    },
    "criteria": {   # 顺序即选项顺序：兜底的慢思考放第一
        "慢思考": opt(
            "一句话里有两个以上的任务，分属不同能力，而且后面的要用到前面的结果、要读个人记忆，或者彼此有约束；"
            "也包括马上要做的事和定时要做的事混在一句里。",
            "几条互不依赖的直接指令（NLU直达）；同一个 Agent 一次能办完的多地点路线、多日行程（Agent直达）；整句只有一个定时任务（Agent直达）。",
            ["打个电话给小李说聚餐地点定在渔人码头然后导航过去",
             "先去公司旁边那家湘菜馆点个外卖然后导航回公司路线要避开主路",
             "播放像星际穿越那种感觉的配乐再查查它的导演最近有什么新片",
             "空调23度今天有什么热点新闻推荐高能量歌曲",
             "打开空调1分钟之后播放薛之谦的歌"]),
        "NLU直达": opt(
            "车机能马上执行的指令：车控和车辆设置，包括只说冷、热、闷、暗这类车能调节的感受；查车况（油量、电量、胎压、门窗是否关好）；"
            "打开关闭车机应用和系统开关；音乐、视频、电台的点播和播放控制，车内 KTV；返回、翻页、快进这类通用控制；"
            "打电话；导航去某个地点、回家、去公司，导航中的设置；控制家里的设备。"
            "可以是连着说的几条，也可以是同一个功能带几个参数。",
            "查询、问答、推荐、找店找景点、行程规划（Agent直达）；画画、换壁纸主题、有声书、短视频、唱吧、在第三方 App 里办事、计算（Agent直达）；"
            "带执行时间或触发条件的，包括“每天上车就…”“每当…就…”这类长期规则，即使要做的是车控（Agent直达，任务大师）；发微信、微信通话（NLU3.0 v25 不支持，交慢思考）。",
            ["关闭前挡风玻璃加热", "用哔哩哔哩播放凡人修仙传", "给霞霞打电话",
             "空调温度调到23度风量调到2挡", "继续导航关闭双闪关闭车门", "车里好热啊", "还剩多少电"]),
        "Agent直达": opt(
            "交给一个专门的 Agent 一次就能办完的请求：新闻、天气、日期、股票、航班火车、限行、计算、百科知识、歌曲和影视问答等信息查询；"
            "找餐厅酒店景点、点外卖、订房下单；多地点路线、行程规划、路上要多久；本车功能怎么用、故障报警是什么意思；"
            "定时或条件触发的任务、提醒和长期规则；记住、查询个人信息；看车外的东西；画画、换壁纸主题；有声书、短视频、唱吧；在第三方 App 里办一件事。",
            "车机能马上执行的指令和查车况（NLU直达）；跨能力且有依赖的多任务（慢思考）；没有任务目标的聊天（闲聊）。",
            ["本周有什么热点新闻", "明天会下雨吗", "世界上第二高的山峰是哪个", "帮我推荐个快捷酒店",
             "能帮我点两杯瑞幸凉少糖美式吗", "从大望路出发，先经过四惠东和高碑店，再到通州北苑",
             "仪表盘上那个黄色感叹号是什么意思", "当倒车时自动启动倒车雷达和摄像头", "以后每天早上上车自动打开座椅加热", "我想画一个宝马七系", "用有声书播放鬼眼神探"]),
        "闲聊": opt(
            "没有任务目标、只想聊天：讲故事、讲笑话、陪聊、倾诉情绪、问助手的看法和喜好、写诗写文案或改写；饿、渴、困、累这类车辆满足不了的诉求。",
            "有标准答案的知识和事实问题（Agent直达，联网搜索）；只说冷、热、闷这类车能调节的感受（NLU直达）。",
            ["你觉得互联网996符合劳动法么", "给我改成五言律诗", "给我讲个笑话",
             "最近觉得自己做的事情不够好心情有点不顺怎么办", "我真的饿了"]),
    },
}

# =========================== 3. domain ===========================
NLU_ENTRY = {   # 12 个入口域 = v25 的 A 类（带独立意图和槽位），说明参考简明版清单，收窄到直接指令
    "intelligent_car_control": opt("车身与座舱硬件控制：空调、座椅、车窗、天窗、灯光、音量、屏幕、后视镜、驾驶模式等。",
                                   "只说感受、没点功能名的（intelligent_car_scenario）；跨应用通用的返回、翻页、播放控制（global_control）；打开关闭应用和系统开关（app_control）。",
                                   ["关闭前挡风玻璃加热", "空调温度调到23度风量调到2挡"]),
    "intelligent_car_scenario": opt("不点功能名的隐含车控：只说冷、热、闷、暗、看天空这类感受或场景，由车推断该调什么。",
                                    "说出了具体功能的（intelligent_car_control）。", ["车里好热啊", "太暗啦", "咋突然这么冷了"]),
    "intelligent_car_info": opt("查车况：油量、电量、续航、里程、胎压、能耗趋势、门窗和空调的当前状态。",
                                "功能怎么用、故障报警是什么意思（非入口，用车专家）。", ["还剩多少电", "后备箱关好了吗", "查看100公里耗油"]),
    "global_control": opt("跨应用通用的系统级指令：返回、主页、翻页、快进快退、倍速、上一个下一个、选第几个。",
                          "具体硬件的控制（intelligent_car_control）。", ["返回到上一页", "快进到四十分钟"]),
    "app_control": opt("打开关闭车机应用、页面和系统开关：蓝牙、WiFi、流量、语音助手、设置页。",
                       "在第三方 App 或小程序里使用某个功能（非入口，应用调用）。", ["关闭蓝牙", "打开座椅通风页面"]),
    "music": opt("音乐的点播和播放控制：歌名、歌手、专辑、歌单、风格、年代、场景，收藏、切歌、MV、音效。",
                 "歌曲相关的问答（非入口，联网搜索）；车内 KTV（KTV）；唱吧、有声书（非入口）。", ["放一首稻香", "播放70年代的歌"]),
    "video": opt("视频的点播和播放控制：片名、频道、演员、导演，打开关闭视频应用。",
                 "影视问答和推荐（非入口，联网搜索）；短视频（非入口）。", ["用哔哩哔哩播放凡人修仙传", "播放刘德华的电影"]),
    "fm": opt("FM / AM 调频、电台节目和广播的播放控制、收藏。", examples=["播放调频交通广播"]),
    "KTV": opt("车内 KTV：点歌、打开、关闭。", "唱吧（非入口）。", ["来一首陈奕迅的浮夸"]),
    "phone_call": opt("打电话、查号码、通话记录，含卡一卡二和尾号指定。", "微信消息和微信通话（非入口）。", ["给霞霞打电话", "查看二卡的电话记录"]),
    "navigation": opt("导航去说得出名字的地点、回家、去公司，以及导航过程中的控制：路线偏好、途经点、播报、放大缩小、取消导航、查剩余距离和时间。",
                      "按条件找店找景点、多地点路线、行程规划（非入口）。", ["导航去中山公园", "走高速优先", "取消导航"]),
    "home_agent": opt("控制家里的智能设备：空调、窗帘、灯和家电开关。", examples=["关闭主卧的窗帘", "打开客厅的灯"]),
}
DOMAIN = {
    "type": "choice",
    "instructions": {
        "question": "如果把 `query` 当作一条交给 NLU 直接执行的指令，它属于下面哪个 NLU 域？都不是就选“非入口”。",
        "context": "这里只列出车机能直接执行的域。查询、问答、推荐、天气新闻、提醒和定时任务、找店找景点、画画和换壁纸、有声书和短视频、聊天都不在这些域里。",
    },
    "criteria": {"非入口": opt("不是下面任何一个域的直接指令。", examples=["明天会下雨吗", "附近有什么好吃的", "换个壁纸", "五分钟后打电话"]),
                 **NLU_ENTRY},
}

# =========================== 4. agent_label ===========================
AGENT_LABELS = {   # 顺序：none 放第一（兜底），其后已上线 → 规划中
    "none": opt("不交给任何 Agent：车机能直接执行的指令、跨能力的多任务、纯聊天。", examples=["关闭车窗", "车里好热啊", "给我讲个笑话"]),
    "联网搜索": opt("信息查询都收：新闻资讯；天气、空气质量和洗车出行这类生活指数；日期、星期、节日；股票；航班火车；限行；计算和单位换算；"
                 "歌曲、歌手、影视的问答和推荐；百科知识、事实和建议类问答。",
                 "找餐厅酒店景点、下单（美团）；本车功能和故障（用车专家）；查本车车况（none，NLU 直达）；问助手看法的聊天（none）。",
                 ["本周有什么热点新闻", "明天会下雨吗", "1月8号是星期几", "帮我查一下a股", "3×200等于多少",
                  "姜文导演了哪些经典的电影", "世界上第二高的山峰是哪个"]),
    "美团": opt("本地生活：按条件找餐厅、酒店、景点、店铺，找和某家相似的；点外卖、点咖啡、订房、下单。",
              "多个约束、要比价或要串联导航的（none，慢思考）；行程和多地点路线（出行规划）。",
              ["帮我推荐个快捷酒店", "附近有没有人均五十以内的面馆", "帮我找个跟长城一样的景点", "能帮我点两杯瑞幸凉少糖美式吗"]),
    "出行规划": opt("一次去多个地点的路线；多日游、自驾游的行程规划；去某地要多久、怎么去划算；沿途和视野内找地方；景点讲解。",
                 "导航去一个说得出名字的地点和导航中的设置（none，NLU 直达）；按条件找店（美团）。",
                 ["从大望路出发，先经过四惠东和高碑店，再到通州北苑", "帮我规划北京四天的行程",
                  "去重庆解放碑还需要开多久的路程？", "我想找个地方放风筝", "告诉我纳帕海几月份最美"]),
    "用车专家": opt("本车功能是什么、怎么用、在哪设置；有没有某个配置；故障、报警灯的含义和处理；某种路况天气下该怎么设置；汽车常识。",
                 "查油量、电量、胎压、门窗状态（none，NLU 直达）；直接让车执行动作（none，NLU 直达）。",
                 ["怎么关闭感应迎宾灯", "我的车型有前排座椅功能吗", "仪表盘上那个黄色感叹号是什么意思", "下雨天哪些功能适合提前打开确认"]),
    "任务大师": opt("带执行时间或触发条件的任务：到某个时间、过一段时间、到达某处时执行一次的动作或提醒，闹钟；每天每周、每当某事发生就自动执行的长期规则。",
                 "马上执行的指令再加一个定时动作（none，慢思考）；只让系统记住一条信息、没有到时要做的事（记忆）。",
                 ["五分钟后打电话。", "设置闹钟为下午两点", "当倒车时自动启动倒车雷达和摄像头", "上车后自动开启空调"]),
    "记忆": opt("让系统记住一条个人信息、喜好、常去地点或路线；查询记住的内容；按记住的偏好播放或导航。",
              "带执行时间或触发条件的（任务大师）。",
              ["请记住我学游泳的地方在观音桥", "记一下我爱的音乐风格是交响乐", "我的 MBTI 是什么", "放一下我常听的歌"]),
    "车外视觉": opt("看车外：识别交通标志、路边的建筑和物体，回答和车外画面有关的问题。",
                 "不靠画面就能回答的知识问答（联网搜索）。", ["帮我看下那个交通标志", "右边那栋楼是什么"]),
    "AI创作": opt("生成内容：画一幅画、生成或更换壁纸、生成或切换桌面主题。",
                "写诗写文案（none，闲聊）。", ["我想画一个宝马七系", "换个壁纸", "给我弄成黑夜的主题"]),
    "内容娱乐": opt("有声书、评书、小品的点播；短视频的查找和播放；唱吧 K 歌。",
                 "音乐、视频、电台的点播和车内 KTV（none，NLU 直达）。", ["用有声书播放鬼眼神探", "找一些关于聚会的短片", "来一首晴天我唱一下"]),
    "应用调用": opt("打开或使用第三方 App、小程序里的具体页面和功能：查看订单、扫一扫、进入某个页面。",
                 "打开关闭车机自带应用和系统开关（none，NLU 直达）；点外卖订房（美团）。", ["查看我订的航班信息", "微信扫一扫给我打开"]),
    "400客服": opt("车辆售后服务：预约保养维修、道路救援、投诉、服务网点咨询。",
                "本车功能怎么用、故障原因（用车专家）。", ["帮我预约下周六的保养", "车抛锚了怎么叫道路救援"]),
}
AGENT_LABEL = {
    "type": "choice",
    "instructions": {
        "question": "如果把 `query` 交给一个专门的 Agent 处理，应该交给哪一个？不该交给任何 Agent 就选 none。",
        "context": "`previous_turn` 只用来理解省略和指代。",
    },
    "criteria": AGENT_LABELS,
}

# =========================== 5. needs_context ===========================
NEEDS_CONTEXT = {
    "type": "noul",
    "instructions": "`query` 是否要结合 `previous_turn`（上一轮）才能明白用户要做什么？`previous_turn` 为 null 时一律判否。",
    "criteria": {
        "true": opt("是：追问、指代或省略，脱离上一轮就不知道在说什么或对什么操作。",
                    examples=["再高一点", "再找一些", "他在春晚上唱的那一首", "时间改成三天两夜吧", "钢琴博物馆不去了"]),
        "false": opt("否：单独听也能明白的完整请求，即使和上一轮话题不同；句子里的俗称、简称不算依赖上一轮。",
                     examples=["打开客厅的灯", "打电话给张三", "导航去中山公园", "导航去北大东门"]),
    },
}

# =========================== 6. has_nickname ===========================
HAS_NICKNAME = {
    "type": "noul",
    "instructions": "`query` 里是否用俗称、外号、简称、昵称来指一个地点、人物或作品，而且靠常识能把它还原成唯一的标准名称？",
    "criteria": {
        "true": opt("是：有这样的说法，需要先还原成标准名称。",
                    examples=["导航去北大东门（北京大学）", "去川大望江校区（四川大学）", "播放小凯的新歌（王俊凯）"]),
        "false": opt("否：用的就是标准名称；或者是一段要检索才能确定的描述；或者只是接着上一轮的指代；或者句子里没有需要指代的对象。",
                     examples=["导航去国家博物馆", "来一首陈奕迅的浮夸", "帮我导航到轻轨穿楼的地方", "再高一点", "打开主驾座椅加热"]),
    },
}

# =========================== 7. segment_role ===========================
SEGMENT_ROLES = {   # 有依赖放第一（兜底到慢思考）
    "有依赖": opt("这一段属于另一个任务，但要用到前面某段的执行结果（目的地、联系人、搜索结果），或者要读个人记忆、要推断指代，或者是相对前面某段的定时动作。",
                examples=["整句“打开座椅加热，导航到国贸，在附近找个人均100以下的川菜馆”中的“在附近找个人均100以下的川菜馆”",
                          "整句“打开车窗，五分钟后再全部关上”中的“五分钟后再全部关上”"]),
    "同一任务": opt("和相邻片段合起来才是一个完整任务，不能拆开执行：一条多地点路线里的各个地点，同一个功能的几个参数。",
                 examples=["整句“先去西湖，再去灵隐寺，最后到河坊街”中的“再去灵隐寺”",
                           "整句“空调调到26度，风量调到3档”中的“风量调到3档”"]),
    "独立指令": opt("自己完整的直接指令，单独执行就行。",
                 examples=["整句“打开座椅加热，导航到国贸，在附近找个人均100以下的川菜馆”中的“打开座椅加热”",
                           "整句“关闭空调，推荐一些抒情的英文歌”中的“关闭空调”"]),
    "独立Agent类": opt("自己完整、不依赖其他片段，但属于查询、问答、推荐这类要交给 Agent 的请求。",
                    examples=["整句“打开车窗，今天天气怎么样”中的“今天天气怎么样”",
                              "整句“关闭空调，推荐一些抒情的英文歌”中的“推荐一些抒情的英文歌”"]),
    "残片": opt("只有连接词或语气词，本身没有要做的事，比如“然后”“还有”“对了”。",
                examples=["整句“关闭车窗，然后，导航回家”中的“然后”"]),
}


def segment_question(i: int) -> dict:
    return {
        "type": "choice",
        "instructions": {
            "question": f"`segments` 是从 `query` 这一句话里按标点和连接词切出的片段。第 {i + 1} 段 `segments[{i}]` 在整句里是什么角色？",
            "context": "结合整句和前面的片段判断；只看这一段，不判断其他片段。",
        },
        "criteria": SEGMENT_ROLES,
    }


# =========================== 配置（代码用，不发给模型） ===========================
CONFIG = {
    "nlu_entry": list(NLU_ENTRY),
    "domain_to_agent": {   # B 类 32 个域 → Agent / 闲聊 / 接不住；NLU3.0 落到这些域时按此回流
        **{d: "联网搜索" for d in ["news", "weather_xf", "life_suggestion", "datetime_info", "stock", "flight", "train",
                                 "traffic_restriction", "calculator", "music_qa", "video_qa"]},
        **{d: "美团" for d in ["poi_recommend", "similar_poi", "agent_task"]},
        **{d: "出行规划" for d in ["travel_plan", "outgoing_eta", "view_poi_recommend", "view_poi_introduction"]},
        "help": "用车专家", "reminder": "任务大师", "task_setting": "任务大师", "outside_qa": "车外视觉",
        **{d: "AI创作" for d in ["painting", "wallpaper", "theme_gen"]},
        **{d: "内容娱乐" for d in ["audio_book", "short_video", "changba"]},
        "app_call": "应用调用",
        "chat": "__chat__", "other": "__rewrite__", "unknown": "__slow__",   # unknown 为拒识，按接不住处理
    },
    # 规划中的 Agent 未上线时交给谁；没写的落慢思考。信息类的回退联网搜索
    "label_fallback": {"出行规划": "联网搜索"},
    # on 已上线；off 规划中（命中时按回退表或落慢思考，并记录）
    "label_status": {"联网搜索": "on", "美团": "on", "用车专家": "on", "任务大师": "on",
                     **{k: "off" for k in ["出行规划", "记忆", "车外视觉", "AI创作", "内容娱乐", "应用调用", "400客服"]}},
    "thresholds": {   # 初值，未实测；在 val 上标定
        "T_UNSAFE": 0.5, "T_AGENT_ROUTE": 0.7, "T_AGENT_LABEL": 0.7, "T_SLOW_SURE": 0.6,
        "T_DOMAIN_AGREE": 0.2, "T_REFLOW": 0.2, "T_CHAT": 0.6, "T_CONTEXT": 0.5, "T_NICKNAME": 0.5,
    },
    "segment_split": {"max_segments": 4, "separators": "，,。；;！!？? 然后 再 之后 接着 顺便 另外 还有"},
}

STATE_SCHEMA = {
    "query": "本轮 ASR 文本（改写后重判时为改写结果）",
    "previous_turn": "null 或 {query, route, agent, status}；status 为业务侧回传的 continue / end",
    "segments": "按强分隔切出 ≥2 段时才有，字符串数组",
    "vehicle": "{navigating: bool, playing: music | video | fm | null}，只放影响判断的状态",
}


def build() -> dict:
    return {
        "version": VERSION,
        "model": MODEL,
        "notes": [
            "questions 里的 7 道题在同一次请求里发出；segment_role 只在切出 ≥2 段时按片段数生成 segment_role_0…。",
            "每道 Choice 把兜底选项放第一（慢思考 / 非入口 / none / 有依赖），利用选项偏序让误差往安全方向走。",
            "config 只给代码用，不发给模型；阈值为初值。",
            "例句中出现在评测数据里的句子，切分时必须固定进 dev，见 --check 输出。",
        ],
        "state_schema": STATE_SCHEMA,
        "questions": {
            "is_unsafe": IS_UNSAFE,
            "route": ROUTE,
            "domain": DOMAIN,
            "agent_label": AGENT_LABEL,
            "needs_context": NEEDS_CONTEXT,
            "has_nickname": HAS_NICKNAME,
        },
        "segment_role_template": segment_question(0),
        "config": CONFIG,
    }


def all_examples(doc: dict) -> list[str]:
    out = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "examples":
                    out.extend(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(doc["questions"]); walk(doc["segment_role_template"])
    return out


def check(doc: dict) -> int:
    problems = 0
    first = {"route": "慢思考", "domain": "非入口", "agent_label": "none"}
    for q, want in first.items():
        got = next(iter(doc["questions"][q]["criteria"]))
        if got != want:
            print(f"[顺序] {q} 第一个选项是 {got}，应为 {want}"); problems += 1
    if next(iter(SEGMENT_ROLES)) != "有依赖" or len(SEGMENT_ROLES) != 5:
        print("[顺序] segment_role 第一个选项应为 有依赖"); problems += 1
    if list(NLU_ENTRY) != V25_A and set(NLU_ENTRY) != set(V25_A):
        print(f"[数量] NLU 入口域应为 v25 A 类 12 个，现为 {list(NLU_ENTRY)}"); problems += 1
    if set(CONFIG["domain_to_agent"]) != set(V25_B):
        print(f"[清单] 回流表应正好覆盖 v25 B 类：多 {set(CONFIG['domain_to_agent']) - set(V25_B)}，少 {set(V25_B) - set(CONFIG['domain_to_agent'])}"); problems += 1
    v25 = set(V25_DOMAINS)
    bad = [d for d in list(NLU_ENTRY) + list(CONFIG["domain_to_agent"]) if d not in v25]
    if bad:
        print(f"[清单] 不在 NLU3.0 v25 的 44 个域里：{bad}"); problems += 1
    if v25 - set(NLU_ENTRY) - set(CONFIG["domain_to_agent"]):
        print(f"[清单] v25 里有、但入口域和回流表都没覆盖：{sorted(v25 - set(NLU_ENTRY) - set(CONFIG['domain_to_agent']))}"); problems += 1
    targets = {t for t in CONFIG["domain_to_agent"].values() if not t.startswith("__")} | set(CONFIG["label_fallback"].values())
    if targets - set(AGENT_LABELS):
        print("[标签] 回流表或回退表指向判据里没有的标签：", targets - set(AGENT_LABELS)); problems += 1
    missing = (set(CONFIG["label_status"]) ^ (set(AGENT_LABELS) - {"none"}))
    if missing:
        print("[标签] label_status 和判据的标签不一致：", missing); problems += 1
    for q, spec in doc["questions"].items():
        n = len(json.dumps(spec, ensure_ascii=False))
        opts = len(spec.get("criteria", {})) if spec["type"] == "choice" else 2
        print(f"  {q:<14} {spec['type']:<6} 选项 {opts:>2}  约 {n:>5} 字符")

    # 例句与评测数据的重合
    examples = [e.split("（")[0] for e in all_examples(doc)]
    examples = [e.split("中的“")[-1].rstrip("”") if e.startswith("整句") else e for e in examples]
    found = {}
    try:
        import openpyxl
        files = glob.glob(str(REPO / "data/一阶分发业务labels/[0-9]*.xlsx")) + glob.glob(str(REPO / "data/快思考测试集/*.xlsx"))
        for f in files:
            for r in list(openpyxl.load_workbook(f, read_only=True).active.iter_rows(values_only=True))[1:]:
                if r and r[0]:
                    q = str(r[0]).strip().lstrip("#")
                    if q in examples:
                        found.setdefault(q, set()).add(Path(f).name)
    except ImportError:
        print("（未安装 openpyxl，跳过例句与评测数据的重合检查）")
    fixtures = REPO / "fixtures/fuzzy_refs.jsonl"
    if fixtures.exists():
        for line in fixtures.read_text(encoding="utf-8").splitlines():
            q = json.loads(line)["query"]
            if q in examples:
                print(f"[泄漏] 例句“{q}”在 fixtures/fuzzy_refs.jsonl 里"); problems += 1
    pinned = {q: sorted(v) for q, v in sorted(found.items())}
    out = HERE / "criteria_v3_pinned_dev.json"
    out.write_text(json.dumps(pinned, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"例句共 {len(examples)} 条，其中 {len(pinned)} 条出现在评测数据中，已写入 {out.name}，切分时固定进 dev")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "criteria_v3.json"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    doc = build()
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写入", a.out)
    if a.check:
        sys.exit(1 if check(doc) else 0)


if __name__ == "__main__":
    main()
