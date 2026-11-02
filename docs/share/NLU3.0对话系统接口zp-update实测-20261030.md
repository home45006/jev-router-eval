# NLU3.0 对话系统接口（/zp/update）实测

> ⚠ 本文测的是 `/zp/update` 对话系统接口（外面包了对话管理、拒识和技能执行），**不是 NLU3.0 模型本身**。模型本身的落域能力见同目录《NLU3.0-v25模型落域实测-20261030.md》。

- 接口：`https://<NLU3 测试环境域名>/zp/update`，服务版本 `2026-10-29 17:37:14`，2026-10-30 15:14 实测；请求格式按 docs/nlu3/Dialogue_4b接口参数说明.txt（debug=true，enable_hotfix=false，多轮同一 session_id）。文档里的内网地址 <对话系统内网地址> 从本机不可达
- 请求 945 次，错误 17 次；时延 p50 418 ms，p90 606 ms
- 落域看两层：**模型分类**取 `debug.nlu_result`（NLU3.0 的落域判断，含多意图）；**最终输出**取 `Payload…Nlu.Domain`（过了对话管理和拒识，测试环境里很多技能没接，会变成 other）

## 一、45 个域逐域测试（每域 5 句：清单例句 + 4 句同类说法）

- **模型分类**：225 句命中 157 句（70%）；5 句全中 18 个域，部分命中 23 个，未命中 4 个
  - 全中：music, intelligent_car_control, phone_call, navigation, global_control, fm, weather_xf, travel_plan, home_agent, stock, reminder, short_video, task_setting, datetime_info, news, flight, painting, train
  - 部分：chat, video_qa, app_call, intelligent_car_info, video, help, poi_recommend, music_qa, wechat_on_car, agent_task, view_poi_recommend, view_poi_introduction, life_suggestion, outgoing_eta, outside_qa, changba, calculator, wallpaper, other, audio_book, theme_gen, traffic_restriction, similar_poi
  - 未命中：intelligent_car_scenario, app_control, app_agent, KTV
- **最终输出**：225 句命中 38 句（17%）；最终输出能落到的域（≥1 句）：music, intelligent_car_control, phone_call, navigation, fm, chat, intelligent_car_info, poi_recommend, intelligent_car_scenario, app_control, app_agent, other

| 域 | 模型分类命中 | 最终输出命中 | 模型分类未命中的句子 → 模型分类 |
|---|---|---|---|
| `music` | 5/5 | 5/5 | — |
| `intelligent_car_control` | 5/5 | 5/5 | — |
| `phone_call` | 5/5 | 2/5 | — |
| `navigation` | 5/5 | 5/5 | — |
| `global_control` | 5/5 | 0/5 | — |
| `fm` | 5/5 | 5/5 | — |
| `chat` | 2/5 | 1/5 | 陪我聊会儿天吧 → (无分类·other)；你叫什么名字 → (无分类·chat)；我今天心情不好 → (无分类·other) |
| `video_qa` | 4/5 | 0/5 | 帮我找王祖蓝参加的剧集 → video |
| `app_call` | 2/5 | 0/5 | 用高德查一下附近的加油站 → navigation；打开微信小程序 → wechat_on_car；查看我的快递 → agent_task |
| `intelligent_car_info` | 4/5 | 3/5 | 下次保养是什么时候 → help |
| `video` | 2/5 | 0/5 | 调整播放速度为2倍 → global_control；下一集 → (无分类·other)；切换到高清 → global_control |
| `help` | 3/5 | 0/5 | 你能做什么 → chat；这个按钮是干嘛的 → (无分类·other) |
| `poi_recommend` | 4/5 | 1/5 | 找个能停车的商场 → navigation |
| `weather_xf` | 5/5 | 0/5 | — |
| `music_qa` | 4/5 | 0/5 | 这首歌叫什么名字 → music |
| `travel_plan` | 5/5 | 0/5 | — |
| `home_agent` | 5/5 | 0/5 | — |
| `stock` | 5/5 | 0/5 | — |
| `reminder` | 5/5 | 0/5 | — |
| `short_video` | 5/5 | 0/5 | — |
| `task_setting` | 5/5 | 0/5 | — |
| `wechat_on_car` | 3/5 | 0/5 | 我要打视频给小阿萨 → (无分类·other)；回复他我马上到 → (无分类·other) |
| `agent_task` | 4/5 | 0/5 | 帮我叫个代驾 → (无分类·other) |
| `datetime_info` | 5/5 | 0/5 | — |
| `view_poi_recommend` | 4/5 | 0/5 | 附近有没有可以遛狗的公园 → poi_recommend |
| `view_poi_introduction` | 3/5 | 0/5 | 介绍一下西湖 → chat；兵马俑的历史 → chat |
| `news` | 5/5 | 0/5 | — |
| `life_suggestion` | 3/5 | 0/5 | 今天紫外线强吗 → weather_xf；穿衣指数 → weather_xf |
| `intelligent_car_scenario` | 0/5 | 4/5 | 太暗啦 → (无分类·intelligent_car_scenario)；好热啊 → (无分类·intelligent_car_scenario)；车里有点闷 → (无分类·intelligent_car_scenario)；我有点冷 → (无分类·intelligent_car_scenario)；雾太大看不清 → (无分类·other) |
| `outgoing_eta` | 2/5 | 0/5 | 去公司要多久 → navigation；到机场还要多长时间 → navigation；几点出发能赶上九点的会 → chat |
| `outside_qa` | 3/5 | 0/5 | 前面那个牌子写的什么 → (无分类·other)；路边那个是什么店 → navigation |
| `changba` | 3/5 | 0/5 | 打开唱吧 → KTV；关闭唱吧 → KTV |
| `app_control` | 0/5 | 1/5 | 开启语音助手和车载WiFi → multi:intelligent_car_control+intelligent_car_control；打开蓝牙 → intelligent_car_control；打开导航 → intelligent_car_control；关闭WiFi → intelligent_car_control；打开设置 → (无分类·other) |
| `calculator` | 4/5 | 0/5 | 一英里等于多少公里 → chat |
| `app_agent` | 0/5 | 1/5 | 微信扫一扫给我打开 → (无分类·app_agent)；打开支付宝付款码 → app_call；刷一下微博热搜 → app_call；打开淘宝搜一下充电宝 → app_call；美团上找一家奶茶店 → navigation |
| `wallpaper` | 4/5 | 0/5 | 换个壁纸 → intelligent_car_control |
| `other` | 1/5 | 5/5 | 再小一点 → (无分类·other)；那个呢 → (无分类·other)；就这个吧 → (无分类·other)；嗯嗯 → (无分类·other) |
| `flight` | 5/5 | 0/5 | — |
| `painting` | 5/5 | 0/5 | — |
| `audio_book` | 1/5 | 0/5 | 播放三体有声书 → fm；我想听评书 → fm；播放郭德纲的相声 → fm；继续播放上次的有声书 → fm |
| `KTV` | 0/5 | 0/5 | 来一首陈奕迅的浮夸 → music；打开KTV → (无分类·other)；我要唱K → changba；切歌 → (无分类·other)；原唱伴唱切换 → changba |
| `train` | 5/5 | 0/5 | — |
| `theme_gen` | 1/5 | 0/5 | 给我弄成黑夜的主题 → intelligent_car_control；换个主题 → intelligent_car_control；切换成科技风主题 → intelligent_car_control；主题换成粉色 → intelligent_car_control |
| `traffic_restriction` | 3/5 | 0/5 | 打开限行区域 → navigation；外地车能进五环吗 → outside_qa |
| `similar_poi` | 3/5 | 0/5 | 类似迪士尼的乐园 → (无分类·other)；跟西湖差不多的地方 → (无分类·other) |

## 二、评测数据抽样

| 来源 | 句数 | 模型无分类 | 多意图 | 拒识 | 最终输出为 other | 模型分类分布（前 6） |
|---|---|---|---|---|---|---|
| 快思考测试集·基础车控 | 89 | 11% | 1 | 12% | 84% | (无分类·other) 10，music_qa 8，chat 7，intelligent_car_info 7，wechat_on_car 7，phone_call 6 |
| 01_记忆agent | 30 | 27% | 0 | 27% | 50% | navigation 10，(无分类·other) 8，chat 5，music 5，reminder 2 |
| 02_出行导航agent | 29 | 24% | 0 | 24% | 83% | (无分类·other) 7，travel_plan 6，navigation 5，view_poi_introduction 4，view_poi_recommend 3，poi_recommend 2 |
| 03_音乐agent | 28 | 7% | 0 | 7% | 57% | music_qa 14，music 11，(无分类·other) 2，intelligent_car_control 1 |
| 05_用车agent | 30 | 23% | 1 | 27% | 97% | help 18，(无分类·other) 7，intelligent_car_info 3，multi:intelligent_car_control+intelligent_car_control 1，intelligent_car_control 1 |
| 06_闲聊 | 30 | 23% | 0 | 20% | 97% | chat 17，(无分类·other) 6，help 2，agent_task 1，view_poi_introduction 1，view_poi_recommend 1 |
| 07_有情知音 | 30 | 10% | 0 | 10% | 100% | chat 14，life_suggestion 7，(无分类·other) 3，music_qa 2，weather_xf 1，view_poi_introduction 1 |
| 08_影视agent | 30 | 7% | 0 | 7% | 100% | video_qa 19，video 8，(无分类·other) 2，music_qa 1 |
| 09_新闻agent | 30 | 10% | 0 | 10% | 97% | news 23，chat 3，(无分类·other) 3，music 1 |
| 10_任务大师 | 30 | 3% | 0 | 3% | 97% | task_setting 27，intelligent_car_control 1，navigation 1，(无分类·other) 1 |
| 11_条件任务 | 30 | 30% | 0 | 30% | 80% | reminder 14，(无分类·other) 9，navigation 3，intelligent_car_control 2，task_setting 1，music 1 |
| 12_中枢复杂多任务 | 26 | 35% | 8 | 58% | 81% | (无分类·other) 7，intelligent_car_control 3，help 2，(无分类·chat) 1，multi:intelligent_car_control+music+navigation 1，multi:intelligent_car_control+music 1 |
| 13_中枢简单任务 | 29 | 10% | 21 | 83% | 97% | intelligent_car_control 4，multi:intelligent_car_control+intelligent_car_control 3，multi:intelligent_car_control+intelligent_car_control+intelligent_car_control 3，(无分类·other) 3，multi:intelligent_car_control+intelligent_car_control+intelligent_car_control+intelligent_car_control 2，multi:intelligent_car_control+app_control+intelligent_car_control+app_control 1 |
| 14_车控模糊意图 | 30 | 73% | 0 | 50% | 60% | (无分类·other) 15，(无分类·intelligent_car_scenario) 7，intelligent_car_scenario 7，chat 1 |
| 16_基础语控 | 30 | 7% | 0 | 3% | 33% | intelligent_car_control 19，video 3，global_control 2，music 2，phone_call 1，help 1 |
| 17_一语多意图 | 30 | 7% | 26 | 93% | 93% | multi:intelligent_car_control+intelligent_car_control 18，multi:intelligent_car_control+music 2，(无分类·other) 2，multi:intelligent_car_control+stock+intelligent_car_control 1，intelligent_car_control 1，multi:travel_plan+weather+view_poi_recommend 1 |
| 18_同功能多参数 | 22 | 5% | 11 | 55% | 82% | intelligent_car_control 10，multi:intelligent_car_control+intelligent_car_control 9，multi:intelligent_car_control+intelligent_car_control+intelligent_car_control 2，(无分类·other) 1 |
| 99_剔除清单 | 30 | 47% | 4 | 60% | 83% | (无分类·other) 14，news 5，intelligent_car_scenario 4，multi:intelligent_car_control+intelligent_car_control 1，multi:intelligent_car_control+navigation 1，multi:intelligent_car_control+intelligent_car_control+intelligent_car_control+app_control 1 |

## 三、多轮（快思考测试集_多轮：同一 session 先发上一轮，再发本轮）

| 上一轮标签 → 本轮标签 | 组数 | 本轮判为“明确相关”上文 | 本轮模型分类分布（前 5） |
|---|---|---|---|
| 基础车控 → 基础车控 | 16 | 4 | music 3，intelligent_car_control 3，navigation 3，weather_xf 2，home_agent 2 |
| 音乐Agent → 音乐Agent | 14 | 9 | music 9，(无分类·other) 4，music_qa 1 |
| 记忆 → 记忆 | 6 | 4 | chat 2，music_qa 2，music 1，(无分类·other) 1 |
| 基础车控 → 音乐Agent | 4 | 2 | music 3，music_qa 1 |
| 用车Agent → 基础车控 | 3 | 0 | wechat_on_car 1，reminder 1，phone_call 1 |
| 基础车控 → 导航Agent | 3 | 0 | (无分类·other) 1，travel_plan 1，view_poi_introduction 1 |
| 基础车控 → 多意图 | 3 | 0 | intelligent_car_control 2，multi:intelligent_car_control+intelligent_car_control 1 |
| 导航Agent → 基础车控 | 2 | 0 | intelligent_car_control 1，reminder 1 |
| 音乐Agent → 基础车控 | 2 | 0 | wechat_on_car 1，navigation 1 |
| 多意图 → 导航Agent | 2 | 0 | (无分类·other) 1，travel_plan 1 |
| 记忆 → 基础车控 | 2 | 0 | music 1，weather_xf 1 |
| 多意图 → 音乐Agent | 1 | 0 | music_qa 1 |
| 基础车控 → 用车Agent | 1 | 0 | help 1 |
| 多意图 → 基础车控 | 1 | 0 | weather_xf 1 |

多轮样例：
- 侧方位停车怎么停 → **用微信给李四发消息** ⇒ wechat_on_car（wechat_on_car	wechat_on_car）
- 打开座椅通风和按摩 → **梦里梦到醒不来的梦是哪首歌** ⇒ music_qa（music_qa	music_qa）
- 带我去福州的那个老街区，中间一条街最热闹，两边全是名人故居的地方。 → **打开空调** ⇒ intelligent_car_control（intelligent_car_control	turn_on_device	name=空调）
- 音乐转换为随机播放 → **切下一首** ⇒ music（music	next）
- 换壁纸 → **雨夜从公司回家想听点安稳的歌** ⇒ music（music	play	scene=家,emotion=安稳）
- 我想就这样牵着你的手不放开是哪首歌 → **只要女的唱的** ⇒ (无分类·other)
- 再找一些 → **只要男人唱的** ⇒ (无分类·other)
- 抬头显示有重影怎么调 → **设个明天7点的闹钟** ⇒ reminder（reminder	reminder）
- 关闭电子狗 → **从国贸去望京SOHO开会，然后去首都机场接人。** ⇒ (无分类·other)
- 还有别的吗 → **第一首是他哪一年春晚唱的** ⇒ (无分类·other)（unknown）
- 狂飙片尾那首歌叫什么 → **用微信给李四发消息** ⇒ wechat_on_car（wechat_on_car	wechat_on_car）
- 邓紫棋在我是歌手第一场唱红的歌是哪首 → **换一批** ⇒ music_qa（music_qa	music_qa）

## 四、其他

- 最终输出 Confidence 取值（次数）：1.0（928）
- 错误 17 条，例：打开hud歌词 → Code=500 service internal error；渝中区白象街有没有民国风照相馆+咖啡的复合空间提供旗袍租赁的 → Code=500 service internal error；给我来几首清新的 Bedroom Pop → Code=500 service internal error；孙燕姿的克卜勒和跳舞的梵谷哪张更实验 → Code=500 service internal error；明天陪客户先导航去他们公司总部再去之前谈合作的那家私房菜馆路上随便聊点本地商圈的发展吧 → Code=500 service internal error

## 五、结论（人工整理）

1. **模型分类能力基本可用**：按 `debug.nlu_result` 算，45 域逐域测试 225 句命中 157 句（70%）；`nlu_result` 为空时用 `SegementInfo.category` 兜底，命中 163 句（72%）。18 个域 5 句全中，包括天气、新闻、股票、航班、火车、日期、提醒、场景任务、行程规划、车家互联、绘画等。
2. **最终输出只有 17%**：大部分域在对话管理层落成 other（测试环境的技能没接，或被拒识），所以**分流要读 `debug.nlu_result`，不能只看最外层的 `Nlu.Domain`**；正式环境是否给出分类结果、给在哪个字段，需要和 NLU 团队确认。
3. **多意图能识别但全部被拒识**：抽样里 72 句判出 `multi:`，72 句都是 `is_rejected=true`，最终输出 other。一语多意图 30 句里 26 句判出多意图（多为 `intelligent_car_control` × 2）。
4. **多轮要用同一 session 顺序发送**（不是 `histories` 字段）：“再大一点儿”“换成英文版的”“那换成去博物馆呢”都能按上文补全；测试集多轮里“承接上文”的组大多能判为“明确相关”。
5. **容易混的域**：app_control 多判成 intelligent_car_control；KTV ↔ changba 互相混；有声书、评书、相声判成 fm；换主题、换壁纸判成 intelligent_car_control；“去公司要多久”判成 navigation（不是 outgoing_eta）；隐含车控（好热啊）只在 `SegementInfo` 里有分类，`nlu_result` 为空。
6. **Confidence 没有区分度**：最终输出 928 次全是 1.0。可用的信号是 `debug.is_rejected` 和 `debug.RejectionInfo`（structure / contextual / completely / meaningful）。
7. **其他**：多意图里出现过 `weather`（不是清单里的 `weather_xf`）；长句、复杂句有 17 次返回 500；`sensitive` 域会对违法有害请求给出分类，不在 45 域清单里。
