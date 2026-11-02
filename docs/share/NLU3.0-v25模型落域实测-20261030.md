# NLU3.0 v25 模型落域摸底

- 接口：`http://<NLU3-v25 服务地址>/v1/chat/completions`（model `qwen3_nlu`，v25，Qwen3-0.6B 全量微调），2026-10-30 16:16 实测；请求按《NLU3.0大模型接口使用说明1030固定版本》：固定 system、扁平 user 串、temperature=0、top_p=0.01
- 请求 945 句，错误 0 句；单次时延 p50 93 ms，p90 235 ms

## 一、45 个域逐域测试（每域 5 句：清单例句 + 4 句同类说法）

- 225 句：严格命中 161 句（72%）；算上歧义句的可接受归属 177 句（79%）
- 可接受口径下 5 句全中 21 个域；≤ 2/5 的域：video（2/5）, wechat_on_car（0/5）, view_poi_introduction（2/5）, outside_qa（2/5）, app_agent（0/5）, audio_book（1/5）, theme_gen（2/5）

| 域 | 严格 | 含可接受 | 未严格命中的句子 → 模型输出 |
|---|---|---|---|
| `music` | 5/5 | 5/5 | — |
| `intelligent_car_control` | 5/5 | 5/5 | — |
| `phone_call` | 5/5 | 5/5 | — |
| `navigation` | 5/5 | 5/5 | — |
| `global_control` | 4/5 | 4/5 | 下一页 → phone_call |
| `fm` | 5/5 | 5/5 | — |
| `chat` | 5/5 | 5/5 | — |
| `video_qa` | 5/5 | 5/5 | — |
| `app_call` | 3/5 | 3/5 | 用高德查一下附近的加油站 → navigation；查看我的快递 → agent_task |
| `intelligent_car_info` | 4/5 | 5/5 | 下次保养是什么时候 → help（可接受） |
| `video` | 2/5 | 2/5 | 调整播放速度为2倍 → global_control；下一集 → unknown；切换到高清 → global_control |
| `help` | 4/5 | 5/5 | 你能做什么 → chat（可接受） |
| `poi_recommend` | 4/5 | 5/5 | 找个能停车的商场 → navigation（可接受） |
| `weather_xf` | 5/5 | 5/5 | — |
| `music_qa` | 4/5 | 5/5 | 这首歌叫什么名字 → music（可接受） |
| `travel_plan` | 5/5 | 5/5 | — |
| `home_agent` | 5/5 | 5/5 | — |
| `stock` | 5/5 | 5/5 | — |
| `reminder` | 5/5 | 5/5 | — |
| `short_video` | 4/5 | 4/5 | 来点抖音热门视频 → video |
| `task_setting` | 4/5 | 4/5 | 每天下班上车就把空调开到22度 → unknown |
| `wechat_on_car` | 0/5 | 0/5 | 我要打视频给小阿萨 → video；用微信给李四发消息 → chat；读一下微信消息 → chat；微信发位置给张三 → unknown；回复他我马上到 → unknown |
| `agent_task` | 4/5 | 4/5 | 帮我叫个代驾 → chat |
| `datetime_info` | 5/5 | 5/5 | — |
| `view_poi_recommend` | 4/5 | 4/5 | 附近有没有可以遛狗的公园 → navigation |
| `view_poi_introduction` | 2/5 | 2/5 | 介绍一下西湖 → chat；故宫有什么好玩的 → view_poi_recommend；兵马俑的历史 → chat |
| `news` | 4/5 | 4/5 | 来点体育新闻 → chat |
| `life_suggestion` | 3/5 | 3/5 | 今天紫外线强吗 → weather_xf；穿衣指数 → weather_xf |
| `intelligent_car_scenario` | 3/5 | 3/5 | 好热啊 → unknown；雾太大看不清 → intelligent_car_control |
| `outgoing_eta` | 1/5 | 4/5 | 去公司要多久 → navigation（可接受）；到机场还要多长时间 → navigation（可接受）；现在去火车站堵不堵 → navigation（可接受）；几点出发能赶上九点的会 → chat |
| `outside_qa` | 2/5 | 2/5 | 旁边那栋楼是什么 → navigation；前面是什么路口 → navigation；路边那个是什么店 → navigation |
| `changba` | 2/5 | 4/5 | 打开唱吧 → app_control（可接受）；唱吧点一首稻香 → KTV；关闭唱吧 → app_control（可接受） |
| `app_control` | 3/5 | 5/5 | 打开蓝牙 → intelligent_car_control（可接受）；关闭WiFi → intelligent_car_control（可接受） |
| `calculator` | 4/5 | 4/5 | 一英里等于多少公里 → chat |
| `app_agent` | 0/5 | 0/5 | 微信扫一扫给我打开 → app_call；打开支付宝付款码 → app_call；刷一下微博热搜 → app_call；打开淘宝搜一下充电宝 → app_call；美团上找一家奶茶店 → navigation |
| `wallpaper` | 3/5 | 4/5 | 换个壁纸 → intelligent_car_control（可接受）；壁纸换成猫咪 → intelligent_car_control |
| `other` | 1/5 | 3/5 | 再小一点 → navigation；那个呢 → unknown（可接受）；就这个吧 → phone_call；嗯嗯 → unknown（可接受） |
| `flight` | 5/5 | 5/5 | — |
| `painting` | 5/5 | 5/5 | — |
| `audio_book` | 1/5 | 1/5 | 播放三体有声书 → fm；我想听评书 → fm；播放郭德纲的相声 → fm；继续播放上次的有声书 → fm |
| `KTV` | 1/5 | 3/5 | 打开KTV → app_control（可接受）；我要唱K → changba；切歌 → music（可接受）；原唱伴唱切换 → changba |
| `train` | 4/5 | 4/5 | 北京到西安的动车 → unknown |
| `theme_gen` | 2/5 | 2/5 | 换个主题 → intelligent_car_control；切换成科技风主题 → intelligent_car_control；主题换成粉色 → intelligent_car_control |
| `traffic_restriction` | 4/5 | 4/5 | 外地车能进五环吗 → chat |
| `similar_poi` | 5/5 | 5/5 | — |

## 二、评测数据抽样

| 来源 | 句数 | unknown | 多意图 | 落域分布（前 6） |
|---|---|---|---|---|
| 快思考测试集·基础车控 | 90 | 6% | 1 | navigation 10，music_qa 9，chat 8，intelligent_car_info 7，phone_call 6，intelligent_car_control 6 |
| 01_记忆agent | 30 | 23% | 0 | chat 10，unknown 7，navigation 7，music 5，reminder 1 |
| 02_出行导航agent | 30 | 3% | 0 | navigation 11，view_poi_introduction 7，travel_plan 5，poi_recommend 2，outgoing_eta 1，similar_poi 1 |
| 03_音乐agent | 30 | 0% | 0 | music_qa 16，music 11，intelligent_car_control 1，chat 1，changba 1 |
| 05_用车agent | 30 | 10% | 0 | help 23，unknown 3，intelligent_car_info 3，intelligent_car_control 1 |
| 06_闲聊 | 30 | 3% | 0 | chat 26，unknown 1，help 1，travel_plan 1，news 1 |
| 07_有情知音 | 30 | 0% | 0 | chat 19，life_suggestion 9，news 1，music_qa 1 |
| 08_影视agent | 30 | 0% | 0 | video_qa 19，video 8，outside_qa 2，music_qa 1 |
| 09_新闻agent | 30 | 3% | 0 | news 18，chat 11，unknown 1 |
| 10_任务大师 | 30 | 0% | 0 | task_setting 27，phone_call 2，help 1 |
| 11_条件任务 | 30 | 17% | 2 | reminder 13，unknown 5，task_setting 4，navigation 2，reminder+navigation 2，music_qa 1 |
| 12_中枢复杂多任务 | 30 | 3% | 14 | chat 4，task_setting 3，navigation 2，reminder 2，intelligent_car_control+intelligent_car_control 2，unknown 1 |
| 13_中枢简单任务 | 30 | 0% | 25 | intelligent_car_control+intelligent_car_control 4，intelligent_car_control+intelligent_car_control+intelligent_car_control 3，navigation 3，intelligent_car_control+navigation 2，navigation+navigation+intelligent_car_control 2，intelligent_car_control+intelligent_car_control+navigation+intelligent_car_control 2 |
| 14_车控模糊意图 | 30 | 23% | 0 | intelligent_car_scenario 13，chat 9，unknown 7，outside_qa 1 |
| 16_基础语控 | 30 | 0% | 0 | intelligent_car_control 15，app_control 5，video 3，music 3，global_control 2，help 1 |
| 17_一语多意图 | 30 | 0% | 28 | intelligent_car_control+intelligent_car_control 16，intelligent_car_control+app_control 3，intelligent_car_control 2，app_control+weather_xf+stock 1，navigation+weather_xf+view_poi_recommend 1，intelligent_car_control+intelligent_car_control+navigation 1 |
| 18_同功能多参数 | 30 | 0% | 30 | intelligent_car_control+intelligent_car_control 27，intelligent_car_control+intelligent_car_control+intelligent_car_control 2，intelligent_car_control+intelligent_car_control+intelligent_car_control+intelligent_car_control 1 |
| 99_剔除清单 | 30 | 20% | 5 | chat 6，unknown 6，intelligent_car_scenario 4，news 4，intelligent_car_control+intelligent_car_control 1，intelligent_car_control+navigation 1 |

## 三、多轮（快思考测试集_多轮；上一轮先单独解析，结果作 previous_nlu_1，tts_1=好的）

| 上一轮标签 → 本轮标签 | 组数 | 本轮落域分布（前 5） |
|---|---|---|
| 基础车控 → 基础车控 | 16 | intelligent_car_control 3，music 2，weather_xf 2，other 2，navigation 2 |
| 音乐Agent → 音乐Agent | 14 | music 9，music_qa 2，other 2，changba 1 |
| 记忆 → 记忆 | 6 | chat 3，music_qa 2，other 1 |
| 基础车控 → 音乐Agent | 4 | music_qa 2，other 1，music 1 |
| 用车Agent → 基础车控 | 3 | other 1，reminder 1，phone_call 1 |
| 基础车控 → 导航Agent | 3 | outgoing_eta 2，navigation 1 |
| 基础车控 → 多意图 | 3 | intelligent_car_control 3 |
| 导航Agent → 基础车控 | 2 | intelligent_car_control 1，reminder 1 |
| 音乐Agent → 基础车控 | 2 | other 1，navigation 1 |
| 多意图 → 导航Agent | 2 | navigation 1，outgoing_eta 1 |
| 记忆 → 基础车控 | 2 | music 1，weather_xf 1 |
| 多意图 → 音乐Agent | 1 | music_qa 1 |
| 基础车控 → 用车Agent | 1 | help 1 |
| 多意图 → 基础车控 | 1 | weather_xf 1 |

多轮样例：
- 侧方位停车怎么停（help+help）→ **用微信给李四发消息** ⇒ `other+other`
- 打开座椅通风和按摩（intelligent_car_control+turn_on_device@name=座椅通风;intelligent_car_control+turn_on_device@name=座椅按摩）→ **梦里梦到醒不来的梦是哪首歌** ⇒ `music_qa+music_qa`
- 带我去福州的那个老街区，中间一条街最热闹，两边全是名人故居的地方。（navigation+navigation@destination=福州的老街区,rank=最热闹,poi_center=名人故居）→ **打开空调** ⇒ `intelligent_car_control+turn_on_device@name=空调`
- 音乐转换为随机播放（music+ctrl_shuffle）→ **切下一首** ⇒ `music+next`
- 换壁纸（intelligent_car_control+transform@name=壁纸）→ **雨夜从公司回家想听点安稳的歌** ⇒ `other+other`
- 我想就这样牵着你的手不放开是哪首歌（music_qa+music_qa）→ **只要女的唱的** ⇒ `music_qa+music_qa`
- 再找一些（unknown）→ **只要男人唱的** ⇒ `music+play@sex=男人`
- 抬头显示有重影怎么调（help+help）→ **设个明天7点的闹钟** ⇒ `reminder+reminder`
- 关闭电子狗（navigation+set_camera_warning@operation=关闭）→ **从国贸去望京SOHO开会，然后去首都机场接人。** ⇒ `navigation+navigation@poi_on_the_way=国贸,destination=望京SOHO,poi_on_the_way=首都机场`
- 还有别的吗（chat+chat）→ **第一首是他哪一年春晚唱的** ⇒ `other+other`
- 狂飙片尾那首歌叫什么（music_qa+music_qa）→ **用微信给李四发消息** ⇒ `other+other`
- 邓紫棋在我是歌手第一场唱红的歌是哪首（music_qa+music_qa）→ **换一批** ⇒ `music+play@singer=邓紫棋,song=我是歌手第一场`

## 四、域清单口径

- 清单 v13（docs/NLU3.0大模型_domain清单.md）有、v25 训练集没有：['app_agent', 'wechat_on_car']——这些域模型不会输出
- v25 训练集有、清单 v13 没有：['row_column_position', 'unknown']
- 输出里出现过、但不在 v25 训练集 45 个域里的：{'similar_words': 1}

## 五、结论与测试合理性说明（人工整理）

**测试口径的修正**：此前（`tools/probe_nlu3.py`）测的是 `/zp/update` 对话系统接口，外面包了对话管理、拒识和技能执行，模型分类要从 debug 里取，多意图会被拒识；本次按《接口使用说明 1030 固定版本》直接调 v25 模型（8048），才是 NLU3.0 本身的落域能力。前一份结果只能说明对话系统测试环境的情况，不代表模型。

1. **单轮落域**：45 域逐域 225 句严格命中 161 句（72%），含歧义句的可接受归属 177 句（79%）。去掉 v25 训练集里没有的 2 个域（`wechat_on_car`、`app_agent`，10 句），可接受口径约 82%（177/215）。21 个域 5 句全中。
   这组句子是我们按清单自造的、刻意覆盖边界说法，比官方测试集难；官方基线为单轮 97.68%、多轮 85.89%、多意图 77.61%。
2. **弱项**：`wechat_on_car`、`app_agent`（训练集里没有这两个域）；有声书 / 评书 / 相声判成 `fm`；换主题判成车控；视频播放控制（倍速、清晰度）判成 `global_control`；车外问答（旁边那栋楼是什么）判成 `navigation`；景点讲解有一半判成 `chat`。
3. **多意图**：能正常输出多块。一语多意图 30 句里 28 句、同功能多参数 30 句全部按多块输出，中枢简单任务 30 句里 25 句；同功能多参数也被拆成多块（如空调温度和风量拆成两个 `AC_control`）。
4. **多轮**：按文档把上一轮句子和它的解析结果放进 `previous_query_1` / `previous_nlu_1` 后，“把它关掉”“只要男人唱的”“切下一首”这类承接句能补全；部分承接句输出 `other+other`（训练集里的“承接上文”类）。测试集多轮的上一轮没有真实答复，`tts_1` 一律写“好的”，列表类承接（第二个、第二首）没测到。
5. **时延**：单次 p50 94 ms、p90 237 ms，远低于 /zp/update 的约 430 ms，对 800 ms 分流预算很友好。
6. **参数口径（项目约定）**：只用 `temperature=0, top_p=0.01`。同一批 225 句各跑 3 次：temperature=0 三次输出完全一致、落域正确 177/177/177；temperature=1 只有 171 句三次一致、落域正确 172–175。NLU3.0 要求相同输入必须得到相同输出，**不使用 temperature=1，也不解析置信度**；“接不住”仍按设计用结构信号（unknown / other、intent 为空、必填槽位缺失）和 Jev 的 domain 分布判断。
7. **域清单口径（已定 2026-10-31：以 v25 为准，见 docs/NLU3.0-v25_domain清单.md；v13 仅供参考）**：v25 训练集与清单 v13 不一致（缺 `wechat_on_car`、`app_agent`，多 `unknown`、`row_column_position`），我们判据里的 19 个入口域包含 `wechat_on_car`；输出里还出现过一次 `similar_words`。
8. **评测数据抽样的说明**：抽样只看落域分布，不判对错（业务标签是“该交给谁”，不是 NLU 域）；例如用车 agent 的句子 77% 落 `help`、新闻 agent 60% 落 `news`、任务大师 90% 落 `task_setting`，与 Agent 归属基本对得上，可作为回流表（非入口域 → Agent）的依据。
