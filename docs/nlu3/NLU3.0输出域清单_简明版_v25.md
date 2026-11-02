# NLU 3.0 输出域清单（v25 简明版）

> 介绍：模型会输出哪些域、哪些域带独立意图和槽位、各自常见的意图名/槽位名。
> 数据口径：v25 定版训练数据 `nlu_train_v25_modi.jsonl`（2026-11-02 统计）。
> 接口：8048 / 8049（置信度版），输出格式 `domain+intent@key=value`，多意图分号分隔。

**输出域总共44类**，分两类：

- **A 类（12 个）**：域 + 独立意图 + 槽位，需要解析 intent 和 slots，见第一节
- **B 类（32个）**：域名直出（intent 与 domain 相同），按 domain 分发即可，见第二节

---

## 一、A 类：带独立意图和槽位的域（12个）

| domain                     | 中文   | intent 数 | 常见意图                                                                                                   | 常见槽位                                                  |
| -------------------------- | ---- | --------:| ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| `music`                    | 音乐   | 45       | play，play_album，add_play_list，play_mv，create_play_list 等 45 个                                          | theme，singer，app，episode，song 等 30 个                  |
| `intelligent_car_control`  | 车控   | 53       | turn_on_device，seat_position_control，turn_off_device，set，AC_control 等 53 个                             | position，gear_direction，area，name，gear_unit 等 31 个    |
| `phone_call`               | 电话   | 17       | make_a_phone_call_by_aiui，lookup_call_log，lookup_phone_num_by_aiui，turn_to_specific_page，select 等 17 个 | simcard，call_type，number，order，contact_name 等 10 个    |
| `navigation`               | 导航   | 50       | search_POI，switch_preference，add_POI，explore_way，query_distance_time 等 50 个                            | device，rank，app，custom_destination，destination 等 28 个 |
| `global_control`           | 全局控制 | 37       | navigate_home，play_by_located，multipe_speed_play，fast_forward，select 等 37 个                            | position，direction，order，type，number 等 10 个           |
| `fm`                       | 电台   | 22       | play，play_radio，play_favorite，add_favorite，next 等 22 个                                                 | category，anchor，source，radio_type，amfm 等 19 个         |
| `video`                    | 视频   | 9        | play_channel，play_actor_video，open，close，play_director_video 等 9 个                                     | app，channel，video_name，actor，director 等 8 个           |
| `app_control`              | 应用控制 | 3        | open_app，close_app，unknown                                                                             | device，app，position，name                              |
| `intelligent_car_info`     | 车况查询 | 13       | device_status_query，unknown，running_status_query，energy_trend_query，ac_state_query 等 13 个              | type，name，mode，position，rule 等 7 个                    |
| `home_agent`               | 车家互联 | 4        | AC_control，turn_off_device，turn_on_device，curtain_control                                              | room，name，mode，type，gear_number 等 13 个                |
| `intelligent_car_scenario` | 隐含车控 | 19       | air_permeability_scene，too_hot_scene，too_cold_scene，look_at_the_sky_scene，din_scene 等 19 个             | —                                                     |
| `KTV`                      | KTV  | 3        | play，open，close                                                                                        | app，singer，song                                       |

## 二、B 类：域名直出的域（32 个）

intent 与 domain 相同，没有独立意图层；按 `domain` 字段分发即可，无需解析 intent/slots。

| domain                  | 中文    |
| ----------------------- | ----- |
| `unknown`               | 拒识    |
| `video_qa`              | 影视问答  |
| `chat`                  | 闲聊    |
| `poi_recommend`         | POI推荐 |
| `app_call`              | 应用调用  |
| `music_qa`              | 音乐问答  |
| `short_video`           | 短视频   |
| `stock`                 | 股票    |
| `view_poi_recommend`    | 视野内推荐 |
| `outside_qa`            | 车外问答  |
| `travel_plan`           | 行程规划  |
| `task_setting`          | 场景任务  |
| `agent_task`            | 智能体代办 |
| `datetime_info`         | 日期时间  |
| `view_poi_introduction` | 景点讲解  |
| `news`                  | 新闻    |
| `life_suggestion`       | 生活建议  |
| `outgoing_eta`          | 出行耗时  |
| `changba`               | 唱吧    |
| `wallpaper`             | 壁纸    |
| `calculator`            | 计算    |
| `other`                 | 其他    |
| `help`                  | 用车专家  |
| `weather_xf`            | 天气查询  |
| `reminder`              | 提醒    |
| `train`                 | 火车    |
| `audio_book`            | 有声书   |
| `painting`              | 绘画    |
| `flight`                | 航班    |
| `similar_poi`           | 相似推荐  |
| `theme_gen`             | 主题生成  |
| `traffic_restriction`   | 限行    |

## 附注

- `unknown`：拒识，直接播报"没听清"类话术， `other`：不在上述域中的兜底落域
- 多意图：输出按 `;` 分块，每块独立一套 domain+intent+slot，块数即意图数
- 意图名为业务英文标识，与车机端 action 映射表配套使用
