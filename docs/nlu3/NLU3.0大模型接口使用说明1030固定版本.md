> 脱敏副本：服务地址、部署命令和服务器路径已替换为占位符；原件不入库，联调时向 NLU 团队获取地址，本地用环境变量 `NLU3_V25_URL` 配置。

# NLU 语义解析接口说明（v25 / 8048）

> 面向联调/模型集成。本文档里的所有输出示例，均为 **8048 端口 v25 模型实测返回**（2026-10-30 实跑）。

---

## 一、端口与版本

| 项            | 值                                                                                                                                                                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **请求 URL**   | `http://<NLU3-v25 服务地址>/v1/chat/completions`                                                                                                                                                                                        |
| 服务类型         | lmdeploy（turbomind）OpenAI 兼容接口                                                                                                                                                                                                          |
| **model 名称** | `qwen3_nlu`                                                                                                                                                                                                                             |
| **模型版本**     | `output/nlu_v25modi_06b_full/` —— Qwen3-0.6B **全量微调**（2026-10-29 训练，3 epoch，train_loss 0.1132）                                                                                                                                          |
| **训练数据版本**   | `data/nlu_train_v25_modi.jsonl`（**101,513 条**，45 个 domain）**输出为纯 `domain+intent@slot` 格式，不带任何尾部标记**                                                                                                                                     |
| 推荐推理参数       | `temperature=0`、`top_p=0.01`、`max_tokens=128`                                                                                                                                                                                           |
| 部署命令         | （已略去：部署命令和服务器路径见 NLU 团队内部文档）|
| 当前服务         | **8048 = v25（联调唯一入口）**；；8046/8047 = 旧协议链路；7860 = NLU大模型web测试网页                                                                                                                                                                          |

**评测基线**（标准版全 domain 测试集，2026-10-30 实测）：单轮 23,849 条 **97.68%**；多轮 4,910 轮 **85.89%**；多意图 728 句 **77.61%**

**system content（固定一句，不要改）**

```
你是一个车载语音助手指令解析引擎，专精于将自然语言指令转化为结构化语义
```

**user content 格式**：一条扁平字符串，英文半角逗号 `,` 分隔字段，字段名后**没有空格**。

```
query:<当前句>,previous_query_1:<上一轮句>,previous_nlu_1:<上一轮标签>,tts_1:<上一轮助手答复>
```

| 字段                  | 含义                                  |
| ------------------- | ----------------------------------- |
| `query:`            | 本轮用户句（必填）                           |
| `previous_query_N:` | 第 N 轮历史用户句；**编号 1 = 最旧，编号越大越接近当前句** |
| `previous_nlu_N:`   | 第 N 轮的语义标签，格式与本接口输出完全一致             |
| `tts_N:`            | 第 N 轮助手语音答复，没有就写 `好的`               |

> 单轮也必须保留三个空的历史字段：`previous_query_1:,previous_nlu_1:,tts_1:`（训练数据即如此，省略会掉分布外）。

---

## 二、完整 curl 请求示例

### 2.1 单轮

```bash
curl -sS http://<NLU3-v25 服务地址>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_nlu","messages":[
        {"role":"system","content":"你是一个车载语音助手指令解析引擎，专精于将自然语言指令转化为结构化语义"},
        {"role":"user","content":"query:把副驾座椅通风调到三档,previous_query_1:,previous_nlu_1:,tts_1:"}
      ],"temperature":0,"top_p":0.01,"max_tokens":128,"stop":["<|im_end|>"]}'
```

返回：

```
intelligent_car_control+seat_ventilation_control@area=座椅,gear_number=3,gear_unit=档,position=副驾
```

### 2.2 多轮（带 1 轮历史）

```bash
curl -sS http://<NLU3-v25 服务地址>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_nlu","messages":[
        {"role":"system","content":"你是一个车载语音助手指令解析引擎，专精于将自然语言指令转化为结构化语义"},
        {"role":"user","content":"query:把它关掉,previous_query_1:打开HUD,previous_nlu_1:intelligent_car_control+turn_on_device@name=ARHUD,tts_1:好的"}
      ],"temperature":0,"top_p":0.01,"max_tokens":128,"stop":["<|im_end|>"]}'
```

返回：

```
intelligent_car_control+turn_off_device@name=ARHUD
```

### 2.3 多意图

请求格式与单轮完全一致，模型自行输出多块：

```bash
curl -sS http://<NLU3-v25 服务地址>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_nlu","messages":[
        {"role":"system","content":"你是一个车载语音助手指令解析引擎，专精于将自然语言指令转化为结构化语义"},
        {"role":"user","content":"query:打开车窗并播放音乐,previous_query_1:,previous_nlu_1:,tts_1:"}
      ],"temperature":0,"top_p":0.01,"max_tokens":128,"stop":["<|im_end|>"]}'
```

返回：

```
intelligent_car_control+turn_on_device@name=车窗;music+play
```

### 2.4 unknown（拒识）

```bash
curl -sS http://<NLU3-v25 服务地址>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_nlu","messages":[
        {"role":"system","content":"你是一个车载语音助手指令解析引擎，专精于将自然语言指令转化为结构化语义"},
        {"role":"user","content":"query:阿斯顿发斯蒂芬阿斯顿,previous_query_1:,previous_nlu_1:,tts_1:"}
      ],"temperature":0,"top_p":0.01,"max_tokens":128,"stop":["<|im_end|>"]}'
```

返回：

```
unknown
```

### 2.5 带 tts 候选列表的继承句

当上一轮助手答复（`tts_1`）里带**候选列表**时，用户后续说「第X个 / 第X首」，把序数**解析成列表里的具体条目**写入槽位。`tts_1` 原样传入即可，**列表建议带序号前缀（`1.xxx 2.xxx 3.xxx`），该格式解析最稳**；纯逗号分隔列表也支持。

```bash
curl -sS http://<NLU3-v25 服务地址>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_nlu","messages":[
        {"role":"system","content":"你是一个车载语音助手指令解析引擎，专精于将自然语言指令转化为结构化语义"},
        {"role":"user","content":"query:给我播放第二首,previous_query_1:周深的成名曲有哪些,previous_nlu_1:music_qa+music_qa,tts_1:1.大鱼 2.光亮 3.花开忘忧"}
      ],"temperature":0,"top_p":0.01,"max_tokens":128,"stop":["<|im_end|>"]}'
```

返回（「第二首」被解析成列表第 2 项）：

```
music+play@singer=周深,song=光亮
```

行为规则：

| 上一轮 tts_1 内容                 | 本轮「第X个/首」的输出行为                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| 带候选列表（推荐 `1.a 2.b 3.c` 序号格式） | **解析成具体条目**，如 `music+play@singer=周深,song=光亮`、`navigation+navigation@destination=四季民福烤鸭店` |
| 无列表（如只有 `好的`）                | 序数直通，如 `navigation+select@number=2`                                                      |

---

## 三、输出格式与解析规则

### 3.1 语法

```
<domain>+<intent>[@<slot>]
```

### 3.2 分隔符总表

| 层级              | 分隔符  | 含义                 | 示例片段                                                   |
| --------------- | ---- | ------------------ | ------------------------------------------------------ |
| 意图块             | `;`  | 多意图时分隔每个块          | `music+play@song=心墙;music+ctrl_single_cycle`           |
| domain / intent | `+`  | 块内第一个 `+` 分割       | `intelligent_car_control` + `seat_ventilation_control` |
| slot 区起点        | `@`  | `@` 之后全是槽位，无槽时整体省略 | `@area=座椅,gear_number=3`                               |
| 槽位项             | `,`  | 槽位之间               | `area=座椅,gear_number=3,gear_unit=档`                    |
| key / value     | `=`  | 槽位内部               | `gear_unit=档`                                          |
| value 多值        | `\|` | 同一个 key 有多个值       | `我想听周杰伦和蔡依林合作的歌 =》singer=周杰伦\|蔡依林`                     |

### 3.3 参考解析代码

```python
def parse_nlu(text):
    """'d+i@k=v,k2=v2;d2+i2' -> [{'domain','intent','slots'}, ...]"""
    blocks = []
    for b in (text or "").split(";"):
        b = b.strip()
        if not b:
            continue
        body, _, slot = b.partition("@")
        domain, _, intent = body.partition("+")
        slots = {}
        for item in slot.split(","):
            item = item.strip()
            if item and "=" in item:
                k, v = item.split("=", 1)
                slots[k.strip()] = v.strip()
        blocks.append({"domain": domain.strip(), "intent": intent.strip(), "slots": slots})
    return blocks
```

### 3.4 输出类型

| 类型       | 输出形态                                                                                             |
| -------- | ------------------------------------------------------------------------------------------------ |
| 无槽       | `chat+chat`                                                                                      |
| 多槽       | `intelligent_car_control+seat_ventilation_control@area=座椅,gear_number=3,gear_unit=档,position=副驾` |
| 同 key 多值 | `music+play@singer=林俊杰\|周杰伦`                                                                     |
| 多意图      | `music+play@song=心墙,theme=dj版;music+ctrl_single_cycle`                                           |
| 拒识 / 无效句 | `unknown`                                                                                        |

### 3.5 postman测试示例

`//测试接口：http://<NLU3-v25 服务地址>/v1/chat/completions`  

`{`

  `"model": "qwen3_nlu",`

  `"messages": [`

    `{"role": "system", "content": "你是一个车载语音助手指令解析引擎，专精于将自然语言指令转化为结构化语义"},`

    `// //  多轮测试输入`

   `{"role":"user","content":"query:给我播放第二首,previous_query_1:周杰伦的成名曲有哪些,previous_nlu_1:intelligent_car_control+turn_on_device@name=ARHUD,tts_1:他的成名曲有七里香，夜曲，以父之名"}`

   `//// 多意图测试输入`

  `// {"role":"user","content":"query:打开车窗并播放音乐,previous_query_1:,previous_nlu_1:,tts_1:"}`

  `],`

  `"temperature": 0.01,`

  `"top_p": 0.95,`

  `"max_tokens": 128,`

  `"top_k": 20,`

  `"stream": false`

`}`



输出示例
{
"id": "chatcmpl-bpwkhdUaEEcvaE36BRbXf3",

    "object": "chat.completion",

    "created": 1793327065,

    "model": "qwen3_nlu",

    "choices": [

        {

            "index": 0,

            "message": {

                "role": "assistant",

                "content": "music+play@singer=周杰伦,song=夜曲",

                "reasoning_content": null,

                "tool_calls": null

            },

            "logprobs": null,

            "output_token_logprobs": null,

            "finish_reason": "stop",

            "output_ids": null,

            "routed_experts": null
],

    "usage": {

        "prompt_tokens": 93,

        "total_tokens": 107,

        "completion_tokens": 14,

        "prompt_tokens_details": {

            "cached_tokens": 0

        },

        "completion_tokens_details": null

    },

    "service_tier": null

}


      
}



## 四、联调测试用例与实测输出（8048 / v25 真实返回，2026-10-30）

### 4.1 单轮

| #   | query       | 实测输出                                                                                             |
| --- | ----------- | ------------------------------------------------------------------------------------------------ |
| 1   | 导航最小化       | `navigation+zoom_map@zoom_type=最小`                                                               |
| 2   | 把副驾座椅通风调到三档 | `intelligent_car_control+seat_ventilation_control@area=座椅,gear_number=3,gear_unit=档,position=副驾` |
| 3   | 打电话给张三      | `phone_call+make_a_phone_call_by_aiui@contact_name=张三`                                           |
| 4   | 我想听周杰伦的晴天   | `music+play@singer=周杰伦,song=晴天`                                                                  |
| 5   | 今天天气怎么样     | `weather_xf+weather_xf`                                                                          |

### 4.2 多轮

拼装方式：把「上一轮 query」放进 `previous_query_1`、「上一轮 nlu」放进 `previous_nlu_1`、`tts_1` 写 `好的`，本轮句放进 `query:`。

| #   | 上一轮 query  | previous_nlu_1                                                  | 本轮 query | 实测输出（v25）                                                                                    |
| --- | ---------- | --------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------- |
| 1   | 打开HUD      | `intelligent_car_control+turn_on_device@name=ARHUD`             | 把它关掉     | `intelligent_car_control+turn_off_device@name=ARHUD`                                         |
| 2   | 我想听周杰伦的歌   | `music+play@singer=周杰伦`                                         | 换一首      | `global_control+next`                                                                        |
| 3   | 帮我找附近的加油站  | `navigation+search_POI@rank=附近,poi=加油站`                         | 导航去第一个   | `navigation+select@number=1`                                                                 |
| 4   | 把主驾座椅加热打开  | `intelligent_car_control+turn_on_device@name=座椅加热,position=主驾`  | 调到二档     | `intelligent_car_control+seat_heating_control@area=座椅,gear_number=2,gear_unit=档,position=主驾` |
| 5   | 周深的成名曲有哪些  | `music_qa+music_qa`（tts_1=`1.大鱼 2.光亮 3.花开忘忧`）                   | 给我播放第二首  | `music+play@singer=周深,song=光亮`                                                               |
| 6   | 推荐附近好吃的烤鸭店 | `poi_recommend+poi_recommend`（tts_1=`1.大董烤鸭 2.四季民福烤鸭店 3.便宜坊烤鸭`） | 导航去第二个   | `navigation+navigation@destination=四季民福烤鸭店`                                                  |

### 4.3 多意图

| #   | query             | 实测输出（v25）                                                                                                                      |
| --- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1   | 给我放一首心墙dj版，单曲循环一下 | `music+play@song=心墙,theme=dj版;music+ctrl_single_cycle`                                                                         |
| 2   | 打开车窗并播放音乐         | `intelligent_car_control+turn_on_device@name=车窗;music+play`                                                                    |
| 3   | 导航到天安门顺便放首歌       | `navigation+navigation@destination=天安门;music+play@scene=主驾`                                                                    |
| 4   | 把空调调到24度再把座椅通风打开  | `intelligent_car_control+AC_control@temperature_number=24,temperature_unit=度;intelligent_car_control+turn_on_device@name=座椅通风` |

### 4.4 unknown

| #   | query            | 实测输出      |
| --- | ---------------- | --------- |
| 1   | 阿斯顿发斯蒂芬阿斯顿       | `unknown` |
| 2   | 呃呃呃额             | `unknown` |
| 3   | 噜啦啦噜啦啦           | `unknown` |
| 4   | asdkjhaksdjahsdk | `unknown` |

---

## 五、易踩的坑（8048 实测）

| 问题                                                   | 说明                                                                                               |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| ❌ `chat_template_kwargs: {"enable_thinking": false}` | lmdeploy turbomind **不支持**，会破坏 Qwen3 chat template 渲染，输出变成自然语言。不要传（该参数只适用于 vLLM / LLaMA-Factory） |
| ❌ 用 8046/8047 的 `\t` 分段格式                            | 那是旧协议，8048 只认逗号扁平串                                                                               |
| ⚠️ 单轮时不写历史字段                                         | 必须保留空的 `previous_query_1:,previous_nlu_1:,tts_1:`，省略即分布外                                         |
| ⚠️ 字段名后加空格                                           | 写成 `previous_query_1: xxx` 与训练数据不一致，别加                                                           |
| ⚠️ `previous_nlu_N` 用别的口径                            | 必须与本接口输出同格式（`domain+intent@slot`），不要带 `→` 标记、不要用 `&` 或 `:` 分隔槽位                                  |
| ⚠️ 助手答复放进 `previous_query_N`                         | 助手的话属于 `tts_N`，历史用户句才放 `previous_query_N`                                                        |
| ⚠️ JSON 里写注释                                         | 会导致请求直接 400                                                                                      |
| ⚪ `temperature` / `top_p`                            | 建议固定 `temperature=0, top_p=0.01`                                                                 |
| ⚪ `max_tokens`                                       | 标签很短，128 足够                                                                                      |

---

## 六、支持的 45 个 domain

完整清单（中文说明、intent 数、例句）见 **`docs/NLU3.0大模型_domain清单.md`**。
下表为训练集 `nlu_train_v25_modi.jsonl`（101,513 条）中 45 个 domain 及样本数（2026-10-30 统计）：

| #   | domain                    | 条数    | #   | domain                     | 条数  |
| --- | ------------------------- | -----:| --- | -------------------------- | ---:|
| 1   | `music`                   | 21836 | 24  | `travel_plan`              | 253 |
| 2   | `intelligent_car_control` | 16593 | 25  | `agent_task`               | 245 |
| 3   | `phone_call`              | 16095 | 26  | `task_setting`             | 231 |
| 4   | `navigation`              | 9370  | 27  | `datetime_info`            | 185 |
| 5   | `global_control`          | 6733  | 28  | `stock`                    | 171 |
| 6   | `fm`                      | 5976  | 29  | `view_poi_introduction`    | 151 |
| 7   | `unknown`                 | 4968  | 30  | `news`                     | 150 |
| 8   | `video_qa`                | 3921  | 31  | `intelligent_car_scenario` | 139 |
| 9   | `chat`                    | 2650  | 32  | `life_suggestion`          | 129 |
| 10  | `video`                   | 2069  | 33  | `outgoing_eta`             | 124 |
| 11  | `other`                   | 1540  | 34  | `changba`                  | 114 |
| 12  | `poi_recommend`           | 1154  | 35  | `wallpaper`                | 68  |
| 13  | `app_control`             | 1152  | 36  | `calculator`               | 63  |
| 14  | `app_call`                | 1067  | 37  | `train`                    | 58  |
| 15  | `intelligent_car_info`    | 946   | 38  | `audio_book`               | 51  |
| 16  | `help`                    | 888   | 39  | `painting`                 | 49  |
| 17  | `weather_xf`              | 443   | 40  | `KTV`                      | 45  |
| 18  | `music_qa`                | 341   | 41  | `similar_poi`              | 42  |
| 19  | `short_video`             | 306   | 42  | `theme_gen`                | 42  |
| 20  | `reminder`                | 283   | 43  | `flight`                   | 36  |
| 21  | `view_poi_recommend`      | 272   | 44  | `traffic_restriction`      | 33  |
| 22  | `outside_qa`              | 272   | 45  | `row_column_position`      | 3   |
| 23  | `home_agent`              | 256   |     |                            |     |
