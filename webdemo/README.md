# 座舱前置分流演示系统

用来演示和验证两件事：**Jev 第一级能不能把句子分准**（快思考还是 Agent 大脑），以及 **Agent 大脑能不能判断出该调哪些能力**。
架构见 `docs/jev-router-arch-v4.html`（第 14 版），判据用仓库根目录的 `criteria_v2.json`。

## 运行

```bash
# 在仓库根目录
pip install -r webdemo/requirements.txt
export TYPESAFE_API_KEY=...        # Jev；不设则用模拟 Jev
export DEEPSEEK_API_KEY=...        # DeepSeek V4 Flash 非思考模式；不设则用模拟大模型
# 可选：DEEPSEEK_MODEL（默认 deepseek-flash，官方文档指定的模型名）、DEEPSEEK_BASE_URL、JEV_MODEL（默认 jev-latest）
# 可选：FAST_MODEL_URL，快思考服务地址，约定见下
# 联网搜索用 DeepSeek 自带的（Anthropic 兼容接口，复用 DEEPSEEK_API_KEY）；可选 DEEPSEEK_SEARCH_MAX_USES（默认 2）
uvicorn webdemo.app:app --port 8000
```

浏览器打开 http://127.0.0.1:8000 。

v5 分流测试页：http://127.0.0.1:8000/v5 （按《Jev 分流详细设计 v2》：Jev 一次判定 7 题 ∥ NLU3.0 → 分发策略 R0–R7；判据读 `docs/share/criteria_v3.json`；NLU3.0 未接入时用模拟，设 `NLU3_URL` 接真实服务；`/v5?q=句子` 打开即发送）。页头可切换决策后端 Jev / StartLux（自部署，默认 `http://81.70.90.78:8090`，用 `STARTLUX_URL` 覆盖，`STARTLUX_API_KEY` 可选），单轮和批量可勾选两边对比（`/v5?cmp=1&q=句子`）。“提示词”标签页可在页面上改 7 道题的问法、判据和选项顺序，应用后对后续请求生效（只存在本机浏览器，不改判据文件）。页面右上角标明每个服务是真实还是模拟。
密钥只从环境变量读，不会写进页面或日志；服务默认只监听本机。

## 功能

| 页面 | 内容 |
|---|---|
| 单句与多轮 | 输入一句话，实时展示全链路：会话检查 → Jev 第一级（每个问题的概率和阈值）→ 分流决策的理由 → 大模型改写 → 快思考 → Agent 大脑（能力标签、计划、会话状态、播报）。每一步的耗时画成瀑布图，鼠标悬停看细节。同一会话里连续输入就是多轮：大脑返回 continue 时占用会话，下一句直接交给大脑；返回 suspend 时这句走正常分流、任务保留。内置两段多轮脚本。 |
| 标签模式 / 编排模式 | 标签模式：大脑一次调用输出能力标签，最快，适合验证判断准不准。编排模式：大脑像正式系统一样调用能力工具（多轮推理，能力用大模型模拟执行），能看到调用顺序、每次推理的耗时和 token。 |
| 批量验证 | 内置 32 句样例、评测集 dev 抽样（仓库根目录有 `eval_set.jsonl` 时出现）或自定义粘贴，并发跑完后给出去向准确率、大脑首选标签准确率、按标签的准确率和耗时分位数；点一行展开看这句的瀑布图。 |
| 耗时与费用 | 本页面跑过的所有请求，按 Jev、改写、大脑单次推理、能力调用、联网搜索、端到端（分快思考和大脑两条路径）画耗时分布图，给出 p50 / p90；下面是 DeepSeek 费用表（每类调用的次数、平均、p90、合计，以及每句的平均费用）。 |
| 流式与首 token | 大模型调用默认流式（设置页可关），记录首 token 耗时：瀑布图里浅色段是等首 token、深色段是流式生成，竖线标首 token；联网搜索另用虚线标出搜索结果到达的时刻，答案边生成边显示在播报框里。耗时页和批量验证给出首 token / 答案首字的 p50、p90。 |
| 大模型费用 | 每次 DeepSeek 调用都按官方价格算钱：输入分命中缓存 / 未命中、输出分别计价，按北京时间区分高峰 / 空闲时段（`webdemo/pricing.py`，价格变动时改那里的表；`DEEPSEEK_PRICE_PERIOD=peak` 可固定按高峰算）。瀑布图每条调用后面标出费用，详情里有逐次明细，批量验证给出合计和每千句费用。 |
| 设置与判据 | 调 is_fast、needs_rewrite、纠错题的阈值；把每道纠错题切换为 off / shadow / on；模拟快思考拒识；编排模式的推理轮数上限。设置存在浏览器里。 |

## 联网搜索

`webdemo/search_deepseek.py` 调用 DeepSeek 自带的联网搜索：走 Anthropic 兼容接口 `/anthropic/v1/messages`，
带服务端执行的 `web_search` 工具、关闭思考模式，一次请求里模型自己决定搜什么、搜几次，并写好播报（OpenAI 兼容接口没有这个能力）。
大脑判出“联网搜索”后调用它；瀑布图里单独一条，详情里有模型用的搜索词、来源、token 用量和答案。
流式接收，记录首个事件、搜索结果到达和答案首字三个时间点。探测：`python -m webdemo.search_deepseek "测试问题"`。实测（GitHub 美国机房）一次 2.5–3 秒。

## 快思考服务约定（FAST_MODEL_URL）

`POST {"query": "...", "previous_turn": {...} | null}`，返回
`{"domain": "...", "intent": "...", "slots": {...}, "confidence": 0.93, "rejected": false}`。
`rejected` 为真或 `confidence` 低于 0.5 时转交 Agent 大脑。不设置时页面显示“快思考未接入”，交给快思考的句子只记录、不执行。

## 和正式系统的差别

- 大脑直接用 OpenAI 兼容接口调用 DeepSeek（`extra_body={"thinking": {"type": "disabled"}}`），没有经过 AgentScope，
  这样每次调用的耗时和 token 都能单独记录。正式实现按 `.claude/skills/agentscope-cockpit` 用 AgentScope，提示词和能力清单可以直接沿用 `pipeline.py`。
- 能力（导航、美团等）没有接真实服务，编排模式里由大模型模拟执行结果，只用来看大脑的调度是否合理和耗时量级。
- 会话保存在服务进程内存里，重启即清空。
- 模拟模式的结果由关键词规则产生，只用来看界面和流程，不代表模型效果。

## 检查

- `python -m webdemo.check_mock`：不需要密钥，跑通全部接口（CI 每次提交都跑）。
- `python -m webdemo.check_real`：需要两个密钥，跑单句、多轮和内置样例，输出 `runs/webdemo/report.md`。
  在 GitHub Actions 上运行：推送 eval/* 分支，`eval-request.env` 里设 `WEBDEMO_CHECK=1`、`SKIP_EVAL=1`。
