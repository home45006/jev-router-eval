# 座舱前置分流演示系统

用来演示和验证两件事：**Jev 第一级能不能把句子分准**（快思考还是 Agent 大脑），以及 **Agent 大脑能不能判断出该调哪些能力**。
架构见 `docs/jev-router-arch-v4.html`（第 14 版），判据用仓库根目录的 `criteria_v2.json`。

## 运行

```bash
# 在仓库根目录
pip install -r webdemo/requirements.txt
export TYPESAFE_API_KEY=...        # Jev；不设则用模拟 Jev
export DEEPSEEK_API_KEY=...        # DeepSeek V4 Flash 非思考模式；不设则用模拟大模型
# 可选：DEEPSEEK_MODEL（默认 deepseek-v4-flash）、DEEPSEEK_BASE_URL、JEV_MODEL（默认 jev-latest）
# 可选：FAST_MODEL_URL，快思考服务地址，约定见下
export DASHSCOPE_API_KEY=...       # 可选：百炼联网搜索 MCP；设置后大脑判出“联网搜索”时会真实搜索
# 可选：SEARCH_MCP_URL（换别家的搜索 MCP）、SEARCH_MCP_TOOL（指定工具名）
uvicorn webdemo.app:app --port 8000
```

浏览器打开 http://127.0.0.1:8000 。页面右上角标明每个服务是真实还是模拟。
密钥只从环境变量读，不会写进页面或日志；服务默认只监听本机。

## 功能

| 页面 | 内容 |
|---|---|
| 单句与多轮 | 输入一句话，实时展示全链路：会话检查 → Jev 第一级（每个问题的概率和阈值）→ 分流决策的理由 → 大模型改写 → 快思考 → Agent 大脑（能力标签、计划、会话状态、播报）。每一步的耗时画成瀑布图，鼠标悬停看细节。同一会话里连续输入就是多轮：大脑返回 continue 时占用会话，下一句直接交给大脑；返回 suspend 时这句走正常分流、任务保留。内置两段多轮脚本。 |
| 标签模式 / 编排模式 | 标签模式：大脑一次调用输出能力标签，最快，适合验证判断准不准。编排模式：大脑像正式系统一样调用能力工具（多轮推理，能力用大模型模拟执行），能看到调用顺序、每次推理的耗时和 token。 |
| 批量验证 | 内置 32 句样例、评测集 dev 抽样（仓库根目录有 `eval_set.jsonl` 时出现）或自定义粘贴，并发跑完后给出去向准确率、大脑首选标签准确率、按标签的准确率和耗时分位数；点一行展开看这句的瀑布图。 |
| 耗时统计 | 本页面跑过的所有请求，按 Jev、改写、大脑单次推理、能力调用、端到端（分快思考和大脑两条路径）画分布图，给出 p50 / p90。 |
| 设置与判据 | 调 is_fast、needs_rewrite、纠错题的阈值；把每道纠错题切换为 off / shadow / on；模拟快思考拒识；编排模式的推理轮数上限。设置存在浏览器里。 |

## 联网搜索 MCP

`webdemo/search_mcp.py` 是一个 MCP 客户端，默认连接阿里云百炼的联网搜索 MCP（依次尝试 streamable HTTP 和 SSE 地址），
用 `Authorization: Bearer $DASHSCOPE_API_KEY` 鉴权，自动从工具列表里选名字带 search 的工具、按它的参数说明填搜索词。
大脑判出“联网搜索”后先调搜索 MCP，再让 DeepSeek 根据结果写播报；瀑布图里搜索单独一条，并拆分“建立连接”和“调用”两段耗时。
换成腾讯云、Tavily 等其他搜索 MCP 时只改 `SEARCH_MCP_URL` / `SEARCH_MCP_KEY`。探测：`python -m webdemo.search_mcp "测试问题"`。

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
