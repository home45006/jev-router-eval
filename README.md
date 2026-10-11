# jev-router-eval

验证 Jev（TypeSafe System One）能否完成车机语音的前置分流打标：输入一句 query，输出要走的标签。

- 语义 sheet 输出**预期domain**（16 个，首字母小写，如 carControl、naviControl），走 NLU 直达
- 大模型agent sheet 输出 **label**（12 个，如 用车agent、条件任务、任务大师），走 Agent 直达
- 闲聊会话里的追问一律跟随上一轮（代码规则，由 `is_followup` 触发）

分流架构见 [`docs/jev-router-arch-v4.html`](docs/jev-router-arch-v4.html)（第 14 版：Jev 只判是否交给快思考模型，其余全部交给 Agent 大脑；落域出错的 query 再加 Jev 纠错题分流）。第 3 版（28 个标签一次判完）保留在 [`docs/jev-router-flow.html`](docs/jev-router-flow.html)。

## 在云端跑（GitHub Actions）

1. 仓库 **Settings → Secrets and variables → Actions → New repository secret**，名称 `TYPESAFE_API_KEY`。
2. **Actions → jev-eval → Run workflow**：
   - `dataset=smoke`：用 `criteria.json` 里的 150 条例句，不需要 xlsx，先验证连通和请求结构。例句也在提示里，准确率会偏高；加 `extra_args=--no-examples` 看去掉例句后的表现。
   - `dataset=full`：先把测试集 xlsx 放进 `data/`，生成 13,662 条的完整评测集，按 `split`、`per_label` 抽样。
   也可以不进页面：把 `eval-request.env` 里的参数改好，推送到任意 `eval/*` 分支，同样会触发。
3. 结果在运行摘要里，`report.md`、`errors.csv`、`confusion.csv` 在 artifact 里。

每次提交还会自动跑一遍不调用 Jev 的流程检查（模拟结果 + SDK 请求结构校验）。

## 在本机跑

```bash
pip install typesafe-sdk openpyxl
export TYPESAFE_API_KEY=...
python eval_jev.py --data fixtures/smoke_set.jsonl --split dev --out runs/smoke     # 冒烟
python build_dataset.py data/测试集.xlsx --out .                                    # 生成完整评测集
python eval_jev.py --split dev --per-label 20 --out runs/dev-v0.4                    # 小样本
```

## 文件

| 文件 | 用途 |
|---|---|
| `criteria.json` | 给 Jev 的三个问题和 28 个选项的判据（由 `make_criteria.py` 生成，改判据请改脚本再重新生成） |
| `eval_jev.py` | 评测脚本：准确率、各标签召回/精确、混淆、置信度阈值、多指令和追问诊断 |
| `build_dataset.py` | 从测试集 xlsx 生成 `eval_set.jsonl` 和 `待确认改标.csv` |
| `eval_two_stage.py` | 前置分流评测。默认第 14 版：第一级 is_fast / needs_rewrite 加 Jev 纠错题，报告去向准确率和纠错题回放；`--stages s1,s2` 复现第 13 版两级分流。推送 eval/* 分支时在 `eval-request.env` 里设 `EVAL_SCRIPT=eval_two_stage.py` |
| `rewrite_check.py` | 用 DeepSeek 改写模糊指代，检查 `fixtures/fuzzy_refs.jsonl`；推送 eval/* 分支时在 `eval-request.env` 里设 `REWRITE_CHECK=1` |
| `make_criteria_v2.py` | 生成两级判据 `criteria_v2.json` |
| `fixtures/fuzzy_refs.jsonl` | 人工构造的 20 条模糊指代样本，用来测 needs_rewrite |
| `make_smoke_set.py` | 从判据例句生成 `fixtures/smoke_set.jsonl` |
| `EVAL_PACKAGE.md` | 评测包 v0.4 原说明（标签规则、切分方式） |

## 待确认

- “条件任务”下 128 行只有提醒、没有车机动作：归 scheduleReminder 还是留在条件任务。确认前两个标签都算对。
- 任务大师里约 4 行疑似一次性任务（如“15分钟后自动关闭视频”）。
- 所有阈值（置信度、`is_followup`）待在 val 上标定。

## AgentScope 开发 skill

`.claude/skills/agentscope-cockpit/` 是给 Claude Code 用的开发指南：用 AgentScope 2.0.9 实现 Agent 大脑、子 Agent、会话三态和 Jev 两级分流。
`examples/` 下的两个示例都能用 `--offline`（假模型）跑通，CI 每次提交都会跑；接真实模型时设置 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL`（默认 `deepseek-v4-flash`，非思考模式）、`TYPESAFE_API_KEY`。
