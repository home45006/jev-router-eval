# Jev 标签评测包（v0.4）

把 C518-1 测试集里的 query 交给 Jev 打标签，并量出准确率。标签共 28 个：

- “语义”sheet：输出**预期domain**，16 个，一律用首字母小写的写法（carControl、naviControl、phone）
- “大模型agent”sheet：输出 **label**，12 个

## 文件

| 文件 | 用途 |
|---|---|
| `criteria.json` | 给 Jev 的三个问题和 28 个选项的判据。调优主要改这个文件 |
| `eval_set.jsonl` | 评测集，13,662 条，已去重并切成 dev / val / test |
| `eval_jev.py` | 评测脚本，输出准确率、混淆、阈值分析 |
| `待确认改标.csv` | 标注与规则冲突、需要人工确认的 216 行 |
| `build_dataset.py` | 从 xlsx 重新生成评测集和待确认清单 |

## 怎么跑

```bash
pip install typesafe-sdk
export TYPESAFE_API_KEY=你的密钥

# 1. 小样本：每个标签最多 20 条，先看方向对不对
python eval_jev.py --split dev --per-label 20 --out runs/dev-v0.3

# 2. 看 runs/dev-v0.3/report.md 和 errors.csv，改 criteria.json，换个 --out 再跑
# 3. 在 val 上定阈值，最后在 test 上出结果（test 不要用来改判据）
python eval_jev.py --split val  --out runs/val-v0.3
python eval_jev.py --split test --out runs/test-v0.3
```

没有密钥时加 `--mock` 可以验证流程，结果是假的。

## 标签和规则

- **domain 写法**：一律首字母小写。原表里 CarControl / carControl 这类大小写不一致的写法按此合并；只有大写写法的 NaviControl、MediaControl、Navi 也改成 naviControl、mediaControl、navi。
- **类别列**不再是标签，只作为 `categories` 字段保留；“全局”保留为 `is_global` 标记。
- **带时间或触发条件的请求分三类**：
  - 提醒用户做某事 → scheduleReminder，传统 NLU 理解，速度快
  - 一次性任务，车机直接执行 → 条件任务，大模型 Agent 理解
  - 周期性任务，反复触发 → 任务大师，大模型 Agent 理解
  原标注里“条件任务”下有 128 行只有提醒、没有车机动作，确认改标之前这些行 scheduleReminder 和条件任务都算对。
- **闲聊追问跟随上一轮**：由代码实现。上一轮标签是闲聊，且 Jev 判断这句是追问，就输出闲聊。

## 切分方式

同一标签内按句式模板分组，同组的句子只会出现在 dev、val、test 其中一份里，避免同一模板换个座位后两边都有。
判据里的例句只取自 dev。样本很少的 domain（CarInfo、unicast、memoryRecord 等）在 dev 里只有 0–3 条。
