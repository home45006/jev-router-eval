---
name: agentscope-cockpit
description: 用 AgentScope 2.0.9 开发智能座舱的 Agent 大脑、大脑下挂的能力和 Jev 前置分流。涉及 AgentScope 的 Agent、工具、子 Agent 编排、会话状态、结构化输出、流式事件、DeepSeek 模型（V4 Flash 非思考模式）、JevClassifierModel、Jev 纠错题，或要把第 14 版分流架构写成代码时使用。
---

# 座舱 Agent 开发（AgentScope 2.0.9）

本项目的 Agent 用阿里的 AgentScope 实现，版本固定 **2.0.9**（Python 3.11+）。架构见
`docs/jev-router-arch-v4.html`（第 14 版）：Jev 第一级只判一句话交给快思考还是 Agent 大脑，没有直达的独立 Agent；
Agent 大脑把快思考、任务大师、条件任务、联网搜索、用车手册、导航、空调、多媒体、美团、闲聊都当作工具自主编排，并持有多轮会话。
上线后某类句子落错了地方，给 Jev 第一级加一道纠错题来修，不改大脑、不重训快思考。

## 先读什么

| 要做的事 | 读 |
|---|---|
| 查某个类或方法怎么用 | `references/api_cheatsheet.md`（已在 2.0.9 上核对） |
| 把架构写成代码、会话怎么管 | `references/cockpit_architecture.md` |
| 大脑、子 Agent、会话三态的可运行示例 | `examples/brain_demo.py` |
| Jev 第一级和纠错题的可运行示例 | `examples/jev_router.py` |
| 多 Agent 的其他编排方式、服务化部署 | `references/upstream/`（官方 agentscope-skill 原文） |

## 必须遵守

1. **核对版本和签名再写代码。** 速查表以外的 API，用
   `python .claude/skills/agentscope-cockpit/scripts/view_module_signature.py --module agentscope.agent.Agent.reply`
   在装了 2.0.9 的环境里查，或者读 `pip download agentscope==2.0.9 --no-binary :all:` 下来的源码。
   官方文档站 docs.agentscope.io 在部分网络环境下打不开，不要凭记忆猜 1.x 的写法（1.x 的 `ReActAgent`、`formatter=` 参数在 2.x 里已经变了）。
2. **车机工具显式放行。** `FunctionTool` 不传 `permission` 时每次调用都会停下来等用户确认。
   普通车控、查询用 `PermissionDecision(behavior=PermissionBehavior.ALLOW, ...)`；支付、下单用 `ASK` 并在前端处理确认事件。
3. **每个会话独立的状态。** 用 `agent.state.model_dump_json()` 按 session_id 存，下一轮
   `AgentState.model_validate_json()` 恢复后重建 Agent。不要让多个会话共用一个 Agent 实例。
4. **大脑每轮返回会话状态。** `reply(..., structured_schema=TurnStatus)`，`status` 取 continue / suspend / end，
   应用代码据此决定下一句是否还交给大脑。超时、轮数上限、退出词在应用代码里兜底。
5. **Jev 每句只调一次。** is_fast、needs_rewrite 和纠错题都是 `BinaryQuestion`，放在同一个 `questions` 字典里一次请求。
   纠错题写在 `criteria_v2.json` 的 `corrections` 里：`status` 为 shadow 时只记日志，为 on 时命中就按 `route` 走，
   `hint` 附在用户原话后交给大脑。新题先 shadow，用 `eval_two_stage.py` 回放看改对、改错各多少，再改成 on。
6. **DeepSeek 用 V4 Flash 非思考模式。** `DeepSeekChatModel(..., parameters=DeepSeekChatModel.Parameters(thinking_enable=False))`，
   请求带 `thinking.type=disabled`，不产生推理 token。模型 ID 和地址从环境变量读：`DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL`（默认 `deepseek-v4-flash`）、
   `DEEPSEEK_FALLBACK_MODEL`（可选，经 `ModelConfig(fallback_model=...)` 生效）、`DEEPSEEK_BASE_URL`、`TYPESAFE_API_KEY`。
   2026-10-11 实测：`/models` 只列出 `deepseek-flash`、`deepseek-v4-pro`，但 `deepseek-v4-flash` 也能调通，服务端返回的模型名是 `deepseek-flash`（同一个模型）。
   非思考模式下推理 token 为 0，改写一句中位 0.97 秒、最慢 1.54 秒（GitHub 美国机房）。
   不要把 key 写进代码或提交到仓库；CI 里用仓库 Secret。
7. **控制时延。** 大脑 `ReActConfig(max_iters=6~10)`；用 `reply_stream()` 把 `TextBlockDeltaEvent.delta` 边生成边送 TTS。

## 最小骨架

```python
from agentscope.agent import Agent, ReActConfig
from agentscope.credential import DeepSeekCredential
from agentscope.message import UserMsg
from agentscope.model import DeepSeekChatModel
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import FunctionTool, Toolkit

ALLOW = PermissionDecision(behavior=PermissionBehavior.ALLOW, message="座舱工具，无需确认")

async def fast_think(query: str) -> str:
    """把说清楚的车机指令交给快思考模型执行。

    Args:
        query: 直接指令，例如“空调调到26度”。
    """
    ...

brain = Agent(
    name="agent_brain",
    system_prompt="你是智能座舱的 Agent 大脑……",
    model=DeepSeekChatModel(
        credential=DeepSeekCredential(api_key=os.environ["DEEPSEEK_API_KEY"],
                                      base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        parameters=DeepSeekChatModel.Parameters(thinking_enable=False),   # 非思考模式
    ),
    toolkit=Toolkit(tools=[FunctionTool(fast_think, permission=ALLOW), ...子 Agent 工具]),
    react_config=ReActConfig(max_iters=8),
)
res = await brain.reply(UserMsg(name="user", content=text), structured_schema=TurnStatus)
```

完整版（子 Agent 工具、会话存储、三态处理）见 `examples/brain_demo.py`。

## 验证

- 离线：`python examples/brain_demo.py --offline`、`python examples/jev_router.py --offline "导航到大裤衩"`。
  它们用 `scripts/scripted_model.py` 里的假模型，只证明代码接线和状态流转正确，不代表模型效果。
- 真实模型：设好环境变量后去掉 `--offline`。本仓库的云端开发环境访问不了 DeepSeek 和 TypeSafe，
  需要真实调用时放到 GitHub Actions 或本机运行。
- 交付前确认：导出的名字存在、构造参数和方法签名与 2.0.9 一致、工具权限已设置、会话状态能存取。
