# AgentScope 2.0.9 API 速查

以下内容都在 `agentscope==2.0.9` 上核对过（源码和实际运行）。版本升级后先用
`scripts/view_module_signature.py` 复查签名。

## 安装

```bash
pip install 'agentscope==2.0.9'                    # Python 3.11+
pip install 'agentscope[classifier-jev]==2.0.9'    # 用 JevClassifierModel 时
```

## 模型与凭证

| 用途 | 类 | 要点 |
|---|---|---|
| DeepSeek | `DeepSeekChatModel(credential=DeepSeekCredential(api_key=..., base_url="https://api.deepseek.com"), model="<模型 ID>")` | 默认 `stream=True`、`max_retries=3`、`context_size=65536`；`client_kwargs` 透传给 `openai.AsyncClient`（如 `timeout`）。非思考模式：`parameters=DeepSeekChatModel.Parameters(thinking_enable=False)`（默认就是 False，请求带 `thinking.type=disabled`；显式写出以免误开） |
| OpenAI 兼容接口 | `OpenAIChatModel(credential=OpenAICredential(api_key=..., base_url=...), model=...)` | 走公司代理或其他兼容服务时用 |
| 通义 | `DashScopeChatModel(credential=DashScopeCredential(api_key=...), model=...)` | 官方示例默认用它 |
| Jev 分类 | `JevClassifierModel(credential=TypeSafeCredential(api_key=...), model="jev-latest", timeout=30.0, max_retries=2)` | 不是聊天模型，`TypeSafeCredential` 只能用于分类器 |

备用模型：`Agent(..., model_config=ModelConfig(max_retries=1, fallback_model=另一个模型))`。

## Agent

```python
Agent(name, system_prompt, model, toolkit=None, middlewares=None, state=None,
      offloader=None, model_config=None, context_config=None, react_config=None,
      injection_config=None)
```

- `await agent.reply(inputs, structured_schema=None) -> Msg`：消费完所有事件，返回最终消息。
- `agent.reply_stream(inputs, structured_schema=None, yield_final_msg=False)`：逐个产出事件，适合边生成边播报。
- `inputs` 可以是 `Msg`、`list[Msg]`、`UserConfirmResultEvent`、`ExternalExecutionResultEvent`、`UserInterruptEvent` 或 `None`。
- 结果判断：`res.finished_reason == ReplyFinishedReason.COMPLETED`（`from agentscope.types import ReplyFinishedReason`），文本用 `res.get_text_content()`。
- `ReActConfig(max_iters=50, ...)`：推理-行动的最大轮数。车上要控制时延，大脑建议 6–10。

## 结构化输出

```python
res = await agent.reply(UserMsg(name="user", content=text), structured_schema=MyPydanticModel)
res.structured_output   # dict，已按 schema 校验
```

内部会给模型加一个名为 **`GenerateStructuredOutput`** 的工具，模型调用它即结束本轮。
写假模型脚本时要用这个名字。

## 消息

`UserMsg(name=..., content="文本" 或 [TextBlock(...), DataBlock(...)])`、`AssistantMsg`、`SystemMsg`。
内容块：`TextBlock`、`ThinkingBlock`、`ToolCallBlock(id, name, input=JSON 字符串)`、`ToolResultBlock`、`DataBlock(source=URLSource(url, media_type) | Base64Source(...))`。

## 工具

```python
FunctionTool(func, name=None, description=None, input_schema=None,
             is_concurrency_safe=..., is_read_only=..., is_state_injected=...,
             middlewares=None, permission=None)
Toolkit(tools=[...])   # 也接受 MCP 客户端和 skill 路径
```

- 参数 schema 从类型注解和 docstring（Google 风格 `Args:`）生成；枚举和范围用 `Literal`、`Annotated[..., Field(...)]`。
- **`permission` 不传时每次调用都要用户确认。** 车机工具传
  `PermissionDecision(behavior=PermissionBehavior.ALLOW, message="...")`（`from agentscope.permission import ...`）。
  支付、下单这类动作用 `PermissionBehavior.ASK`，前端处理 `RequireUserConfirmEvent` 后用 `UserConfirmResultEvent` 续上。
- 函数返回字符串即可，`FunctionTool` 会转成工具结果。不要把 `Msg` 或事件当工具结果返回。

## 状态与会话

- `agent.state` 是 `AgentState`（pydantic），字段有 `session_id`、`summary`、`context`、`reply_context`、`permission_context`、`tool_context`、`tasks_context`、`middle_context`。
- 持久化：`state.model_dump_json()`；恢复：`AgentState.model_validate_json(raw)`，再 `Agent(..., state=恢复的 state)`。
- **不要在并发的不同会话之间共享同一个 Agent 实例**；按会话存 state，每轮重建 Agent。

## 流式事件（`agentscope.event`）

| 事件 | 常用字段 |
|---|---|
| `ReplyStartEvent` / `ReplyEndEvent` | `reply_id`；结束事件有 `finished_reason`、`error` |
| `TextBlockDeltaEvent` | `delta`（增量文本，接 TTS） |
| `TextBlockEndEvent` | `text` |
| `ToolCallStartEvent` | `tool_call_id`、`tool_call_name`（可用来播报“正在帮你点餐”） |
| `ToolResultEndEvent` | `tool_call_id`、`state` |
| `RequireUserConfirmEvent` / `RequireExternalExecutionEvent` | 流会在这里停住，等外部结果 |

## 分类器（`agentscope.classifier`）

```python
BinaryQuestion(instructions=str, criteria=BinaryCriteria(true=str, false=str))
ChoiceQuestion(instructions=str, criteria={"选项名": "描述文字", ...})   # 描述只能是字符串
ScoreQuestion(instructions=str, criteria=["0 分描述", "1 分描述", ...])
res = await classifier(state=str 或 dict, questions={"名字": 问题})
res.content["名字"]   # BinaryAnswer.probability / ChoiceAnswer.choice, .confidence, .probabilities
res.usage             # input_tokens、output_tokens、time
```

## 中间件（`agentscope.middleware`）

钩子：`on_reply`、`on_reasoning`、`on_acting`、`on_check_permission`、`on_model_call`、`on_compress_context`、`on_system_prompt`。
内置：`ModelRouterMiddleware`（用分类器按句子选聊天模型）、`ReplyBudgetControlMiddleware`、`TracingMiddleware`、`TTSMiddleware`、长期记忆（`Mem0Middleware`、`ReMeMiddleware`、`AgenticMemoryMiddleware`）、`RAGMiddleware`。

`ModelRouterMiddleware(classifier_model=JevClassifierModel(...), candidates=[ChatModelCandidate(name, model, description), ...])`
能让同一个 Agent 按句子在大小模型之间切换，例如闲聊用小模型、多步编排用大模型。

## 多 Agent

- 子 Agent 当工具：见 `examples/brain_demo.py` 的 `make_subagent_tool`；简单的 `await worker.reply(...)` 包装处理不了子 Agent 的工具确认。
- `GoalPipeline(executor, verifier, max_iters)`：执行-校验循环。
- `TeamPipeline`、服务层的 Agent Team、远程 `A2AAgent`：见 `references/upstream/multi_agent_orchestration.md`。
