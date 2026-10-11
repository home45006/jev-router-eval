# 座舱前置分流在 AgentScope 里怎么落地

架构以 `docs/jev-router-arch-v4.html`（第 13 版）为准，这里只讲怎么用 AgentScope 实现。

## 组件对应

| 设计里的环节 | AgentScope 里的实现 | 示例 |
|---|---|---|
| 有进行中的会话吗 | 应用代码：按 session_id 查会话存储里是否有大脑的 `AgentState` | `brain_demo.SessionStore` |
| Jev 第一级（is_fast、needs_rewrite） | `JevClassifierModel` + 两个 `BinaryQuestion`，一次请求 | `jev_router.JevRouter.route` |
| 大模型改写模糊指代 | 一个没有工具的 `Agent`（DeepSeek），只输出改写后的一句话；用 `structured_schema` 约束输出 | 自行实现 |
| 快思考模型 | 应用直接调用快思考服务；同时包装成 `FunctionTool` 交给大脑 | `brain_demo.fast_think` |
| Jev 第二级（6 选 1） | `JevClassifierModel` + 一个 `ChoiceQuestion` | `jev_router.JevRouter.route` |
| 4 个独立 Agent | 各自一个 `Agent`，前置分流直接调用 | 按 `make_subagent_tool` 的写法建 |
| Agent 大脑 | 一个 `Agent`，toolkit 里是快思考工具、各子 Agent 工具、联网搜索工具 | `brain_demo.build_brain` |
| 会话占用的三种状态 | 大脑每轮 `structured_schema=TurnStatus`，返回 continue / suspend / end | `brain_demo.TurnStatus` |

## 一轮请求的完整顺序

```python
async def on_utterance(session_id, query, previous_turn, now_playing):
    if store.load(session_id, "agent_brain"):             # 有进行中的会话
        turn = await handle_turn(session_id, query, store, models)
        if turn.status != "suspend":
            return speak(turn.speech)
        # suspend：这句不归当前任务，往下走正常分流；会话保留

    route = await router.route(query, previous_turn, now_playing)
    if route.destination == "快思考":
        text = await rewrite(query) if route.rewrite else query
        result = await fast_think_service(text, previous_turn)
        if result.rejected or result.confidence < FAST_T:   # 快思考拒识或低置信
            route = await router.route(query, previous_turn, now_playing, allow_back=False)
        else:
            return execute(result)
    if route.destination == "Agent大脑":
        turn = await handle_turn(session_id, query, store, models)
        return speak(turn.speech)
    return await standalone_agents[route.destination](query)
```

挂起后恢复：被挂起的那句处理完后，下一句如果又回到大脑，大脑的上下文里还留着之前的任务，
在 system prompt 里要求它先确认是否继续（“刚才的外卖还要继续下单吗”）。

## 子 Agent 和会话

- 子 Agent 的 state 也按会话保存（`store.save(session_id, spec.name, ...)`），美团、导航这类多轮的子任务才能接上。
- 会话由大脑持有。用户的后续发言先到大脑，大脑决定转给哪个子 Agent。
- 子 Agent 看不到用户和大脑的完整对话，大脑调用时要在 `task` 里写清上下文。
- 超时、轮数上限、退出词这些兜底放在应用代码里，不要只靠大脑判断。会话结束时 `store.drop(session_id)`。

## 时延

- 大脑 `ReActConfig(max_iters=6~10)`；子 Agent 一般 3~5。
- 用 `reply_stream` 拿 `TextBlockDeltaEvent.delta` 边生成边送 TTS；`ToolCallStartEvent.tool_call_name` 可以用来先播报“正在帮你查”。
- DeepSeek 用 `client_kwargs={"timeout": ...}` 设超时，`ModelConfig(fallback_model=...)` 设备用模型。
- 闲聊和简单问答可以用 `ModelRouterMiddleware` 切到更快的模型，复杂编排再用大模型。

## 测试

- 离线：`scripts/scripted_model.py` 的 `ScriptedChatModel`（按脚本返回回复、工具调用）和 `ScriptedClassifier`（固定概率）。
  只用来验证接线和状态流转，报告里要和真实模型的结果分开写。
- 分流准确率：用仓库里的 `eval_two_stage.py` 在 GitHub Actions 上跑（本地网络访问不了 TypeSafe）。
- 大脑：需要另建多轮场景集，看任务完成率、调了哪些能力、轮数和耗时；测试集里没有这类数据。
