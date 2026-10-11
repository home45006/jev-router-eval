# 座舱前置分流在 AgentScope 里怎么落地

架构以 `docs/jev-router-arch-v4.html`（第 14 版）为准，这里只讲怎么用 AgentScope 实现。
第 14 版只有两个去向：快思考，或 Agent 大脑。任务大师、条件任务、联网搜索、用车手册等不再直达，都挂在大脑下面。

## 组件对应

| 设计里的环节 | AgentScope 里的实现 | 示例 |
|---|---|---|
| 有进行中的会话吗 | 应用代码：按 session_id 查会话存储里是否有大脑的 `AgentState` | `brain_demo.SessionStore` |
| Jev 第一级（is_fast、needs_rewrite、纠错题） | `JevClassifierModel` + 若干 `BinaryQuestion`，一次请求 | `jev_router.JevRouter.route` |
| Jev 纠错题 | `criteria_v2.json` 的 `corrections`；shadow 只记日志，on 命中按 `route` 走 | `jev_router.load_questions` |
| 大模型改写模糊指代 | 一个没有工具的 `Agent`（DeepSeek V4 Flash 非思考），只输出改写后的一句话 | 自行实现；提示词见 `rewrite_check.py` |
| 快思考模型 | 应用直接调用快思考服务；同时包装成 `FunctionTool` 交给大脑 | `brain_demo.fast_think` |
| Agent 大脑 | 一个 `Agent`，toolkit 里是快思考工具和各能力的子 Agent 工具 | `brain_demo.build_brain` |
| 大脑下挂的能力 | 每个能力一个 `Agent`，用 `make_subagent_tool` 包成工具；能力清单是 `SUB_AGENTS` | `brain_demo.SUB_AGENTS` |
| 会话占用的三种状态 | 大脑每轮 `structured_schema=TurnStatus`，返回 continue / suspend / end | `brain_demo.TurnStatus` |

## 一轮请求的完整顺序

```python
async def on_utterance(session_id, query, previous_turn, now_playing):
    if store.load(session_id, "agent_brain"):             # 有进行中的会话
        turn = await handle_turn(session_id, query, store, models)
        if turn.status != "suspend":
            return speak(turn.speech)
        # suspend：这句不归当前任务，往下走正常分流；会话保留

    route = await router.route(query, previous_turn, now_playing)   # 一次 Jev 请求
    if route.destination == "快思考":
        text = await rewrite(query) if route.rewrite else query
        result = await fast_think_service(text, previous_turn)
        if not (result.rejected or result.confidence < FAST_T):   # 快思考拒识或低置信时交给大脑
            return execute(result)
    turn = await handle_turn(session_id, query, store, models, hint=route.hint)
    return speak(turn.speech)
```

挂起后恢复：被挂起的那句处理完后，下一句如果又回到大脑，大脑的上下文里还留着之前的任务，
在 system prompt 里要求它先确认是否继续（“刚才的外卖还要继续下单吗”）。

## 纠错题的生命周期

1. 线上发现一类句子落错（例如“山地模式时空气悬架调到便携装载模式”被交给快思考）。
2. 在 `make_criteria_v2.py` 的 `CORRECTIONS` 里加一道题：`instructions`、`criteria.true/false`（带正例和反例）、`route`、`hint`，`status="shadow"`。
3. 用 `eval_two_stage.py` 在评测集上回放，看“改变去向 / 改对 / 改错”。有改错的就收窄描述，直到改错为 0 或可以接受。
4. 线上先 shadow，日志里看命中的句子没有误伤，再改成 `status="on"`。
5. 快思考的能力清单、第一级判据或大脑提示词修好后，把这道题改成 `off`。纠错题保持在十道以内。

## 子 Agent 和会话

- 子 Agent 的 state 也按会话保存（`store.save(session_id, spec.name, ...)`），美团、导航这类多轮的子任务才能接上。
- 会话由大脑持有。用户的后续发言先到大脑，大脑决定转给哪个子 Agent。
- 子 Agent 看不到用户和大脑的完整对话，大脑调用时要在 `task` 里写清上下文。
- 超时、轮数上限、退出词这些兜底放在应用代码里，不要只靠大脑判断。会话结束时 `store.drop(session_id)`。

## 时延

- 大脑 `ReActConfig(max_iters=6~10)`；子 Agent 一般 3~5。
- 用 `reply_stream` 拿 `TextBlockDeltaEvent.delta` 边生成边送 TTS；`ToolCallStartEvent.tool_call_name` 可以用来先播报“正在帮你查”。
- DeepSeek 用 V4 Flash 非思考模式（`Parameters(thinking_enable=False)`），`client_kwargs={"timeout": ...}` 设超时，`ModelConfig(fallback_model=...)` 设备用模型。
- 闲聊和简单问答可以用 `ModelRouterMiddleware` 切到更快的模型，复杂编排再用大模型。

## 测试

- 离线：`scripts/scripted_model.py` 的 `ScriptedChatModel`（按脚本返回回复、工具调用）和 `ScriptedClassifier`（固定概率）。
  只用来验证接线和状态流转，报告里要和真实模型的结果分开写。
- 分流准确率和纠错题回放：用仓库里的 `eval_two_stage.py` 在 GitHub Actions 上跑（本地网络访问不了 TypeSafe）。
- 大脑：需要另建多轮场景集，看任务完成率、调了哪些能力、轮数和耗时；测试集里没有这类数据。
