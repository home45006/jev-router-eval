"""座舱 Agent 大脑的最小实现（AgentScope 2.0.9）。

演示四件事：
1. 快思考模型作为工具（FunctionTool）交给大脑；
2. 子 Agent（美团、导航、闲聊……）包装成工具，由大脑自主调度；
3. 大脑每轮用结构化输出返回会话状态：continue / suspend / end；
4. 每个会话的 AgentState 存成 JSON，每轮重建 Agent，不在会话之间共享 Agent 实例。

运行：
    离线（假模型，只验证接线，结果不代表模型效果）：
        python brain_demo.py --offline
    接真实模型：
        export DEEPSEEK_API_KEY=...
        export DEEPSEEK_MODEL=<DeepSeek V4 Flash 的模型 ID>
        export DEEPSEEK_BASE_URL=https://api.deepseek.com   # 走代理时改成代理地址
        python brain_demo.py
"""
import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from agentscope.agent import Agent, ReActConfig
from agentscope.credential import DeepSeekCredential
from agentscope.message import UserMsg
from agentscope.model import ChatModelBase, DeepSeekChatModel
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, Toolkit
from agentscope.types import ReplyFinishedReason

# FunctionTool 默认每次调用都要用户确认；车机工具在这里统一放行。
# 涉及支付、下单这类高风险动作，改成 ASK 并在前端处理 RequireUserConfirmEvent。
ALLOW = PermissionDecision(behavior=PermissionBehavior.ALLOW, message="座舱工具，无需确认")


# --------------------------------------------------------------------------- #
# 会话状态
# --------------------------------------------------------------------------- #
class TurnStatus(BaseModel):
    """大脑每轮的结构化输出。"""

    status: Literal["continue", "suspend", "end"] = Field(
        description="continue：任务还在进行，下一句仍交给大脑；"
        "suspend：这句话不属于当前任务，交回前置分流处理，任务保留稍后继续；"
        "end：任务完成、用户放弃或闲聊结束，释放会话",
    )
    speech: str = Field(description="播报给用户的话；status 为 suspend 时可以为空")


@dataclass
class SessionStore:
    """按会话保存大脑和子 Agent 的 AgentState。线上换成 Redis 或数据库。"""

    states: dict[str, str] = field(default_factory=dict)

    def load(self, session_id: str, agent_name: str) -> AgentState | None:
        raw = self.states.get(f"{session_id}/{agent_name}")
        return AgentState.model_validate_json(raw) if raw else None

    def save(self, session_id: str, agent_name: str, state: AgentState) -> None:
        self.states[f"{session_id}/{agent_name}"] = state.model_dump_json()

    def drop(self, session_id: str) -> None:
        for key in [k for k in self.states if k.startswith(f"{session_id}/")]:
            del self.states[key]


# --------------------------------------------------------------------------- #
# 能力：快思考工具和子 Agent
# --------------------------------------------------------------------------- #
async def fast_think(query: str) -> str:
    """把一条或连着几条说清楚的车机指令交给快思考模型解析并执行，返回执行结果。

    Args:
        query: 说清楚的直接指令，例如“空调调到26度并打开座椅加热”。
    """
    # TODO: 换成快思考服务的真实调用，返回 domain、意图、槽位和执行结果。
    return json.dumps({"query": query, "status": "done", "note": "快思考服务未接入，这是占位结果"}, ensure_ascii=False)


@dataclass
class SubAgentSpec:
    """一个由大脑调度的子 Agent。"""

    name: str
    description: str          # 给大脑看的工具说明：它能做什么、什么时候用
    system_prompt: str
    tools: list = field(default_factory=list)


SUB_AGENTS = [
    SubAgentSpec("meituan_agent", "美团 Agent：找餐厅、点外卖、下单。需要多轮确认菜品和地址。",
                 "你是美团点餐助手。缺信息时向用户提问，下单前复述订单。"),
    SubAgentSpec("navigation_agent", "导航 Agent：靠描述找地点、规划路线、多地点行程、顺路找加油站餐馆。",
                 "你是车载导航助手。"),
    SubAgentSpec("chat_agent", "闲聊 Agent：开放问答、百科、写诗、建议和陪伴聊天。用户在闲聊时调用。",
                 "你是车上的聊天伙伴，回答简短口语化。"),
]


def make_subagent_tool(spec: SubAgentSpec, session_id: str, store: SessionStore,
                       model_factory: Callable[[str], ChatModelBase]) -> FunctionTool:
    """把子 Agent 包装成大脑的工具。子 Agent 的状态按会话保存，多轮对话能接上。"""

    async def run(task: str) -> str:
        agent = Agent(
            name=spec.name,
            system_prompt=spec.system_prompt,
            model=model_factory(spec.name),
            toolkit=Toolkit(tools=spec.tools) if spec.tools else None,
            state=store.load(session_id, spec.name),
        )
        res = await agent.reply(UserMsg(name="brain", content=task))
        store.save(session_id, spec.name, agent.state)
        if res.finished_reason != ReplyFinishedReason.COMPLETED:
            return f"{spec.name} 未完成：{res.finished_reason}"
        return res.get_text_content()

    return FunctionTool(
        run,
        name=spec.name,
        description=spec.description + " 参数 task 写清楚任务和必要的上下文；子 Agent 看不到用户原话以外的对话。",
        permission=ALLOW,
    )


BRAIN_PROMPT = """你是智能座舱的 Agent 大脑。你可以调用以下能力完成用户的事：
- fast_think：执行说清楚的车控、媒体、导航、电话等直接指令；
- 各个子 Agent：美团点餐、导航规划、闲聊等。
一件事需要几个能力配合时，按顺序调度它们。每轮结束时用 GenerateStructuredOutput 返回会话状态：
任务还要继续时 status=continue；用户这句话和当前任务无关（例如点外卖途中说“把空调调低”）时 status=suspend；
任务完成、用户放弃或闲聊告一段落时 status=end。"""


def build_brain(session_id: str, store: SessionStore, model_factory: Callable[[str], ChatModelBase]) -> Agent:
    tools = [FunctionTool(fast_think, permission=ALLOW)]
    tools += [make_subagent_tool(s, session_id, store, model_factory) for s in SUB_AGENTS]
    return Agent(
        name="agent_brain",
        system_prompt=BRAIN_PROMPT,
        model=model_factory("agent_brain"),
        toolkit=Toolkit(tools=tools),
        state=store.load(session_id, "agent_brain"),
        react_config=ReActConfig(max_iters=8),  # 车上要控制时延，限制推理轮数
    )


async def handle_turn(session_id: str, text: str, store: SessionStore,
                      model_factory: Callable[[str], ChatModelBase]) -> TurnStatus:
    """大脑处理一轮。调用方根据返回的 status 决定下一句是否还交给大脑。"""
    brain = build_brain(session_id, store, model_factory)
    res = await brain.reply(UserMsg(name="user", content=text), structured_schema=TurnStatus)
    if res.finished_reason != ReplyFinishedReason.COMPLETED or not res.structured_output:
        store.drop(session_id)
        return TurnStatus(status="end", speech="抱歉，这件事我没办成。")
    turn = TurnStatus.model_validate(res.structured_output)
    if turn.status == "end":
        store.drop(session_id)
    else:
        store.save(session_id, "agent_brain", brain.state)
    return turn


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def real_model_factory(agent_name: str) -> ChatModelBase:
    return DeepSeekChatModel(
        credential=DeepSeekCredential(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        ),
        model=os.environ["DEEPSEEK_MODEL"],
    )


def offline_model_factory():
    """离线脚本：三轮对话——开始点外卖、途中说调空调（挂起）、回来继续点。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripted_model import ScriptedChatModel, call, say

    scripts = {
        "agent_brain": [
            call("meituan_agent", {"task": "用户想点外卖，送到公司"}, "b1"),
            call("GenerateStructuredOutput", {"status": "continue", "speech": "附近有三家川菜，要哪家？"}, "b2"),
            call("GenerateStructuredOutput", {"status": "suspend", "speech": ""}, "b3"),
            call("meituan_agent", {"task": "选第二家，菜要中辣"}, "b4"),
            call("GenerateStructuredOutput", {"status": "continue", "speech": "好的，中辣。地址用公司吗？"}, "b5"),
        ],
        "meituan_agent": [say("附近有三家川菜：A、B、C。"), say("已选 B 家，口味中辣。")],
    }

    def factory(agent_name: str) -> ChatModelBase:
        return ScriptedChatModel([scripts[agent_name].pop(0)] if agent_name == "meituan_agent"
                                 else scripts[agent_name])

    # 大脑每轮重建，但要共享同一个脚本队列
    brain_model = ScriptedChatModel(scripts["agent_brain"])
    return lambda name: brain_model if name == "agent_brain" else factory(name)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="用假模型验证接线")
    args = ap.parse_args()
    factory = offline_model_factory() if args.offline else real_model_factory
    store = SessionStore()
    for text in ["帮我点份外卖送到公司", "先把空调调低一点", "第二家，要中辣的"]:
        turn = await handle_turn("demo-session", text, store, factory)
        print(f"用户：{text}\n大脑：status={turn.status} speech={turn.speech!r}\n")
        # status=suspend 时，这句话应交给前置分流（Jev 第一级）处理；会话保留，下一句仍可回到大脑


if __name__ == "__main__":
    asyncio.run(main())
