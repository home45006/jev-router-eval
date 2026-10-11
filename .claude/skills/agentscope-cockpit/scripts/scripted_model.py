"""离线测试用的假模型：不联网，按顺序返回预先写好的回复。

用法见 examples/brain_demo.py 的 --offline 模式。只用来验证编排流程和代码接线，
不代表真实模型的效果；报告结果时要和真实 DeepSeek / Jev 的运行区分开。
"""
import json
from typing import Any

from pydantic import BaseModel

from agentscope.classifier import (
    BinaryAnswer,
    ChoiceAnswer,
    ChoiceQuestion,
    ClassifierModelBase,
    ClassifierResponse,
)
from agentscope.credential import DeepSeekCredential, TypeSafeCredential
from agentscope.formatter import DeepSeekChatFormatter
from agentscope.message import TextBlock, ToolCallBlock
from agentscope.model import ChatModelBase, ChatResponse


def say(text: str) -> ChatResponse:
    """一条纯文本回复。"""
    return ChatResponse(content=[TextBlock(text=text)], is_last=True)


def call(name: str, args: dict, call_id: str = "call-1") -> ChatResponse:
    """一次工具调用。结构化输出的工具名是 GenerateStructuredOutput。"""
    return ChatResponse(
        content=[ToolCallBlock(id=call_id, name=name, input=json.dumps(args, ensure_ascii=False))],
        is_last=True,
    )


class ScriptedChatModel(ChatModelBase):
    """按顺序返回 script 里的 ChatResponse。script 用完会抛 IndexError。"""

    class Parameters(BaseModel):
        """无参数。"""

    def __init__(self, script: list[ChatResponse]) -> None:
        super().__init__(
            credential=DeepSeekCredential(api_key="offline"),
            model="scripted",
            parameters=self.Parameters(),
            stream=False,
        )
        # Agent 会读取 model.formatter，自定义模型必须设置
        self.formatter = DeepSeekChatFormatter()
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def _call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        self.calls.append({"tools": [t["function"]["name"] for t in (tools or [])]})
        return self.script.pop(0)


class ScriptedClassifier(ClassifierModelBase):
    """假的 Jev：二元问题返回固定概率，选择题返回固定选项。"""

    def __init__(self, binary: dict[str, float], choice: str | None = None) -> None:
        super().__init__(credential=TypeSafeCredential(api_key="offline"), model="scripted")
        self.binary, self.choice = binary, choice

    async def __call__(self, state, questions, **kwargs) -> ClassifierResponse:
        content = {}
        for name, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                pick = self.choice or next(iter(q.criteria))
                content[name] = ChoiceAnswer(choice=pick, confidence=0.9,
                                             probabilities={k: (0.9 if k == pick else 0.1 / (len(q.criteria) - 1)) for k in q.criteria})
            else:
                content[name] = BinaryAnswer(probability=self.binary.get(name, 0.0))
        return ClassifierResponse(model="scripted", content=content)
