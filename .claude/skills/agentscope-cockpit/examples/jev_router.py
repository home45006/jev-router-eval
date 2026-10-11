"""前置分流：用 AgentScope 内置的 JevClassifierModel 实现两级分流（第 13 版架构）。

第一级（每句一次）：is_fast、needs_rewrite 两个二元问题
第二级（只有不走快思考的句子）：4 个独立 Agent、Agent 大脑、退回快思考 6 选 1

判据来自仓库根目录的 criteria_v2.json（由 make_criteria_v2.py 生成）。
AgentScope 的 ChoiceQuestion 只接受“选项名 → 一段描述文字”，所以这里把
what / not_for / examples 拼成一段文字。阈值要在 val 上标定，下面是占位值。

运行：
    离线（假 Jev，只验证接线）：python jev_router.py --offline "导航到大裤衩"
    真实：export TYPESAFE_API_KEY=...；pip install 'agentscope[classifier-jev]'
          python jev_router.py "把空调调到二十六度然后导航去我常去的那家商场"
"""
import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from agentscope.classifier import (
    BinaryCriteria,
    BinaryQuestion,
    ChoiceQuestion,
    ClassifierModelBase,
)

REPO = Path(__file__).resolve().parents[4]
IS_FAST_T = 0.7          # 占位，val 上标定
REWRITE_T = 0.5          # 占位
STAGE2_CONF_T = 0.5      # 第二级置信度不足时交给 Agent 大脑
BACK = "退回快思考"


def _flatten(option: dict) -> str:
    text = option["what"]
    if option.get("not_for"):
        text += " 不包括：" + option["not_for"]
    if option.get("examples"):
        text += " 例：" + "；".join(option["examples"])
    return text


def load_questions(criteria_path: Path) -> tuple[dict, dict]:
    crit = json.loads(criteria_path.read_text(encoding="utf-8"))
    stage1 = {
        name: BinaryQuestion(instructions=q["instructions"], criteria=BinaryCriteria(**q["criteria"]))
        for name, q in crit["stage1"].items()
    }
    ins = crit["stage2"]["instructions"]
    stage2 = {
        "label": ChoiceQuestion(
            instructions=" ".join(ins.values()) if isinstance(ins, dict) else ins,
            criteria={k: _flatten(v) for k, v in crit["stage2"]["options"].items()},
        ),
    }
    return stage1, stage2


@dataclass
class Route:
    destination: str            # 快思考 / 用车手册Agent / 任务大师 / 条件任务 / 联网搜索Agent / Agent大脑
    rewrite: bool = False       # 交给快思考前是否先用大模型改写模糊指代
    detail: dict | None = None  # 两级的原始概率，便于记日志和标定阈值


class JevRouter:
    def __init__(self, classifier: ClassifierModelBase, criteria_path: Path = REPO / "criteria_v2.json") -> None:
        self.classifier = classifier
        self.stage1, self.stage2 = load_questions(criteria_path)

    async def route(self, query: str, previous_turn: dict | None = None,
                    now_playing: dict | None = None, allow_back: bool = True) -> Route:
        state = {"query": query}
        if previous_turn:
            state["previous_turn"] = previous_turn
        if now_playing:
            state["now_playing"] = now_playing

        s1 = (await self.classifier(state=state, questions=self.stage1)).content
        is_fast = s1["is_fast"].probability
        rewrite = s1["needs_rewrite"].probability
        detail = {"is_fast": is_fast, "needs_rewrite": rewrite}
        if is_fast >= IS_FAST_T:
            return Route("快思考", rewrite=rewrite >= REWRITE_T, detail=detail)

        s2 = (await self.classifier(state=state, questions=self.stage2)).content["label"]
        detail.update(label=s2.choice, confidence=s2.confidence)
        if s2.choice == BACK and allow_back:
            return Route("快思考", rewrite=rewrite >= REWRITE_T, detail=detail)
        if s2.choice == BACK or s2.confidence < STAGE2_CONF_T:
            return Route("Agent大脑", detail=detail)
        return Route(s2.choice, detail=detail)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from scripted_model import ScriptedClassifier
        classifier = ScriptedClassifier({"is_fast": 0.92, "needs_rewrite": 0.88})
    else:
        from agentscope.classifier import JevClassifierModel
        from agentscope.credential import TypeSafeCredential
        classifier = JevClassifierModel(credential=TypeSafeCredential(api_key=os.environ["TYPESAFE_API_KEY"]))
    route = await JevRouter(classifier).route(args.query)
    print(route)


if __name__ == "__main__":
    asyncio.run(main())
