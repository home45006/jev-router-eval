"""前置分流：用 AgentScope 内置的 JevClassifierModel 实现第 14 版架构。

每句只调一次 Jev，同一次请求里问：
    is_fast        能不能直接交给快思考模型
    needs_rewrite  有没有要靠常识还原的模糊指代（交给快思考前先用 DeepSeek 改写）
    纠错题         criteria_v2.json 里 corrections 中 status 不是 off 的题
去向只有两个：快思考，或 Agent 大脑。纠错题命中（status=on）时按它的 route 走，
route 是 Agent大脑 时把 hint（建议调用的能力）一起交给大脑；status=shadow 的题只记录命中。

判据来自仓库根目录的 criteria_v2.json（由 make_criteria_v2.py 生成）。阈值要在 val 上标定，下面是当前值。

运行：
    离线（假 Jev，只验证接线）：python jev_router.py --offline "导航到大裤衩"
    真实：export TYPESAFE_API_KEY=...；pip install 'agentscope[classifier-jev]'
          python jev_router.py "把空调调到二十六度然后导航去我常去的那家商场"
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agentscope.classifier import BinaryCriteria, BinaryQuestion, ClassifierModelBase

REPO = Path(__file__).resolve().parents[4]
IS_FAST_T = 0.7          # val 上第一级准确率 93.0%（v2.2 判据）
REWRITE_T = 0.5
CORRECTION_T = 0.5       # 纠错题命中阈值，回放时标定
FAST, BRAIN = "快思考", "Agent大脑"

log = logging.getLogger("jev_router")


def _question(q: dict) -> BinaryQuestion:
    return BinaryQuestion(instructions=q["instructions"], criteria=BinaryCriteria(**q["criteria"]))


@dataclass
class Correction:
    name: str
    status: str               # on / shadow
    route: str                # 快思考 / Agent大脑
    hint: str | None          # route 为 Agent大脑 时建议大脑调用的能力


def load_questions(criteria_path: Path) -> tuple[dict[str, BinaryQuestion], dict[str, Correction]]:
    crit = json.loads(criteria_path.read_text(encoding="utf-8"))
    questions = {name: _question(q) for name, q in crit["stage1"].items()}
    corrections = {}
    for name, q in crit.get("corrections", {}).items():
        if q.get("status", "off") == "off":
            continue
        questions[name] = _question(q)
        corrections[name] = Correction(name, q["status"], q["route"], q.get("hint"))
    return questions, corrections


@dataclass
class Route:
    destination: str                    # 快思考 / Agent大脑
    rewrite: bool = False               # 交给快思考前是否先用大模型改写模糊指代
    hint: str | None = None             # 纠错题给大脑的能力提示
    correction: str | None = None       # 决定去向的纠错题
    detail: dict = field(default_factory=dict)   # 原始概率，记日志、标定阈值、回放纠错题用


class JevRouter:
    def __init__(self, classifier: ClassifierModelBase, criteria_path: Path = REPO / "criteria_v2.json") -> None:
        self.classifier = classifier
        self.questions, self.corrections = load_questions(criteria_path)

    async def route(self, query: str, previous_turn: dict | None = None, now_playing: dict | None = None) -> Route:
        state = {"query": query}
        if previous_turn:
            state["previous_turn"] = previous_turn
        if now_playing:
            state["now_playing"] = now_playing

        ans = (await self.classifier(state=state, questions=self.questions)).content
        detail = {k: v.probability for k, v in ans.items()}
        rewrite = detail["needs_rewrite"] >= REWRITE_T

        hits = sorted((c for c in self.corrections.values() if detail[c.name] >= CORRECTION_T),
                      key=lambda c: -detail[c.name])
        for c in hits:
            if c.status == "shadow":
                log.info("shadow correction hit: %s -> %s, query=%s", c.name, c.route, query)
        live = [c for c in hits if c.status == "on"]
        if live:
            c = live[0]
            return Route(c.route, rewrite=rewrite and c.route == FAST, hint=c.hint, correction=c.name, detail=detail)

        if detail["is_fast"] >= IS_FAST_T:
            return Route(FAST, rewrite=rewrite, detail=detail)
        return Route(BRAIN, detail=detail)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    if args.offline:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from scripted_model import ScriptedClassifier
        classifier = ScriptedClassifier({"is_fast": 0.92, "needs_rewrite": 0.88, "fix_mode_rule": 0.1,
                                         "fix_setting_toggle": 0.05})
    else:
        from agentscope.classifier import JevClassifierModel
        from agentscope.credential import TypeSafeCredential
        classifier = JevClassifierModel(credential=TypeSafeCredential(api_key=os.environ["TYPESAFE_API_KEY"]))
    route = await JevRouter(classifier).route(args.query)
    print(route)


if __name__ == "__main__":
    asyncio.run(main())
