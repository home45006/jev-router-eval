"""不需要密钥：用模拟 Jev 和模拟大模型把所有接口跑一遍（CI 每次提交都跑）。"""
import json
import os

for k in ("TYPESAFE_API_KEY", "DEEPSEEK_API_KEY", "FAST_MODEL_URL"):
    os.environ.pop(k, None)

from fastapi.testclient import TestClient  # noqa: E402

from .app import app  # noqa: E402


def last(resp):
    ev = json.loads(resp.text.splitlines()[-1])
    assert ev["type"] == "result", ev
    return ev["result"]


with TestClient(app) as c:
    assert c.get("/").status_code == 200
    assert c.get("/api/config").json()["backends"]["jev"] == "mock"
    r = last(c.post("/api/run", json={"query": "导航到大裤衩", "session_id": "a"}))
    assert r["destination"] == "快思考" and r["rewritten"] == "导航到中央电视台总部大楼", r
    r = last(c.post("/api/run", json={"query": "把空调调到二十六度然后导航去我常去的那家商场", "session_id": "b",
                                      "settings": {"brain_mode": "orchestrate"}}))
    assert r["destination"] == "Agent大脑" and len(r["labels"]) >= 2 and r["brain"]["tools_called"], r
    r = last(c.post("/api/run", json={"query": "右后座椅通风调成1", "session_id": "c",
                                      "settings": {"simulate_fast_reject": True}}))
    assert r["destination"] == "Agent大脑" and "快思考拒识 → 大脑" in r["path"], r
    turns = [last(c.post("/api/run", json={"query": q, "session_id": "d"}))
             for q in ["帮我点份外卖送到公司", "先把空调调低一点", "第二家，要中辣的"]]
    assert turns[0]["session"]["active"] and turns[1].get("suspended") and turns[2]["labels"] == ["美团"], turns
    from .check_real import fmt
    for x in turns:
        fmt(x)                                   # 会话占用时没有 Jev 结果，报告格式也要能处理
    items = c.get("/api/samples").json()["builtin"]
    rows = [json.loads(l) for l in c.post("/api/batch", json={"items": items}).text.splitlines()]
    assert len(rows) == len(items) and not any("error" in x for x in rows)
print("webdemo mock check ok")
