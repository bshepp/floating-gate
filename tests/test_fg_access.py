"""fg_access: the observed Okika inputs through the frozen gate -- wiring + the recorded outcome shape
(the outcome itself is a RECORD of observed inputs, so we check it matches the inputs, not that it is
'right')."""
import json

from fg_access import (OKIKA_OTC2902K, OKIKA_FLEXANALOG_AN231K04, CANDIDATES,
                       access_record, run_access, main)


def test_every_input_carries_an_evidence_grade():
    for c in CANDIDATES:
        assert set(c["inputs"]) == set(c["evidence"])
        assert all(c["evidence"].values())


def test_otc2902k_record_is_buy_with_inference_grade_ceiling_and_open_vendor_question():
    r = access_record(OKIKA_OTC2902K)
    v = r["verdict"]
    assert v["status"] == "BUY"
    assert v["audit_ceiling"] == "(cpu, physical)"      # as sold; in-fabric NOT established
    assert v["in_fabric_path"] == "unknown"
    assert v["vendor_query_needed"] is True


def test_flexanalog_is_the_negative_control():
    v = access_record(OKIKA_FLEXANALOG_AN231K04)["verdict"]
    assert v["status"] == "FORECLOSE"
    assert v["reason"].startswith("not a floating-gate")


def test_budget_override_can_foreclose_the_buy():
    res = run_access(budget_usd=2000.0)
    assert res["n_buy"] == 0 and res["headline"] is None
    assert res["records"][0]["verdict"]["reason"].startswith("price exceeds")


def test_main_writes_json(tmp_path):
    out = tmp_path / "access.json"
    res = main(["--out", str(out)])
    data = json.loads(out.read_text())
    assert data["headline"] == OKIKA_OTC2902K["device"]
    assert res["n_buy"] == 1
