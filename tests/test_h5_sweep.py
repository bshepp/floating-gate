"""Phase-0 H5 runner: pure helpers, sweep integration, verdict assembly, and CLI."""
from h5_sweep import foreclosure_wall, rule_audit, sweep


def test_foreclosure_wall_finds_first_subthreshold():
    sigmas = [0.0, 0.01, 0.05, 0.1]
    snr = [5.0, 4.0, 0.5, 0.1]
    assert foreclosure_wall(sigmas, snr) == 0.05


def test_foreclosure_wall_none_when_robust():
    assert foreclosure_wall([0.0, 0.1], [5.0, 3.2]) is None


def test_foreclosure_wall_scans_in_sigma_order():
    # input order is scrambled; the wall is still found by ascending train_sigma
    assert foreclosure_wall([0.1, 0.0, 0.05], [0.2, 5.0, 4.0]) == 0.1


def test_rule_audit_half_is_not_physical():
    assert rule_audit("half").is_physical() is False


def test_rule_audit_full_is_physical():
    assert rule_audit("full").is_physical() is True


def test_sweep_returns_one_snr_row_per_sigma():
    rows = sweep("half", [0.0, 0.1], n_epochs=10, seeds=range(2))
    assert [r["train_sigma"] for r in rows] == [0.0, 0.1]
    for r in rows:
        assert r["snr_mean"] >= 0.0
        assert len(r["snr_per_seed"]) == 2
        assert r["audit"]["error_site"] == "off_array_cpu"


def test_sweep_full_rule_carries_in_fabric_audit():
    rows = sweep("full", [0.01], n_epochs=10, seeds=range(2))
    assert rows[0]["audit"]["error_site"] == "on_array_local"


def test_sweep_is_deterministic():
    a = sweep("half", [0.05], n_epochs=10, seeds=range(2))
    b = sweep("half", [0.05], n_epochs=10, seeds=range(2))
    assert a[0]["snr_per_seed"] == b[0]["snr_per_seed"]


from h5_sweep import run_h5
from verdict import SIGMA_REALISTIC_FG


def test_run_h5_assembles_both_verdicts_and_audits():
    res = run_h5(sigmas=[0.0, 0.01, 0.1], seeds=range(2), n_epochs=10)
    # frozen primary verdict keyed on the half rule
    assert res["verdict"]["status"] in {"PASS", "BOUNDARY", "FORECLOSE"}
    assert res["verdict"]["snr_half"] == res["snr_half"]
    assert res["verdict"]["provisional"] is True
    assert res["config"]["sigma_fg"] == SIGMA_REALISTIC_FG
    # elevated co-equal in-fabric verdict keyed on the full rule
    assert res["in_fabric_verdict"]["snr_full"] == res["snr_full"]
    assert res["in_fabric_verdict"]["status"] in {"PASS", "BOUNDARY", "FORECLOSE"}
    assert res["in_fabric_verdict"]["is_physical"] is True
    # two-axis audit: half NOT both-physical, full both-physical
    assert res["audit"]["half"]["is_physical"] is False
    assert res["audit"]["full"]["is_physical"] is True


def test_run_h5_inserts_sigma_fg_if_missing():
    res = run_h5(sigmas=[0.0, 0.1], seeds=range(1), n_epochs=8)
    assert SIGMA_REALISTIC_FG in res["config"]["sigmas"]


def test_run_h5_audit_breakdown_is_json_safe():
    import json
    res = run_h5(sigmas=[0.01], seeds=range(1), n_epochs=8)
    json.dumps(res)  # must not raise (tuple keys would)
