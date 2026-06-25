"""Phase-1 transfer runner: SNR-at-config, robust K-scan, assembly, CLI. Structure/wiring/determinism
only (no science numbers)."""
import json
from fg_params import fg_device_config, SIGMA_FG
from transfer_sweep import _snr_at_cfg, min_reads_to_pass, run_transfer, main


def test_snr_at_cfg_returns_nonneg_and_is_deterministic():
    cfg = fg_device_config()
    a = _snr_at_cfg("half", cfg, SIGMA_FG, n_epochs=10, seeds=range(2))
    b = _snr_at_cfg("half", cfg, SIGMA_FG, n_epochs=10, seeds=range(2))
    assert a >= 0.0
    assert a == b


def test_min_reads_to_pass_reports_snr_and_K_from_grid():
    cfg = fg_device_config()
    res = min_reads_to_pass(cfg, SIGMA_FG, K_grid=[1, 4], n_epochs=10, seeds=range(2))
    assert set(res) == {"snr_robust", "robust_reads_K"}
    assert res["robust_reads_K"] in (1, 4)
    assert res["snr_robust"] >= 0.0


def test_run_transfer_assembles_gate_elevated_and_audits():
    res = run_transfer(n_epochs=10, seeds=range(2), K_grid=[1, 4])
    assert res["verdict"]["status"] in {"PASS", "BOUNDARY", "FORECLOSE"}
    assert res["verdict"]["snr_half"] == res["snr_half"]
    assert res["verdict"]["provisional"] is True
    # elevated in-fabric records present
    assert "full_status" in res["verdict"] and "robust_status" in res["verdict"]
    assert res["robust_reads_K"] in (1, 4)
    # two-axis audit: half not both-physical, full both-physical
    assert res["audit"]["half"]["is_physical"] is False
    assert res["audit"]["full"]["is_physical"] is True
    # Observed knob sweeps present and JSON-safe
    assert set(res["sweeps"]) and json.dumps(res)


def test_run_transfer_endurance_recorded():
    res = run_transfer(n_epochs=10, seeds=range(1), K_grid=[1])
    assert res["verdict"]["endurance_ok"] is True       # 10 writes << 1e5 budget
    assert res["verdict"]["writes_per_cell"] == 10


def test_main_quick_writes_json_with_verdict(tmp_path):
    out = tmp_path / "transfer.json"
    res = main(["--quick", "--out", str(out)])
    assert out.exists()
    data = json.loads(out.read_text())
    assert set(data) >= {"config", "nominal_cfg", "verdict", "sweeps", "audit",
                         "snr_half", "snr_full", "snr_robust", "robust_reads_K", "writes_per_cell"}
    assert data["verdict"]["provisional"] is True
    assert res["config"]["n_seeds"] == 2
