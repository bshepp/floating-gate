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


def test_run_transfer_accepts_measured_cfg_and_confirmed_flag():
    """Phase 1b: the same frozen gate re-runs on a MEASURED DeviceConfig with params_confirmed=True."""
    import dataclasses
    from fg_measured import device_config_from_measurements, rms_error_from_levels
    m = {"sigma_read": 0.005, "nonlin_raw": 3.0, "nonlin_effective": 1.0, "asym": 0.5,
         "sigma_write": 0.02, "drift": 0.0, "prog_rms_error": rms_error_from_levels(512)}
    cfg = device_config_from_measurements(m, "closed_loop")
    res = run_transfer(n_epochs=5, seeds=range(1), K_grid=[1], cfg=cfg, sigma_fg=cfg.sigma_read,
                       params_confirmed=True)
    assert res["nominal_cfg"] == dataclasses.asdict(cfg)
    assert res["config"]["sigma_fg"] == 0.005
    assert res["verdict"]["provisional"] is False


def test_main_measured_flag_runs_confirmed(tmp_path):
    from fg_measured import rms_error_from_levels
    mfile = tmp_path / "m.json"
    mfile.write_text(json.dumps({"sigma_read": 0.005, "nonlin_raw": 3.0, "asym": 0.5,
                                 "sigma_write": 0.02, "drift": 0.0,
                                 "prog_rms_error": rms_error_from_levels(256)}))
    out = tmp_path / "t.json"
    res = main(["--quick", "--measured", str(mfile), "--write-mode", "open_loop", "--out", str(out)])
    assert res["verdict"]["provisional"] is False
    assert res["nominal_cfg"]["n_levels"] == 0 and res["nominal_cfg"]["nonlin"] == 3.0
