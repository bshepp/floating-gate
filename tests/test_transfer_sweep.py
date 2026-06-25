"""Phase-1 transfer runner: SNR-at-config, robust K-scan, assembly, CLI. Structure/wiring/determinism
only (no science numbers)."""
from fg_params import fg_device_config, SIGMA_FG
from transfer_sweep import _snr_at_cfg, min_reads_to_pass


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
