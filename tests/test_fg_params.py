"""The committed FG operating point: structural sanity (no science numbers asserted)."""
import dataclasses

from memristor import DeviceConfig
from fg_params import SIGMA_FG, ENDURANCE_BUDGET, fg_device_config, FG_SWEEPS, ROBUST_K_GRID


def test_sigma_fg_at_or_below_phase0_provisional():
    assert 0.0 < SIGMA_FG <= 0.01


def test_endurance_budget_is_a_positive_cycle_count():
    assert isinstance(ENDURANCE_BUDGET, int) and ENDURANCE_BUDGET > 0


def test_nominal_cfg_is_fg_directional():
    cfg = fg_device_config()
    assert isinstance(cfg, DeviceConfig)
    assert cfg.sigma_read == SIGMA_FG
    assert cfg.drift == 0.0          # FG retention is excellent
    assert cfg.nonlin >= 2.0         # FG update is (effectively) nonlinear
    assert cfg.n_levels >= 128       # high-precision closed-loop programming
    assert cfg.asym < 0.5            # tunneling/injection asymmetry


def test_fg_sweeps_keys_are_deviceconfig_fields():
    fields = {f.name for f in dataclasses.fields(DeviceConfig)}
    assert set(FG_SWEEPS).issubset(fields)
    for vals in FG_SWEEPS.values():
        assert len(vals) >= 2


def test_robust_k_grid_ascending_from_one():
    assert ROBUST_K_GRID[0] == 1
    assert ROBUST_K_GRID == sorted(ROBUST_K_GRID)
