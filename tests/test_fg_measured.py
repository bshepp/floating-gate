"""fg_measured: each inversion recovers the parameter that generated synthetic data from the VENDORED
device model (round-trips through vendor_02/memristor.py), plus the assembly + protocol table."""
import math

import numpy as np
import pytest

from memristor import DeviceConfig, apply_pulse, apply_drift, read_conductance
from fg_measured import (
    rms_error_from_levels, levels_from_rms_error, levels_from_bits, bits_from_levels,
    nonlin_from_pot_steps, nonlin_from_dep_steps, dG0_from_pot_step, asym_from_steps,
    sigma_write_from_step_std, sigma_read_from_std, drift_from_retention,
    device_config_from_measurements, REQUIRED_MEASUREMENTS, CHARACTERIZATION_PROTOCOL,
)


def _G(g, cfg):
    return (np.asarray(g) - cfg.g_min) / (cfg.g_max - cfg.g_min)


def _step(cfg, G, sign, rng):
    """One deterministic vendored pulse from normalized state G; returns the normalized step."""
    g = cfg.g_min + G * (cfg.g_max - cfg.g_min)
    g2 = apply_pulse(np.array([g]), sign, cfg, rng)
    return float(_G(g2, cfg)[0] - G)


# --- precision --------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 16, 64, 256, 1024])
def test_levels_rms_round_trip(n):
    assert levels_from_rms_error(rms_error_from_levels(n)) == n


def test_rms_error_matches_vendored_quantizer_empirically():
    """Snap-to-nearest on the vendored level grid has RMS error ~ step/sqrt(12) for uniform targets."""
    cfg = DeviceConfig(n_levels=64, nonlin=0.0, asym=1.0, sigma_write=0.0, sigma_read=0.0)
    rng = np.random.default_rng(0)
    levels = np.linspace(cfg.g_min, cfg.g_max, cfg.n_levels)
    targets = rng.uniform(cfg.g_min, cfg.g_max, size=200_000)
    snapped = levels[np.argmin(np.abs(targets[:, None] - levels), axis=1)]
    rms = np.sqrt(np.mean(((snapped - targets) / (cfg.g_max - cfg.g_min)) ** 2))
    assert rms == pytest.approx(rms_error_from_levels(64), rel=0.02)


def test_continuous_conventions():
    assert rms_error_from_levels(0) == 0.0 and rms_error_from_levels(1) == 0.0
    assert levels_from_rms_error(0.0) == 0
    assert levels_from_bits(8) == 256
    assert bits_from_levels(256) == 8.0 and bits_from_levels(0) == 0.0


# --- update shape (round-trips through the vendored apply_pulse) --------------------------------------

@pytest.mark.parametrize("nu", [0.0, 1.0, 2.0, 3.0])
def test_nonlin_and_dG0_recovered_from_potentiation_ladder(nu):
    cfg = DeviceConfig(nonlin=nu, asym=0.3, n_levels=0, sigma_write=0.0, sigma_read=0.0, dG0=0.05)
    rng = np.random.default_rng(1)
    s_lo, s_hi = _step(cfg, 0.1, +1, rng), _step(cfg, 0.6, +1, rng)
    assert nonlin_from_pot_steps(s_lo, s_hi, 0.1, 0.6) == pytest.approx(nu, abs=1e-9)
    assert dG0_from_pot_step(s_lo, 0.1, nu) == pytest.approx(cfg.dG0, rel=1e-9)


@pytest.mark.parametrize("nu", [1.0, 2.0])
def test_nonlin_recovered_from_depression_ladder(nu):
    cfg = DeviceConfig(nonlin=nu, asym=0.3, n_levels=0, sigma_write=0.0, sigma_read=0.0)
    rng = np.random.default_rng(2)
    s_lo, s_hi = _step(cfg, 0.3, -1, rng), _step(cfg, 0.9, -1, rng)
    assert nonlin_from_dep_steps(s_lo, s_hi, 0.3, 0.9) == pytest.approx(nu, abs=1e-9)


@pytest.mark.parametrize("asym", [0.2, 0.5, 1.0])
def test_asym_recovered_at_mid_and_off_mid(asym):
    cfg = DeviceConfig(nonlin=2.0, asym=asym, n_levels=0, sigma_write=0.0, sigma_read=0.0)
    rng = np.random.default_rng(3)
    for G in (0.5, 0.3):
        dep, pot = _step(cfg, G, -1, rng), _step(cfg, G, +1, rng)
        assert asym_from_steps(dep, pot, G, cfg.nonlin) == pytest.approx(asym, rel=1e-9)


def test_nonlin_clipped_at_zero_and_input_validation():
    assert nonlin_from_pot_steps(0.05, 0.06, 0.1, 0.5) == 0.0     # step GROWS with G -> clipped
    with pytest.raises(ValueError):
        nonlin_from_pot_steps(0.05, 0.04, 0.5, 0.1)
    with pytest.raises(ValueError):
        asym_from_steps(-0.01, 0.0, 0.5, 2.0)


def test_sigma_write_recovered_from_cycle_to_cycle():
    cfg = DeviceConfig(nonlin=0.0, asym=1.0, n_levels=0, sigma_write=0.3, sigma_read=0.0, dG0=0.05)
    rng = np.random.default_rng(4)
    g = np.full(50_000, 0.5 * (cfg.g_min + cfg.g_max))
    steps = _G(apply_pulse(g, +1, cfg, rng), cfg) - 0.5
    assert sigma_write_from_step_std(float(np.std(steps)), cfg.dG0) == pytest.approx(0.3, rel=0.03)


# --- read + retention ------------------------------------------------------------------------------

def test_sigma_read_recovered_from_repeated_reads():
    cfg = DeviceConfig(sigma_read=0.02)
    rng = np.random.default_rng(5)
    g = np.full(50_000, 3.0)
    reads = read_conductance(g, cfg, rng)
    assert sigma_read_from_std(float(np.std(reads)), cfg.g_max - cfg.g_min) == pytest.approx(0.02, rel=0.03)


@pytest.mark.parametrize("drift", [0.0, 0.01, 0.2])
def test_drift_recovered_from_retention(drift):
    cfg = DeviceConfig(drift=drift)
    g = np.array([cfg.g_min + 0.1 * (cfg.g_max - cfg.g_min)])
    gt = g.copy()
    for _ in range(25):
        gt = apply_drift(gt, cfg)
    G0, Gt = float(_G(g, cfg)[0]), float(_G(gt, cfg)[0])
    assert drift_from_retention(G0, Gt, 25) == pytest.approx(drift, abs=1e-9)


def test_drift_edge_cases():
    assert drift_from_retention(0.5, 0.5, 10) == 0.0          # at rest: no information
    assert drift_from_retention(0.9, 0.5, 10) == 1.0          # fully relaxed
    with pytest.raises(ValueError):
        drift_from_retention(0.9, 0.8, 0)


# --- assembly ---------------------------------------------------------------------------------------

_M = {"sigma_read": 0.008, "nonlin_raw": 3.2, "nonlin_effective": 1.5, "asym": 0.4,
      "sigma_write": 0.05, "drift": 0.0, "prog_rms_error": rms_error_from_levels(1024)}


def test_closed_loop_uses_effective_nonlin_and_landing_quantizer():
    cfg = device_config_from_measurements(_M, "closed_loop")
    assert cfg.nonlin == 1.5 and cfg.n_levels == 1024
    assert (cfg.sigma_read, cfg.asym, cfg.sigma_write, cfg.drift) == (0.008, 0.4, 0.05, 0.0)


def test_open_loop_uses_raw_nonlin_and_no_quantizer():
    cfg = device_config_from_measurements(_M, "open_loop")
    assert cfg.nonlin == 3.2 and cfg.n_levels == 0


def test_closed_loop_defaults_effective_to_raw_conservatively():
    m = {k: v for k, v in _M.items() if k != "nonlin_effective"}
    assert device_config_from_measurements(m, "closed_loop").nonlin == 3.2


def test_differential_pair_forces_symmetric_updates():
    cfg = device_config_from_measurements(_M, "open_loop", differential_pair=True)
    assert cfg.asym == 1.0 and cfg.nonlin == 3.2
    assert device_config_from_measurements(_M, "open_loop").asym == 0.4


def test_assembly_validation():
    with pytest.raises(KeyError):
        device_config_from_measurements({k: _M[k] for k in REQUIRED_MEASUREMENTS[1:]})
    with pytest.raises(KeyError):
        device_config_from_measurements({k: _M[k] for k in REQUIRED_MEASUREMENTS}, "closed_loop")
    with pytest.raises(ValueError):
        device_config_from_measurements(_M, "sideways")


def test_protocol_covers_every_device_input():
    feeds = " ".join(row["feeds"] for row in CHARACTERIZATION_PROTOCOL)
    for field in ("sigma_read", "nonlin_raw", "asym", "sigma_write", "n_levels", "drift"):
        assert field in feeds
    assert "run_mode_local_write" in feeds          # the in-fabric question is on the bench list
    names = [r["name"] for r in CHARACTERIZATION_PROTOCOL]
    assert "global erase + re-inject cycle" in names   # injection-only parts need the maintenance op
    assert "per-cell depression test" in names
    assert all({"name", "feeds", "procedure"} <= set(r) for r in CHARACTERIZATION_PROTOCOL)
