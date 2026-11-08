"""fg_diffpair: pair representation round-trips, injection == the vendored potentiation branch, global
erase + re-inject preserves w, both directions symmetric, trainer/runner wiring + determinism. Structure
only (no science numbers -- those are Observed in the run)."""
import dataclasses
import json

import numpy as np
import pytest

from memristor import apply_pulse
from fg_diffpair import (
    diffpair_nominal, renorm_sigma_default, pair_read_sigma, pair_effective_g, pair_from_g, inject,
    needs_renorm, renormalize, train_diffpair, snr_at, rule_audit, run_diffpair, main, _build,
    HEADROOM, NU_RAW_NOMINAL,
)


def test_nominal_is_raw_nu_open_loop_symmetric():
    cfg = diffpair_nominal()
    assert cfg.nonlin == NU_RAW_NOMINAL == 3.0 and cfg.asym == 1.0 and cfg.n_levels == 0
    assert 0 < renorm_sigma_default() < 0.01
    assert pair_read_sigma(0.01) == pytest.approx(0.01 / np.sqrt(2))


def test_pair_round_trip_and_renormalized_form():
    cfg = diffpair_nominal()
    g = np.linspace(cfg.g_min, cfg.g_max, 11)
    Gp, Gm = pair_from_g(g, cfg)
    assert np.allclose(pair_effective_g(Gp, Gm, cfg), g)
    assert np.all(np.minimum(Gp, Gm) == 0.0)                 # smaller cell at 0 (max headroom)
    assert np.all((Gp >= 0) & (Gp <= 1) & (Gm >= 0) & (Gm <= 1))


def test_inject_is_the_vendored_potentiation_branch_without_quantizer_or_asym():
    cfg = dataclasses.replace(diffpair_nominal(), n_levels=64, asym=0.2, sigma_write=0.0)
    G = np.array([0.0, 0.3, 0.7])
    got = inject(G, cfg, np.random.default_rng(0))
    ref_cfg = dataclasses.replace(cfg, n_levels=0, asym=1.0)
    g_cell = cfg.g_min + G * (cfg.g_max - cfg.g_min)
    ref = (apply_pulse(g_cell, +1, ref_cfg, np.random.default_rng(0)) - cfg.g_min) / (cfg.g_max - cfg.g_min)
    assert np.allclose(got, ref)
    assert np.all(got > G)                                   # injection only ever raises charge
    assert got[0] - G[0] > got[2] - G[2]                     # nu=3: step shrinks with G (the raw wall)


def test_depress_is_injecting_minus_cell_and_is_symmetric_with_potentiate():
    cfg = dataclasses.replace(diffpair_nominal(), sigma_write=0.0)
    rng = np.random.default_rng(0)
    Gp, Gm = np.array([0.2]), np.array([0.2])
    up = pair_effective_g(inject(Gp, cfg, rng), Gm, cfg) - pair_effective_g(Gp, Gm, cfg)
    down = pair_effective_g(Gp, inject(Gm, cfg, rng), cfg) - pair_effective_g(Gp, Gm, cfg)
    assert up > 0 and down < 0 and up == pytest.approx(-down)


def test_renorm_triggers_on_headroom_and_preserves_w():
    cfg = diffpair_nominal()
    Gp, Gm = np.array([0.95, 0.1]), np.array([0.2, 0.0])
    assert needs_renorm(Gp, Gm, HEADROOM) and not needs_renorm(Gp * 0.5, Gm, HEADROOM)
    g_before = pair_effective_g(Gp, Gm, cfg)
    Gp2, Gm2 = renormalize(Gp, Gm, cfg, np.random.default_rng(0), renorm_sigma=0.0)
    assert np.allclose(pair_effective_g(Gp2, Gm2, cfg), g_before)   # exact with no landing error
    assert np.all(np.minimum(Gp2, Gm2) == 0.0)
    Gp3, Gm3 = renormalize(Gp, Gm, cfg, np.random.default_rng(0), renorm_sigma=0.01)
    assert np.all((Gp3 >= 0) & (Gp3 <= 1)) and not np.allclose(Gp3, Gp2)


def test_train_is_deterministic_counts_erases_and_returns_keys():
    topo, X, y, phi, dphi, s_fn = _build()
    cfg = dataclasses.replace(diffpair_nominal(), dG0=0.3)   # big pulses -> cells saturate -> erases happen
    kw = dict(n_epochs=15, seed=3, train_sigma=0.01, s_fn=s_fn, headroom=0.5)
    a = train_diffpair("half", topo, X, y, phi, dphi, 1.5, cfg, **kw)
    b = train_diffpair("half", topo, X, y, phi, dphi, 1.5, cfg, **kw)
    assert set(a) == {"g", "curve", "n_erases", "max_injections_per_cell", "renorm_writes"}
    assert np.array_equal(a["g"], b["g"]) and a["n_erases"] == b["n_erases"]
    assert a["n_erases"] >= 1 and a["renorm_writes"] == a["n_erases"] * topo.n_params
    assert np.all((a["g"] >= cfg.g_min) & (a["g"] <= cfg.g_max))
    f = train_diffpair("full", topo, X, y, phi, dphi, 1.5, cfg, n_epochs=3, seed=0, s_fn=s_fn)
    assert len(f["curve"]) == 3
    with pytest.raises(ValueError):
        train_diffpair("sideways", topo, X, y, phi, dphi, 1.5, cfg, n_epochs=1, s_fn=s_fn)


def test_snr_at_shape():
    r = snr_at("half", diffpair_nominal(), 0.01, n_epochs=5, seeds=range(1))
    assert set(r) == {"snr", "n_erases", "max_injections_per_cell"} and r["snr"] >= 0


def test_rule_audit_renorm_writes_break_both_physical():
    assert rule_audit("full", 100, 0).is_physical() is True
    assert rule_audit("full", 100, 10).is_physical() is False     # the maintenance op is host-targeted
    assert rule_audit("half", 100, 0).is_physical() is False


def test_run_diffpair_assembles_and_main_quick_writes_json(tmp_path):
    out = tmp_path / "d.json"
    res = main(["--quick", "--out", str(out)])
    data = json.loads(out.read_text())
    assert res["verdict"]["status"] in {"PASS", "BOUNDARY", "FORECLOSE"}
    assert res["verdict"]["snr_half"] == res["half"]["snr"]
    assert res["verdict"]["provisional"] is True
    assert {"nonlin", "headroom"} == set(res["sweeps"]) and len(res["sweeps"]["nonlin"]) == 2
    assert data["config"]["n_seeds"] == 2 and data["audit"]["half"]["is_physical"] is False
    assert res["writes_per_cell"] == res["verdict"]["writes_per_cell"]
