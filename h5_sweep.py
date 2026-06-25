"""Phase-0 H5 read-noise gate for 02's in-fabric rules on a floating-gate array.

A THIN DRIVER over the vendored 02 sim (vendor_02/, ported @95868e7): no physics is
reimplemented. For each (rule, train_sigma) it trains the vendored noisy rule, evaluates
clean-convention separation/floor, and forms SNR = separation/floor (02's task.verdict
definition). It records the FROZEN primary verdict (classify_h5, keyed on the buildable
half-co-located rule) PLUS an elevated co-equal in-fabric verdict (the both-physical
fully-co-located rule) PLUS the two-axis audit for both rules. Outcomes recorded, not asserted.
See docs/superpowers/specs/2026-06-24-phase0-h5-sweep-runner-design.md."""
import argparse
import json
import os
import sys

import numpy as np

# Make the vendored 02 sim importable when run as a script (conftest only covers pytest).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor_02"))

from crossbar import build_full_diff, make_read_nonlinearity, make_cocontent
from manhattan import evaluate_separation
from readnoise import train_manhattan_noisy, train_coupled_noisy
from memristor import DeviceConfig
from task import parity3_data, verdict as task_verdict
from audit import Audit, PHYSICAL, CPU
from verdict import classify_h5, SIGMA_REALISTIC_FG, PASS_SNR, FORECLOSE_SNR

RULES = ("half", "full")
# fraction-of-span training read-noise grid; includes sigma_fg=0.01 and brackets it below/above.
DEFAULT_SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def rule_audit(rule):
    """Two-axis provenance for a rule. half = off-array CPU gradient + physical write
    (NOT both-physical); full = in-fabric local gradient + physical write (both-physical)."""
    a = Audit()
    if rule == "half":
        a.record(CPU, PHYSICAL)
    elif rule == "full":
        a.record(PHYSICAL, PHYSICAL)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return a


def foreclosure_wall(sigmas, snr_means, thresh=FORECLOSE_SNR):
    """First train_sigma (scanned in ascending sigma order) whose mean SNR < thresh — the
    foreclosure cliff — or None if SNR never falls below thresh in the swept range."""
    for s, snr in sorted(zip(sigmas, snr_means), key=lambda p: p[0]):
        if snr < thresh:
            return float(s)
    return None


def _build(H=8, a=0.4):
    """Rebuild the parity-3 task + differential crossbar topology + read nonlinearity +
    co-content (orchestration only; all builders are vendored from 02)."""
    topo = build_full_diff(H)
    X, y = parity3_data()
    phi, dphi = make_read_nonlinearity(a)
    s_fn = make_cocontent(a)
    return topo, X, y, phi, dphi, s_fn


def _cell(rule, train_sigma, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
          n_epochs=300, seeds=range(10)):
    """One (rule, train_sigma) cell. Trains each seed's noisy rule, evaluates clean-convention
    (sep, floor), and forms per-seed SNR via 02's task.verdict; returns the aggregated row.
    The eval RNG is a fresh fixed-seed generator -> pure/deterministic (mirrors 02's sweep)."""
    topo, X, y, phi, dphi, s_fn = _build(H, a)
    cfg = DeviceConfig()
    eval_rng = np.random.default_rng(12345)
    snrs, seps, audit = [], [], None
    for s in seeds:
        if rule == "half":
            res = train_manhattan_noisy(topo, X, y, phi, dphi, V_drive, cfg,
                                        n_epochs=n_epochs, seed=int(s), target=target,
                                        train_sigma=train_sigma)
        elif rule == "full":
            res = train_coupled_noisy(topo, X, y, phi, dphi, V_drive, cfg,
                                      n_epochs=n_epochs, seed=int(s), target=target,
                                      beta=beta, s_fn=s_fn, train_sigma=train_sigma, mode="sign")
        else:
            raise ValueError(f"unknown rule {rule!r}")
        audit = res["audit"]
        sep, floor = evaluate_separation(topo, res["g"], X, y, phi, dphi, V_drive, cfg, eval_rng)
        snrs.append(float(task_verdict(sep, floor)["snr"]))
        seps.append(float(sep))
    return {"rule": rule, "train_sigma": float(train_sigma),
            "snr_mean": float(np.mean(snrs)), "snr_per_seed": snrs,
            "sep_mean": float(np.mean(seps)), "audit": audit}


def sweep(rule, sigmas, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
          n_epochs=300, seeds=range(10)):
    """Run `rule` over the train_sigma grid (serial); one SNR row per sigma, input order."""
    seeds = list(seeds)
    return [_cell(rule, s, H=H, a=a, V_drive=V_drive, beta=beta, target=target,
                  n_epochs=n_epochs, seeds=seeds) for s in sigmas]
