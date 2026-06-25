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
    foreclosure cliff — or None if SNR never falls below thresh in the swept range.

    Caveat: a returned 0.0 means the CLEAN baseline (train_sigma=0) is itself sub-threshold —
    the rule never learned — which reads the same as a wall at the first noisy point. Check the
    train_sigma=0 SNR to disambiguate (cf. 02's cliff_table). For the gated real run the clean
    baseline is expected to pass, so a 0.0 wall there would signal a training failure, not a wall."""
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


def _status_from_snr(snr):
    """The same PASS/BOUNDARY/FORECLOSE bands classify_h5 uses, applied to the full rule
    so its in-fabric verdict is recorded co-equally (not just as a collapse boolean)."""
    if snr >= PASS_SNR:
        return "PASS"
    if snr < FORECLOSE_SNR:
        return "FORECLOSE"
    return "BOUNDARY"


def run_h5(sigmas=None, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
           n_epochs=300, seeds=range(10), sigma_fg=SIGMA_REALISTIC_FG, sigma_confirmed=False):
    """Drive both rules over the train_sigma grid; assemble the frozen primary verdict, the
    elevated in-fabric verdict, the foreclosure walls, and the two-axis audits."""
    seeds = list(seeds)
    sigmas = list(DEFAULT_SIGMAS if sigmas is None else sigmas)
    if sigma_fg not in sigmas:
        sigmas = sorted(set(sigmas) | {sigma_fg})

    curves = {rule: sorted(sweep(rule, sigmas, H=H, a=a, V_drive=V_drive, beta=beta,
                                 target=target, n_epochs=n_epochs, seeds=seeds),
                           key=lambda r: r["train_sigma"])
              for rule in RULES}

    def _snr_at(rule, s):
        for r in curves[rule]:
            if r["train_sigma"] == s:
                return r["snr_mean"]
        raise KeyError(s)

    snr_half = _snr_at("half", sigma_fg)
    snr_full = _snr_at("full", sigma_fg)

    primary = classify_h5(snr_half, snr_full, sigma_fg=sigma_fg, sigma_confirmed=sigma_confirmed)

    walls = {rule: foreclosure_wall([r["train_sigma"] for r in curves[rule]],
                                    [r["snr_mean"] for r in curves[rule]])
             for rule in RULES}

    in_fabric = {"rule": "full_coupled_two_read", "snr_full": float(snr_full),
                 "status": _status_from_snr(snr_full), "wall": walls["full"],
                 "is_physical": True,
                 "note": "both-physical / physics-local; the read-noise-fragile rule"}

    audit = {}
    for rule in RULES:
        a_rec = rule_audit(rule)
        audit[rule] = {"breakdown": {f"{g}|{w}": c for (g, w), c in a_rec.breakdown().items()},
                       "is_physical": a_rec.is_physical()}

    return {
        "config": {"H": H, "a": a, "V_drive": V_drive, "beta": beta, "target": target,
                   "n_epochs": n_epochs, "n_seeds": len(seeds), "sigmas": sigmas,
                   "sigma_fg": float(sigma_fg)},
        "curves": curves, "walls": walls,
        "snr_half": float(snr_half), "snr_full": float(snr_full),
        "verdict": primary, "in_fabric_verdict": in_fabric, "audit": audit,
    }


_QUICK = {"sigmas": [0.0, 0.01, 0.1], "seeds": range(2), "n_epochs": 20}


def _print_summary(res):
    c = res["config"]
    half = {r["train_sigma"]: r["snr_mean"] for r in res["curves"]["half"]}
    full = {r["train_sigma"]: r["snr_mean"] for r in res["curves"]["full"]}
    print("=" * 78)
    print("PHASE-0 H5 read-noise gate — 02's rules on a floating-gate array (parity-3)")
    print("=" * 78)
    print(f"sigma_fg={c['sigma_fg']}  n_seeds={c['n_seeds']}  n_epochs={c['n_epochs']}  beta={c['beta']}")
    print("\n-- mean SNR vs train_sigma --")
    print(f"  {'train_sigma':>12} {'half(SNR)':>12} {'full(SNR)':>12}")
    for s in c["sigmas"]:
        print(f"  {s:>12} {half.get(s, float('nan')):>12.3f} {full.get(s, float('nan')):>12.3f}")
    print(f"\nforeclosure walls (train_sigma): half={res['walls']['half']}  full={res['walls']['full']}")
    v = res["verdict"]
    print(f"\nPRIMARY (frozen classify_h5, keys on the buildable HALF rule): {v['status']}")
    print(f"  snr_half={v['snr_half']:.3f}  provisional={v['provisional']}  "
          f"full_collapses_as_expected={v['full_rule_collapses_as_expected']}")
    f = res["in_fabric_verdict"]
    print(f"IN-FABRIC (elevated, the both-physical FULL rule): {f['status']}  "
          f"snr_full={f['snr_full']:.3f}  is_physical={f['is_physical']}")
    print(f"audit: half is_physical={res['audit']['half']['is_physical']}  "
          f"full is_physical={res['audit']['full']['is_physical']}  "
          f"(robust rule is NOT in-fabric; in-fabric rule is the fragile one)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Phase-0 H5 read-noise gate for 02's rules on a floating-gate array.")
    ap.add_argument("--out", default="h5_sweep_parity3.json")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sigmas", default=None,
                    help="comma-separated train_sigma grid (overrides DEFAULT_SIGMAS)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny local smoke (few sigmas/seeds/epochs)")
    args = ap.parse_args(argv)

    sigmas = [float(x) for x in args.sigmas.split(",")] if args.sigmas else None
    if args.quick:
        res = run_h5(sigmas=_QUICK["sigmas"], seeds=_QUICK["seeds"],
                     n_epochs=_QUICK["n_epochs"], beta=args.beta)
    else:
        res = run_h5(sigmas=sigmas, seeds=range(args.seeds), n_epochs=args.epochs, beta=args.beta)

    _print_summary(res)
    with open(args.out, "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"\nraw numbers -> {args.out}")
    return res


if __name__ == "__main__":
    main()
