"""Phase-1 transfer sim gate runner. Re-sims 02's vendored learning rules at an FG-realistic
DeviceConfig and classifies whether a custom-silicon tapeout is earned (the frozen classify_transfer).
A thin driver: no physics is reimplemented -- every model call lives in vendor_02/. Outcomes recorded,
not asserted. See docs/superpowers/specs/2026-06-25-phase1-transfer-sim-gate-design.md."""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

# Make the vendored 02 sim importable when run as a script (conftest only covers pytest).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor_02"))

from crossbar import build_full_diff, make_read_nonlinearity, make_cocontent
from manhattan import evaluate_separation
from readnoise import train_manhattan_noisy, train_coupled_noisy
from task import parity3_data, verdict as task_verdict
from verdict import classify_transfer, PASS_SNR
from fg_params import (fg_device_config, SIGMA_FG, ENDURANCE_BUDGET, FG_SWEEPS, ROBUST_K_GRID)
from fg_rules import effective_sigma, writes_per_cell
from fg_measured import device_config_from_measurements
from h5_sweep import rule_audit   # reuse the two-axis audit mapping (half=cpu,phys; full=phys,phys)


def _build(H=8, a=0.4):
    """Rebuild the parity-3 task + differential crossbar topology + read nonlinearity + co-content
    (orchestration only; all builders are vendored from 02)."""
    topo = build_full_diff(H)
    X, y = parity3_data()
    phi, dphi = make_read_nonlinearity(a)
    s_fn = make_cocontent(a)
    return topo, X, y, phi, dphi, s_fn


def _snr_at_cfg(rule, cfg, train_sigma, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
                n_epochs=300, seeds=range(10)):
    """Mean SNR (separation/floor, 02's task.verdict) over seeds for `rule` trained at DeviceConfig
    `cfg` and training read noise `train_sigma`; clean-convention eval. Fresh fixed-seed eval RNG ->
    deterministic."""
    topo, X, y, phi, dphi, s_fn = _build(H, a)
    eval_rng = np.random.default_rng(12345)
    snrs = []
    for s in seeds:
        if rule == "half":
            res = train_manhattan_noisy(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=n_epochs,
                                        seed=int(s), target=target, train_sigma=train_sigma)
        elif rule == "full":
            res = train_coupled_noisy(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=n_epochs,
                                      seed=int(s), target=target, beta=beta, s_fn=s_fn,
                                      train_sigma=train_sigma, mode="sign")
        else:
            raise ValueError(f"unknown rule {rule!r}")
        sep, floor = evaluate_separation(topo, res["g"], X, y, phi, dphi, V_drive, cfg, eval_rng)
        snrs.append(float(task_verdict(sep, floor)["snr"]))
    return float(np.mean(snrs))


def min_reads_to_pass(cfg, sigma_fg, *, K_grid=None, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
                      n_epochs=300, seeds=range(10)):
    """Robust-in-fabric search: the smallest read-averaging K (ascending grid) at which the full
    (two-read) rule reaches PASS_SNR at effective read noise sigma_fg/sqrt(K). If none clears the bar,
    return the best SNR seen and the K that produced it. Models multi-read averaging via the
    effective_sigma equivalence (fg_rules)."""
    K_grid = ROBUST_K_GRID if K_grid is None else K_grid
    best_snr, best_K = -1.0, sorted(K_grid)[-1]
    for K in sorted(K_grid):
        snr = _snr_at_cfg("full", cfg, effective_sigma(sigma_fg, K), H=H, a=a, V_drive=V_drive,
                          beta=beta, target=target, n_epochs=n_epochs, seeds=seeds)
        if snr > best_snr:
            best_snr, best_K = snr, K
        if snr >= PASS_SNR:
            return {"snr_robust": float(snr), "robust_reads_K": int(K)}
    return {"snr_robust": float(best_snr), "robust_reads_K": int(best_K)}


def _knob_sweeps(nominal_cfg, sigma_fg, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
                 n_epochs=300, seeds=range(10)):
    """Observed robustness: vary each FG knob one-at-a-time around the nominal cfg and record the
    half-rule SNR at each value (the gate object's sensitivity to each write-side non-ideality)."""
    out = {}
    for knob, vals in FG_SWEEPS.items():
        rows = []
        for v in vals:
            cfg = dataclasses.replace(nominal_cfg, **{knob: v})
            rows.append({"value": v,
                         "snr_half": _snr_at_cfg("half", cfg, sigma_fg, H=H, a=a, V_drive=V_drive,
                                                 beta=beta, target=target, n_epochs=n_epochs,
                                                 seeds=seeds)})
        out[knob] = rows
    return out


def run_transfer(*, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3, n_epochs=300, seeds=range(10),
                 sigma_fg=SIGMA_FG, endurance_budget=ENDURANCE_BUDGET, params_confirmed=False,
                 K_grid=None, cfg=None):
    """Drive the gate (half rule at the nominal FG cfg), the elevated in-fabric records (the two-read
    rule, and the multi-read-averaged robust candidate), the endurance check, and the Observed knob
    sweeps; classify with the frozen classify_transfer; record the two-axis audit for both rules.
    `cfg` defaults to the pre-registered fg_device_config(); pass a MEASURED DeviceConfig (fg_measured)
    with params_confirmed=True to re-run the same frozen gate on a real device (Phase 1b)."""
    seeds = list(seeds)
    cfg = fg_device_config() if cfg is None else cfg
    kw = dict(H=H, a=a, V_drive=V_drive, beta=beta, target=target, n_epochs=n_epochs, seeds=seeds)

    snr_half = _snr_at_cfg("half", cfg, sigma_fg, **kw)            # GATE (buildable object)
    snr_full = _snr_at_cfg("full", cfg, sigma_fg, **kw)            # in-fabric two-read at FG op point
    robust = min_reads_to_pass(cfg, sigma_fg, K_grid=K_grid, **kw)  # multi-read-averaged candidate
    wpc = writes_per_cell(n_epochs)

    verdict = classify_transfer(snr_half, snr_full, robust["snr_robust"], wpc, endurance_budget,
                                robust["robust_reads_K"], params_confirmed=params_confirmed)

    sweeps = _knob_sweeps(cfg, sigma_fg, **kw)

    audit = {}
    for rule in ("half", "full"):
        a_rec = rule_audit(rule)
        audit[rule] = {"breakdown": {f"{g}|{w}": c for (g, w), c in a_rec.breakdown().items()},
                       "is_physical": a_rec.is_physical()}

    return {
        "config": {"H": H, "a": a, "V_drive": V_drive, "beta": beta, "target": target,
                   "n_epochs": n_epochs, "n_seeds": len(seeds), "sigma_fg": float(sigma_fg),
                   "endurance_budget": int(endurance_budget)},
        "nominal_cfg": dataclasses.asdict(cfg),
        "snr_half": float(snr_half), "snr_full": float(snr_full),
        "snr_robust": float(robust["snr_robust"]), "robust_reads_K": int(robust["robust_reads_K"]),
        "writes_per_cell": int(wpc),
        "verdict": verdict, "sweeps": sweeps, "audit": audit,
    }


_QUICK = {"seeds": range(2), "n_epochs": 20, "K_grid": [1, 4]}


def _print_summary(res):
    c = res["config"]
    v = res["verdict"]
    print("=" * 78)
    print("PHASE-1 transfer sim gate — 02's rules on an FG array at FG-realistic params (parity-3)")
    print("=" * 78)
    print(f"sigma_fg={c['sigma_fg']}  n_seeds={c['n_seeds']}  n_epochs={c['n_epochs']}  "
          f"endurance_budget={c['endurance_budget']}")
    print(f"\nGATE (frozen classify_transfer, keys on the buildable HALF rule): {v['status']}")
    print(f"  snr_half={v['snr_half']:.3f}  writes_per_cell={v['writes_per_cell']}  "
          f"endurance_ok={v['endurance_ok']}  provisional={v['provisional']}")
    print(f"IN-FABRIC (elevated): two-read full={v['full_status']} (snr={v['snr_full']:.3f}); "
          f"multi-read robust={v['robust_status']} (snr={v['snr_robust']:.3f} at K={v['robust_reads_K']}); "
          f"in_fabric_achieved={v['in_fabric_achieved']}")
    print(f"audit: half is_physical={res['audit']['half']['is_physical']}  "
          f"full is_physical={res['audit']['full']['is_physical']}")
    print("\n-- Observed: half-rule SNR vs each FG knob (one-at-a-time) --")
    for knob, rows in res["sweeps"].items():
        cells = "  ".join(f"{r['value']}:{r['snr_half']:.2f}" for r in rows)
        print(f"  {knob:<12} {cells}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Phase-1 transfer sim gate for 02's rules on a floating-gate array.")
    ap.add_argument("--out", default="transfer_parity3.json")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--quick", action="store_true", help="tiny local smoke (few seeds/epochs/K)")
    ap.add_argument("--measured", default=None,
                    help="JSON of bench measurements (fg_measured.REQUIRED_MEASUREMENTS + prog_rms_error); "
                         "re-runs the frozen gate at the MEASURED DeviceConfig with params_confirmed=True")
    ap.add_argument("--write-mode", default="closed_loop", choices=["closed_loop", "open_loop"],
                    help="with --measured: which write mode the learner uses (see fg_measured)")
    ap.add_argument("--differential", action="store_true",
                    help="with --measured: injection-only part, weight is a differential pair (asym -> 1)")
    args = ap.parse_args(argv)

    extra = {}
    if args.measured:
        with open(args.measured) as fp:
            m = json.load(fp)
        cfg = device_config_from_measurements(m, write_mode=args.write_mode,
                                              differential_pair=args.differential)
        extra = dict(cfg=cfg, sigma_fg=cfg.sigma_read, params_confirmed=True)

    if args.quick:
        res = run_transfer(seeds=_QUICK["seeds"], n_epochs=_QUICK["n_epochs"],
                           beta=args.beta, K_grid=_QUICK["K_grid"], **extra)
    else:
        res = run_transfer(seeds=range(args.seeds), n_epochs=args.epochs, beta=args.beta, **extra)

    _print_summary(res)
    with open(args.out, "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"\nraw numbers -> {args.out}")
    return res


if __name__ == "__main__":
    main()
