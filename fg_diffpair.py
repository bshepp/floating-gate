"""Phase 1c: the device-faithful INJECTION-ONLY differential-pair learner (pre-registered; see
docs/superpowers/specs/2026-11-08-phase1c-diffpair-injection-only-design.md).

On the RASP / OTC2902 floating-gate family hot-electron injection is the only per-cell mechanism and it only
RAISES the pFET current; tunneling erase is GLOBAL. So a single cell cannot be depressed and the device-
faithful weight is a differential pair w = I+ - I-:
    potentiate w  -> inject the PLUS cell      (vendored potentiation pulse: dG = dG0 (1-G)^nu + noise)
    depress  w    -> inject the MINUS cell     (same mechanism, same nonlinearity -> no asymmetry)
Cells only grow. When any cell runs out of headroom (G > HEADROOM) the WHOLE array is erased and every pair
is closed-loop re-injected to its retained differential value with the smaller cell at G=0 (the
renormalization / maintenance op), landing with error renorm_sigma. The learner never computes anything in
that op -- it preserves w -- but it IS a host-targeted write and is recorded separately in the audit.

Hypothesis H6 (pre-registered): with renormalization both cells sit near the bottom of their range where the
raw update factor (1-G)^nu ~ 1, so the pair LINEARIZES the raw nonlinearity that foreclosed the single-cell
model at nu >= 3 (Phase 1: half-rule SNR 0.20 at nu=3). Gate: the frozen classify_transfer (unchanged) on
the half rule at the pre-registered nominal DIFFPAIR_NOMINAL (raw nu=3). 08-authored; composes vendored
primitives; never edits vendor_02/.
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor_02"))

from memristor import apply_pulse, apply_drift                      # vendored device physics
from readnoise import noisy_read                                    # vendored explicit-sigma read
from manhattan import init_conductances, monolithic_grad, evaluate_separation
from coupled import free_voltages, nudged_solve, _SIGMA
from crossbar import build_full_diff, make_read_nonlinearity, make_cocontent, forward
from task import parity3_data, separation, verdict as task_verdict
from verdict import classify_transfer, PASS_SNR
from fg_params import fg_device_config, SIGMA_FG, ENDURANCE_BUDGET, ROBUST_K_GRID
from fg_rules import effective_sigma
from fg_measured import rms_error_from_levels
from audit import Audit, CPU, PHYSICAL

# --- pre-registered constants (committed before the run) --------------------------------------------
HEADROOM = 0.9                 # erase + re-inject the array when any cell's normalized charge exceeds this
RENORM_LEVELS = 256            # closed-loop re-inject lands with the 8-bit landing error of the Phase-1 nominal
NU_RAW_NOMINAL = 3.0           # the raw FG nonlinearity that FORECLOSED the single-cell half rule (Phase 1)
NU_SWEEP = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
HEADROOM_SWEEP = [0.3, 0.6, 0.9]
DIFFPAIR_K_GRID = [1, 100, 10_000]   # robust-in-fabric K scan (coarse; Phase 1 foreclosed at every K)


def diffpair_nominal():
    """The pre-registered operating point: the Phase-1 FG nominal with the RAW nonlinearity (nu=3), open-loop
    pulses (n_levels=0: a single injection pulse is not quantized; the quantizer belongs to the closed-loop
    re-inject, modeled by renorm_sigma) and asym=1 (both directions are injection)."""
    return dataclasses.replace(fg_device_config(), nonlin=NU_RAW_NOMINAL, asym=1.0, n_levels=0)


def renorm_sigma_default(n_levels=RENORM_LEVELS):
    """Landing error (fraction of a cell's span) of the closed-loop re-inject."""
    return rms_error_from_levels(n_levels)


def pair_read_sigma(sigma):
    """Read noise of w=(G+ - G-) in units of the EFFECTIVE span: two independent cell reads at sigma each,
    combined with weight 1/2 -> sigma/sqrt(2)."""
    return float(sigma) / float(np.sqrt(2.0))


# --- pair representation ----------------------------------------------------------------------------

def pair_effective_g(Gp, Gm, cfg):
    """Effective conductance of the pair: D = G+ - G- in [-1, 1] mapped onto [g_min, g_max]."""
    D = np.asarray(Gp, float) - np.asarray(Gm, float)
    return cfg.g_min + 0.5 * (D + 1.0) * (cfg.g_max - cfg.g_min)


def pair_from_g(g, cfg):
    """The renormalized pair that realizes g: the smaller cell at 0 (maximal headroom)."""
    D = 2.0 * (np.asarray(g, float) - cfg.g_min) / (cfg.g_max - cfg.g_min) - 1.0
    return np.maximum(D, 0.0), np.maximum(-D, 0.0)


def inject(G, cfg, rng):
    """One open-loop injection pulse on a CELL array (normalized charge G in [0,1]): the vendored potentiation
    branch dG = dG0 (1-G)^nu (+ cycle-to-cycle noise), no quantizer, no asymmetry (depression never fires)."""
    cell_cfg = dataclasses.replace(cfg, n_levels=0, asym=1.0)
    g_cell = cfg.g_min + np.asarray(G, float) * (cfg.g_max - cfg.g_min)
    g_new = apply_pulse(g_cell, +1, cell_cfg, rng)
    return (g_new - cfg.g_min) / (cfg.g_max - cfg.g_min)


def needs_renorm(Gp, Gm, headroom=HEADROOM):
    """True iff ANY cell has exhausted its headroom (then the whole array is erased: tunneling is global)."""
    return bool(np.any(Gp > headroom) or np.any(Gm > headroom))


def renormalize(Gp, Gm, cfg, rng, renorm_sigma):
    """Global erase + closed-loop re-inject of every pair to its retained differential value, smaller cell at
    0, each cell landing with N(0, renorm_sigma) error (clipped to [0,1]). Preserves w up to that error."""
    g = pair_effective_g(Gp, Gm, cfg)
    Gp2, Gm2 = pair_from_g(g, cfg)
    if renorm_sigma > 0:
        Gp2 = Gp2 + renorm_sigma * rng.normal(size=Gp2.shape)
        Gm2 = Gm2 + renorm_sigma * rng.normal(size=Gm2.shape)
    return np.clip(Gp2, 0.0, 1.0), np.clip(Gm2, 0.0, 1.0)


# --- the two rules on the pair ---------------------------------------------------------------------

def _apply_signed_pulses(Gp, Gm, pot, dep, cfg, rng, inj_count):
    """Potentiate -> inject plus cells; depress -> inject minus cells. Counts injections per pair."""
    Gp, Gm = Gp.copy(), Gm.copy()
    if np.any(pot):
        Gp[pot] = inject(Gp[pot], cfg, rng)
    if np.any(dep):
        Gm[dep] = inject(Gm[dep], cfg, rng)
    inj_count = inj_count + pot.astype(int) + dep.astype(int)
    return Gp, Gm, inj_count


def half_signs(topo, g_eff, X, y, phi, dphi, V_drive, cfg, rng, train_sigma, target):
    """Half-co-located rule: ONE noisy read of the effective weights, off-array CPU gradient sign."""
    g_meas = noisy_read(g_eff, pair_read_sigma(train_sigma), cfg, rng)
    grad = monolithic_grad(topo, g_meas, X, y, phi, dphi, V_drive, target=target)
    s = np.sign(grad)
    return s < 0, s > 0            # pot (grad<0 -> raise g), dep


def full_signs(topo, g_eff, X, y, phi, dphi, V_drive, cfg, rng, train_sigma, beta, target, s_fn):
    """Fully-co-located two-read contrastive rule (02's coupled sign rule) on the effective weights."""
    g_free = noisy_read(g_eff, pair_read_sigma(train_sigma), cfg, rng)
    g_nudge = noisy_read(g_eff, pair_read_sigma(train_sigma), cfg, rng)
    Vf = free_voltages(topo, g_free, X, phi, dphi, V_drive)
    Vn = nudged_solve(topo, g_nudge, X, y, phi, dphi, V_drive, beta, target, Vf=Vf)
    dg = np.zeros(topo.n_params)
    for (a, b, idx) in topo.param_edges:
        sf = np.asarray(s_fn(Vf[:, a] - Vf[:, b]), float)
        sn = np.asarray(s_fn(Vn[:, a] - Vn[:, b]), float)
        dg[idx] = _SIGMA * float(np.sum(sn - sf)) / beta
    s = np.sign(dg)
    return s > 0, s < 0


def train_diffpair(rule, topo, X, y, phi, dphi, V_drive, cfg, *, n_epochs=300, seed=0, target=0.3,
                   train_sigma=0.0, beta=0.1, s_fn=None, headroom=HEADROOM, renorm_sigma=None):
    """Train the pair array with `rule` in {"half","full"}; eval clean-convention (cfg.sigma_read) as in
    02/08. Returns the best effective g, SNR inputs, injection + erase counts. Deterministic in seed."""
    renorm_sigma = renorm_sigma_default() if renorm_sigma is None else float(renorm_sigma)
    if s_fn is None:
        s_fn = make_cocontent(0.4)
    rng = np.random.default_rng(seed)
    g0 = init_conductances(topo, rng, cfg)                 # same unprogrammed start as 02
    Gp, Gm = pair_from_g(g0, cfg)
    inj = np.zeros(topo.n_params, dtype=int)
    n_erases = 0
    curve, best_sep, best_g = [], -np.inf, g0.copy()
    for _ in range(n_epochs):
        g_eff = pair_effective_g(Gp, Gm, cfg)
        if rule == "half":
            pot, dep = half_signs(topo, g_eff, X, y, phi, dphi, V_drive, cfg, rng, train_sigma, target)
        elif rule == "full":
            pot, dep = full_signs(topo, g_eff, X, y, phi, dphi, V_drive, cfg, rng, train_sigma, beta,
                                  target, s_fn)
        else:
            raise ValueError(f"unknown rule {rule!r}")
        Gp, Gm, inj = _apply_signed_pulses(Gp, Gm, pot, dep, cfg, rng, inj)
        if cfg.drift > 0:                                   # retention on each cell (FG: ~0)
            span = cfg.g_max - cfg.g_min
            Gp = (apply_drift(cfg.g_min + Gp * span, cfg) - cfg.g_min) / span
            Gm = (apply_drift(cfg.g_min + Gm * span, cfg) - cfg.g_min) / span
        if needs_renorm(Gp, Gm, headroom):
            Gp, Gm = renormalize(Gp, Gm, cfg, rng, renorm_sigma)
            n_erases += 1
        g = pair_effective_g(Gp, Gm, cfg)
        ep_sep = separation(forward(topo, g, X, phi, dphi, V_drive), y)
        curve.append(float(ep_sep))
        if ep_sep > best_sep:
            best_sep, best_g = ep_sep, g.copy()
    return {"g": best_g, "curve": curve, "n_erases": int(n_erases),
            "max_injections_per_cell": int(inj.max()) if inj.size else 0,
            "renorm_writes": int(n_erases) * int(topo.n_params)}


# --- runner -----------------------------------------------------------------------------------------

def _build(H=8, a=0.4):
    topo = build_full_diff(H)
    X, y = parity3_data()
    phi, dphi = make_read_nonlinearity(a)
    return topo, X, y, phi, dphi, make_cocontent(a)


def snr_at(rule, cfg, train_sigma, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3, n_epochs=300,
           seeds=range(10), headroom=HEADROOM, renorm_sigma=None):
    """Mean SNR over seeds (02's task.verdict) + mean erase / injection counts for the pair learner."""
    topo, X, y, phi, dphi, s_fn = _build(H, a)
    eval_rng = np.random.default_rng(12345)
    snrs, erases, injs = [], [], []
    for s in seeds:
        res = train_diffpair(rule, topo, X, y, phi, dphi, V_drive, cfg, n_epochs=n_epochs, seed=int(s),
                             target=target, train_sigma=train_sigma, beta=beta, s_fn=s_fn,
                             headroom=headroom, renorm_sigma=renorm_sigma)
        sep, floor = evaluate_separation(topo, res["g"], X, y, phi, dphi, V_drive, cfg, eval_rng)
        snrs.append(float(task_verdict(sep, floor)["snr"]))
        erases.append(res["n_erases"])
        injs.append(res["max_injections_per_cell"])
    return {"snr": float(np.mean(snrs)), "n_erases": float(np.mean(erases)),
            "max_injections_per_cell": float(np.mean(injs))}


def rule_audit(rule, n_learning_updates, n_renorm_writes):
    """Two-axis audit for the pair learner: learning updates as in Phase 0/1 (half = cpu,physical; full =
    physical,physical) PLUS the renormalization re-inject, a host-TARGETED physical write that computes no
    gradient (recorded as (cpu, physical) so a both-physical claim cannot hide it)."""
    a = Audit()
    a.record(CPU if rule == "half" else PHYSICAL, PHYSICAL, n_updates=max(1, int(n_learning_updates)))
    if n_renorm_writes > 0:
        a.record(CPU, PHYSICAL, n_updates=int(n_renorm_writes))
    return a


def run_diffpair(*, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3, n_epochs=300, seeds=range(10),
                 sigma_fg=SIGMA_FG, endurance_budget=ENDURANCE_BUDGET, K_grid=None, cfg=None,
                 nu_sweep=None, headroom_sweep=None, params_confirmed=False):
    """H6 gate + elevated records + sweeps, classified by the FROZEN classify_transfer."""
    seeds = list(seeds)
    cfg = diffpair_nominal() if cfg is None else cfg
    K_grid = DIFFPAIR_K_GRID if K_grid is None else list(K_grid)
    nu_sweep = NU_SWEEP if nu_sweep is None else list(nu_sweep)
    headroom_sweep = HEADROOM_SWEEP if headroom_sweep is None else list(headroom_sweep)
    kw = dict(H=H, a=a, V_drive=V_drive, beta=beta, target=target, n_epochs=n_epochs, seeds=seeds)

    half = snr_at("half", cfg, sigma_fg, **kw)                       # GATE (H6)
    full = snr_at("full", cfg, sigma_fg, **kw)                       # in-fabric two-read on the pair
    best = {"snr_robust": -1.0, "robust_reads_K": K_grid[-1]}
    for K in sorted(K_grid):
        r = snr_at("full", cfg, effective_sigma(sigma_fg, K), **kw)
        if r["snr"] > best["snr_robust"]:
            best = {"snr_robust": r["snr"], "robust_reads_K": int(K)}
        if r["snr"] >= PASS_SNR:
            break
    # oxide stress per cell over a run: its own injections + one tunneling stress per global erase
    wpc = int(round(half["max_injections_per_cell"] + half["n_erases"]))
    verdict = classify_transfer(half["snr"], full["snr"], best["snr_robust"], wpc, endurance_budget,
                                best["robust_reads_K"], params_confirmed=params_confirmed)

    sweeps = {"nonlin": [], "headroom": []}
    for nu in nu_sweep:
        r = snr_at("half", dataclasses.replace(cfg, nonlin=nu), sigma_fg, **kw)
        sweeps["nonlin"].append({"value": nu, "snr_half": r["snr"], "n_erases": r["n_erases"]})
    for hr in headroom_sweep:
        r = snr_at("half", cfg, sigma_fg, headroom=hr, **kw)
        sweeps["headroom"].append({"value": hr, "snr_half": r["snr"], "n_erases": r["n_erases"]})

    topo = build_full_diff(H)
    audit = {}
    for rule, rec in (("half", half), ("full", full)):
        a_rec = rule_audit(rule, n_epochs * topo.n_params, rec["n_erases"] * topo.n_params)
        audit[rule] = {"breakdown": {f"{g}|{w}": c for (g, w), c in a_rec.breakdown().items()},
                       "is_physical": a_rec.is_physical()}
    return {
        "config": {"H": H, "a": a, "V_drive": V_drive, "beta": beta, "target": target,
                   "n_epochs": n_epochs, "n_seeds": len(seeds), "sigma_fg": float(sigma_fg),
                   "endurance_budget": int(endurance_budget), "headroom": HEADROOM,
                   "renorm_sigma": renorm_sigma_default(), "K_grid": K_grid},
        "nominal_cfg": dataclasses.asdict(cfg),
        "half": half, "full": full, "robust": best, "writes_per_cell": wpc,
        "verdict": verdict, "sweeps": sweeps, "audit": audit,
    }


_QUICK = {"seeds": range(2), "n_epochs": 20, "K_grid": [1, 4], "nu_sweep": [0.0, 3.0],
          "headroom_sweep": [0.9]}


def _print_summary(res):
    c, v = res["config"], res["verdict"]
    print("=" * 78)
    print("PHASE-1c H6 — injection-only DIFFERENTIAL-PAIR learner at RAW FG nonlinearity (parity-3)")
    print("=" * 78)
    print(f"sigma_fg={c['sigma_fg']}  n_seeds={c['n_seeds']}  n_epochs={c['n_epochs']}  "
          f"headroom={c['headroom']}  renorm_sigma={c['renorm_sigma']:.4f}  nu_raw={res['nominal_cfg']['nonlin']}")
    print(f"\nGATE (frozen classify_transfer, half rule on the pair): {v['status']}")
    print(f"  snr_half={v['snr_half']:.3f}  erases/run={res['half']['n_erases']:.1f}  "
          f"max_inj/cell={res['half']['max_injections_per_cell']:.0f}  writes_per_cell={v['writes_per_cell']}  "
          f"endurance_ok={v['endurance_ok']}  provisional={v['provisional']}")
    print(f"IN-FABRIC (elevated): two-read full={v['full_status']} (snr={v['snr_full']:.3f}); "
          f"robust={v['robust_status']} (snr={v['snr_robust']:.3f} at K={v['robust_reads_K']}); "
          f"in_fabric_achieved={v['in_fabric_achieved']}")
    print(f"audit: half is_physical={res['audit']['half']['is_physical']}  "
          f"full is_physical={res['audit']['full']['is_physical']}  (renorm writes are (cpu, physical))")
    print("\n-- Observed: half-rule SNR (and erases/run) vs raw nu, pair model --")
    print("  " + "  ".join(f"{r['value']}:{r['snr_half']:.2f}({r['n_erases']:.0f})" for r in res["sweeps"]["nonlin"]))
    print("-- Observed: half-rule SNR (and erases/run) vs headroom --")
    print("  " + "  ".join(f"{r['value']}:{r['snr_half']:.2f}({r['n_erases']:.0f})" for r in res["sweeps"]["headroom"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase-1c H6: injection-only differential-pair FG learner.")
    ap.add_argument("--out", default="diffpair_parity3.json")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--quick", action="store_true", help="tiny local smoke")
    args = ap.parse_args(argv)
    if args.quick:
        res = run_diffpair(beta=args.beta, **_QUICK)
    else:
        res = run_diffpair(seeds=range(args.seeds), n_epochs=args.epochs, beta=args.beta)
    _print_summary(res)
    with open(args.out, "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"\nraw numbers -> {args.out}")
    return res


if __name__ == "__main__":
    main()
