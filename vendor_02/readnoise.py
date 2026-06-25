# PORTED from 02-memristive-crossbar/readnoise.py @95868e7 — do not edit to diverge; re-vendor on upstream change.
"""Stage 5 -- read-noise amplification under NOISY TRAINING relaxations. The first study to depart from
the clean-training convention of Verdicts #1-#3: read noise (train_sigma) is injected into the training
relaxations themselves, for BOTH rules, behind a default-off knob (train_sigma=0 reproduces the clean
rules bit-for-bit). Eval is held at cfg.sigma_read (clean-eval convention). Coupled draws TWO independent
noisy reads per step (free+nudged); Manhattan draws ONE (forward) -- faithful device physics, part of the
hypothesis. Outcomes recorded, not asserted.
See docs/superpowers/specs/2026-06-21-stage5-readnoise-amplification-design.md."""
import argparse
import json
import multiprocessing as mp

import numpy as np

from crossbar import build_full_diff, make_read_nonlinearity, make_cocontent, forward
from coupled import free_voltages, nudged_solve, _SIGMA
from manhattan import init_conductances, evaluate_separation, monolithic_grad
from memristor import apply_pulse, apply_drift, DeviceConfig
from task import parity3_data, separation, verdict


def noisy_read(g, sigma, cfg, rng):
    """A read-noise draw with an EXPLICIT sigma (the training read noise), independent of cfg.sigma_read
    (the eval read noise). g + sigma*span*N(0,1), clipped to bounds. sigma<=0 -> clean copy, no rng draw."""
    g = np.asarray(g, float)
    if sigma <= 0:
        return g.copy()
    span = cfg.g_max - cfg.g_min
    g_read = g + sigma * span * rng.normal(size=g.shape)
    return np.clip(g_read, cfg.g_min, cfg.g_max)


def manhattan_step_noisy(topo, g, X, y, phi, dphi, V_drive, cfg, rng, target=0.3, train_sigma=0.0):
    """One Manhattan step with the OFF-ARRAY gradient computed from ONE noisy forward read; the sign pulse
    hits the TRUE g (read noise corrupts the measurement, not the state). train_sigma=0 == manhattan_step."""
    g_meas = noisy_read(g, train_sigma, cfg, rng)                 # one physical forward read
    grad = monolithic_grad(topo, g_meas, X, y, phi, dphi, V_drive, target=target)
    sign = np.sign(grad)
    g_new = g.copy()
    pot = sign < 0
    dep = sign > 0
    if np.any(pot):
        g_new[pot] = apply_pulse(g[pot], +1, cfg, rng)
    if np.any(dep):
        g_new[dep] = apply_pulse(g[dep], -1, cfg, rng)
    audit = {"rule": "manhattan_sign", "error_site": "off_array_cpu", "increment_site": "physical_pulse"}
    return g_new, audit


def _mag_scale(dg):
    """Mean-normalized, 3x-capped per-edge step magnitude (Effect B). Mean step == the sign rule's; only
    the per-edge distribution carries magnitude info. Zero vector -> all-ones (nominal step)."""
    m = np.abs(np.asarray(dg, float))
    denom = float(np.mean(m))
    if denom <= 1e-12:
        return np.ones_like(m)
    return np.clip(m / denom, 0.0, 3.0)


def coupled_step_noisy(topo, g, X, y, phi, dphi, V_drive, cfg, rng, s_fn, beta=0.1, target=0.3,
                       train_sigma=0.0, mode="sign"):
    """One coupled-learning step under noisy training. TWO independent physical reads: free over g_free,
    nudged over g_nudge (the nudge err uses the noisy free output). Local per-edge co-content difference
    dg = _SIGMA/beta * sum(s(dV^N) - s(dV^F)); no CPU gradient. mode='sign' -> fixed-amplitude sign pulse
    (== coupled.coupled_step at train_sigma=0). mode='magnitude' -> mean-normalized variable-amplitude pulse
    (Effect B). The pulse hits the TRUE g."""
    g_free = noisy_read(g, train_sigma, cfg, rng)                # read #1 (free relaxation)
    g_nudge = noisy_read(g, train_sigma, cfg, rng)               # read #2 (nudged relaxation, independent)
    Vf = free_voltages(topo, g_free, X, phi, dphi, V_drive)
    Vn = nudged_solve(topo, g_nudge, X, y, phi, dphi, V_drive, beta, target, Vf=Vf)
    dg = np.zeros(topo.n_params)
    for (a, b, idx) in topo.param_edges:
        sf = np.asarray(s_fn(Vf[:, a] - Vf[:, b]), float)
        sn = np.asarray(s_fn(Vn[:, a] - Vn[:, b]), float)
        dg[idx] = _SIGMA * float(np.sum(sn - sf)) / beta
    s = np.sign(dg)
    g_new = g.copy()
    pot = s > 0
    dep = s < 0
    scale = _mag_scale(dg) if mode == "magnitude" else None
    if np.any(pot):
        g_new[pot] = apply_pulse(g[pot], +1, cfg, rng, scale=(scale[pot] if scale is not None else 1.0))
    if np.any(dep):
        g_new[dep] = apply_pulse(g[dep], -1, cfg, rng, scale=(scale[dep] if scale is not None else 1.0))
    tag = "coupled_learning_mag" if mode == "magnitude" else "coupled_learning"
    audit = {"rule": tag, "error_site": "on_array_local", "increment_site": "physical_pulse"}
    return g_new, audit


def train_manhattan_noisy(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=300, seed=0, target=0.3,
                          train_sigma=0.0):
    """Full-batch Manhattan training with noisy training reads. Eval is CLEAN-convention (cfg.sigma_read).
    train_sigma=0 reproduces manhattan.train. Outcome reported, not asserted."""
    rng = np.random.default_rng(seed)
    g = init_conductances(topo, rng, cfg)
    curve, audit = [], None
    best_sep, best_g = -np.inf, g.copy()
    for _ in range(n_epochs):
        g, audit = manhattan_step_noisy(topo, g, X, y, phi, dphi, V_drive, cfg, rng,
                                        target=target, train_sigma=train_sigma)
        g = apply_drift(g, cfg)
        np.clip(g, cfg.g_min, cfg.g_max, out=g)
        ep_sep = separation(forward(topo, g, X, phi, dphi, V_drive), y)
        curve.append(ep_sep)
        if ep_sep > best_sep:
            best_sep, best_g = ep_sep, g.copy()
    sep_mean, _ = evaluate_separation(topo, best_g, X, y, phi, dphi, V_drive, cfg, rng)
    final_mean, _ = evaluate_separation(topo, g, X, y, phi, dphi, V_drive, cfg, rng)
    return {"g": best_g, "separation": float(sep_mean), "final_separation": float(final_mean),
            "curve": curve, "audit": audit}


def train_coupled_noisy(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=300, seed=0, target=0.3,
                        beta=0.1, s_fn=None, train_sigma=0.0, mode="sign"):
    """Full-batch coupled-learning training with noisy training reads (two per step). Eval is CLEAN-
    convention. train_sigma=0, mode='sign' reproduces coupled.train_coupled. Outcome reported, not asserted."""
    if s_fn is None:
        s_fn = make_cocontent(0.4)
    rng = np.random.default_rng(seed)
    g = init_conductances(topo, rng, cfg)
    curve, audit = [], None
    best_sep, best_g = -np.inf, g.copy()
    for _ in range(n_epochs):
        g, audit = coupled_step_noisy(topo, g, X, y, phi, dphi, V_drive, cfg, rng, s_fn,
                                      beta=beta, target=target, train_sigma=train_sigma, mode=mode)
        g = apply_drift(g, cfg)
        np.clip(g, cfg.g_min, cfg.g_max, out=g)
        ep_sep = separation(forward(topo, g, X, phi, dphi, V_drive), y)
        curve.append(ep_sep)
        if ep_sep > best_sep:
            best_sep, best_g = ep_sep, g.copy()
    sep_mean, _ = evaluate_separation(topo, best_g, X, y, phi, dphi, V_drive, cfg, rng)
    final_mean, _ = evaluate_separation(topo, g, X, y, phi, dphi, V_drive, cfg, rng)
    return {"g": best_g, "separation": float(sep_mean), "final_separation": float(final_mean),
            "curve": curve, "audit": audit}


DEFAULT_SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]      # training read-noise fraction of span
DEFAULT_BETAS = [0.02, 0.05, 0.1, 0.2, 0.5]


def _readnoise_sweep(rule, mode, cfg, topo, X, y, phi, dphi, V_drive, n_epochs, seeds, s_fn,
                     beta, target, train_sigma):
    """Run one rule/mode over seeds at fixed (beta, train_sigma); clean-eval each via evaluate_separation;
    count SNR>=3 PASSes. Fresh constant-seed eval RNG -> pure function (determinism + safe parallelism)."""
    seps, finals, hits = [], [], 0
    rng = np.random.default_rng(12345)
    audit = None
    seeds = list(seeds)
    for s in seeds:
        if rule == "manhattan":
            res = train_manhattan_noisy(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=n_epochs,
                                        seed=int(s), target=target, train_sigma=train_sigma)
        elif rule == "coupled":
            res = train_coupled_noisy(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=n_epochs,
                                      seed=int(s), target=target, beta=beta, s_fn=s_fn,
                                      train_sigma=train_sigma, mode=mode)
        else:
            raise ValueError(f"unknown rule {rule!r}")
        audit = res["audit"]
        finals.append(res["final_separation"])
        sep, floor = evaluate_separation(topo, res["g"], X, y, phi, dphi, V_drive, cfg, rng)
        seps.append(sep)
        if verdict(sep, floor)["status"] == "PASS":
            hits += 1
    return {"sep_mean": float(np.mean(seps)), "sep_per_seed": [float(v) for v in seps],
            "final_mean": float(np.mean(finals)), "frac_pass": hits / len(seeds), "audit": audit}


def _unit_result(rule, mode, train_sigma, beta, H, a, V_drive, n_epochs, seeds, target):
    """ONE sweep unit -- a pure function of PRIMITIVES ONLY (picklable; Windows-spawn-safe). Rebuilds
    topo/task/cfg/phi/s_fn and runs one (rule, mode, train_sigma, beta) column. Returns a tagged row."""
    topo = build_full_diff(H)
    X, y = parity3_data()
    cfg = DeviceConfig()
    phi, dphi = make_read_nonlinearity(a)
    s_fn = make_cocontent(a)
    sw = _readnoise_sweep(rule, mode, cfg, topo, X, y, phi, dphi, V_drive, n_epochs, list(seeds),
                          s_fn, beta, target, train_sigma)
    return {"rule": rule, "mode": mode, "train_sigma": train_sigma, "beta": beta, **sw}


def _run_unit(spec):
    """Top-level Pool worker: unpack a spec dict into _unit_result (picklable boundary)."""
    return _unit_result(**spec)


def run_readnoise(H=8, a=0.4, V_drive=1.5, n_epochs=300, seeds=range(10), beta=0.1, target=0.3,
                  sigmas=None, betas=None, nproc=1):
    """Stage-5: read-noise-in-training head-to-head (Manhattan vs coupled, both sign-pulse) + the coupled
    sigma x beta grid (Effect A) + a coupled magnitude-pulse variant (Effect B). Outcome recorded, not
    asserted. nproc>1 runs the pure units across processes; the result is bit-identical to nproc=1."""
    sigmas = DEFAULT_SIGMAS if sigmas is None else list(sigmas)
    betas = DEFAULT_BETAS if betas is None else list(betas)
    if beta not in betas:
        betas = sorted(set(betas) | {beta})
    seeds = list(seeds)
    shared = dict(H=H, a=a, V_drive=V_drive, n_epochs=n_epochs, seeds=seeds, target=target)

    work = []
    for s in sigmas:                                       # manhattan: one column (no beta dependence)
        work.append({"rule": "manhattan", "mode": "sign", "train_sigma": s, "beta": beta, **shared})
    for mode in ("sign", "magnitude"):                     # coupled grid: mode x beta x sigma
        for b in betas:
            for s in sigmas:
                work.append({"rule": "coupled", "mode": mode, "train_sigma": s, "beta": b, **shared})

    if nproc and nproc > 1:
        with mp.Pool(nproc) as pool:
            results = pool.map(_run_unit, work)            # Pool.map preserves input order
    else:
        results = [_run_unit(w) for w in work]

    man = [r for r in results if r["rule"] == "manhattan"]
    grid = {"sign": [r for r in results if r["rule"] == "coupled" and r["mode"] == "sign"],
            "magnitude": [r for r in results if r["rule"] == "coupled" and r["mode"] == "magnitude"]}
    head = {"manhattan_sign": man,
            "coupled_sign": [r for r in grid["sign"] if r["beta"] == beta],
            "coupled_magnitude": [r for r in grid["magnitude"] if r["beta"] == beta]}
    audit = {k: (v[0]["audit"] if v else None) for k, v in head.items()}
    return {"H": H, "a": a, "n_epochs": n_epochs, "n_seeds": len(seeds), "beta": beta,
            "sigmas": sigmas, "betas": betas, "head_to_head": head, "grid": grid, "audit": audit}


def _first_foreclosure(rows, thresh=0.5, key="train_sigma"):
    """The first row whose frac_pass falls below `thresh` (the foreclosure cliff), or None."""
    return next((r[key] for r in rows if r["frac_pass"] < thresh), None)


def cliff_table(result):
    """Headline: first-foreclosure train_sigma for each series at default beta, plus the coupled/Manhattan
    ratio (<<1 => coupled binds sooner => CONFIRMED; ~1 => REFUTED). Pure post-processing; recorded, not
    asserted. A None ratio means manhattan never forecloses in the swept range (or forecloses at
    train_sigma=0); read the individual cliffs above it."""
    h = result["head_to_head"]
    m = _first_foreclosure(h["manhattan_sign"])
    cs = _first_foreclosure(h["coupled_sign"])
    cm = _first_foreclosure(h["coupled_magnitude"])
    ratio = (cs / m) if (m not in (None, 0) and cs is not None) else None
    return {"manhattan_sign_cliff": m, "coupled_sign_cliff": cs,
            "coupled_magnitude_cliff": cm, "coupled_over_manhattan": ratio}


def beta_scaling_table(result):
    """Effect A signature: coupled-sign first-foreclosure train_sigma as a function of beta (smaller beta
    should foreclose earlier if the beta/sigma sign-SNR mechanism holds)."""
    out = []
    rows = result["grid"]["sign"]
    for b in result["betas"]:
        col = sorted([r for r in rows if r["beta"] == b], key=lambda r: r["train_sigma"])
        out.append({"beta": b, "coupled_sign_cliff": _first_foreclosure(col)})
    return out


def mode_table(result):
    """Effect B: coupled sign vs magnitude at default beta, per train_sigma."""
    h = result["head_to_head"]
    sign = {r["train_sigma"]: r for r in h["coupled_sign"]}
    mag = {r["train_sigma"]: r for r in h["coupled_magnitude"]}
    return [{"train_sigma": s, "sign_frac": sign[s]["frac_pass"], "mag_frac": mag[s]["frac_pass"],
             "sign_sep": sign[s]["sep_mean"], "mag_sep": mag[s]["sep_mean"]} for s in sorted(sign)]


# A tiny subset for --quick: a clean point + one noisy point, two betas, so a local end-to-end smoke runs
# in seconds before paying for the full EC2 sweep.
_QUICK = {"sigmas": [0.0, 0.1], "betas": [0.05, 0.1]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-noise amplification under noisy training (Stage-5).")
    ap.add_argument("--nproc", type=int, default=1, help="parallel worker processes (units are pure)")
    ap.add_argument("--out", default="readnoise_parity3.json")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="tiny local smoke (few sigmas/betas/seeds/epochs)")
    ap.add_argument("--sigmas", default=None,
                    help="comma-separated train_sigma grid (overrides DEFAULT_SIGMAS), e.g. 0,1e-4,1e-3,1e-2")
    ap.add_argument("--betas", default=None,
                    help="comma-separated beta grid (overrides DEFAULT_BETAS), e.g. 0.1,0.5,1.0")
    args = ap.parse_args(argv)

    sigmas = [float(x) for x in args.sigmas.split(",")] if args.sigmas else None
    betas = [float(x) for x in args.betas.split(",")] if args.betas else None

    if args.quick:
        res = run_readnoise(H=8, n_epochs=20, seeds=range(2), sigmas=_QUICK["sigmas"],
                            betas=_QUICK["betas"], nproc=args.nproc)
    else:
        res = run_readnoise(n_epochs=args.epochs, seeds=range(args.seeds), nproc=args.nproc,
                            sigmas=sigmas, betas=betas)

    ct = cliff_table(res)
    print("=" * 78)
    print("STAGE 5 -- read-noise amplification under noisy training relaxations (parity-3)")
    print("=" * 78)
    print(f"manhattan: {res['audit']['manhattan_sign']['error_site']}    "
          f"coupled: {res['audit']['coupled_sign']['error_site']}    "
          f"n_seeds={res['n_seeds']} n_epochs={res['n_epochs']} default_beta={res['beta']}")
    print("\n-- head-to-head first foreclosure in train_sigma (frac_pass<0.5), default beta --")
    print(f"  manhattan_sign      cliff = {ct['manhattan_sign_cliff']}")
    print(f"  coupled_sign        cliff = {ct['coupled_sign_cliff']}")
    print(f"  coupled_magnitude   cliff = {ct['coupled_magnitude_cliff']}")
    print(f"  coupled/manhattan ratio   = {ct['coupled_over_manhattan']}  "
          f"(<<1 => coupled binds sooner; None => manhattan robust across range — see cliffs above)")
    print("\n-- Effect A: coupled-sign cliff vs beta (smaller beta should foreclose earlier) --")
    for row in beta_scaling_table(res):
        print(f"  beta={row['beta']:<6} cliff(train_sigma) = {row['coupled_sign_cliff']}")
    print("\n-- Effect B: coupled sign vs magnitude at default beta (frac_pass) --")
    for row in mode_table(res):
        print(f"  train_sigma={row['train_sigma']:<6} sign={row['sign_frac']:.2f}  mag={row['mag_frac']:.2f}")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nraw numbers -> {args.out}")


if __name__ == "__main__":
    main()
