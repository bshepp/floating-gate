# PORTED from 02-memristive-crossbar/coupled.py @95868e7 — do not edit to diverge; re-vendor on upstream change.
"""Coupled learning / equilibrium propagation on the differential memristive crossbar -- the FULLY
co-located rule (Verdict #3). Two physical Kirchhoff relaxations -- free, and output-nudged by a net-zero
current-injection pair -- give a LOCAL per-edge update Delta g ~ (1/beta)(s_nudged - s_free), with the
co-content s = integral_0^dV phi(v) v dv; no CPU gradient is used. Bracketed against the half-co-located
Manhattan rule. Outcomes recorded, not asserted. See docs/superpowers/specs/2026-06-19-stage4-coupled-design.md."""
import json

import numpy as np

from crossbar import (edges_from_params, assemble_node_volts, forward, build_full_diff,
                      make_read_nonlinearity, make_cocontent)
from solver import assemble_laplacian, solve_batched
from memristor import apply_pulse, apply_drift, DeviceConfig, IDEAL
from task import parity3_data, separation, verdict
from manhattan import init_conductances, evaluate_separation, train


# Global sign of the contrastive update (eqprop convention). Set empirically by the Task-2 FD-direction
# check: the value that makes Delta g align with the descent direction -dE/dg. Derivation suggests -1.
_SIGMA = -1.0


def solve_inject(edges, node_volts, floating, phi, dphi, inject, max_iter=60, tol=1e-11):
    """Batched Newton Kirchhoff solve WITH current injection: solves L_g(V)*V = inject at floating nodes
    (vs = 0 in solve_batched). `inject` is (P, n), nonzero only at nudged output nodes. Built on
    assemble_laplacian so the vendored solver.py is untouched."""
    V = np.array(node_volts, float)
    P, n = V.shape
    fset = set(int(i) for i in floating)
    fidx = np.array(sorted(fset), int)
    fixed = np.array([i for i in range(n) if i not in fset], int)
    V[:, fidx] = 0.0
    L0 = assemble_laplacian(edges, V, lambda u, w: np.full(u.shape, float(w)))
    Lff0 = L0[:, fidx[:, None], fidx[None, :]]
    Lfb0 = L0[:, fidx[:, None], fixed[None, :]]
    rhs0 = inject[:, fidx] - np.einsum('pij,pj->pi', Lfb0, V[:, fixed])
    V[:, fidx] = np.linalg.solve(Lff0, rhs0[..., None])[..., 0]
    g_fn = lambda u, w: float(w) * np.asarray(phi(u), float)
    gdyn = lambda u, w: float(w) * (np.asarray(phi(u), float) + u * np.asarray(dphi(u), float))
    last = np.inf
    for _ in range(max_iter):
        R = np.einsum('pij,pj->pi', assemble_laplacian(edges, V, g_fn), V)[:, fidx] - inject[:, fidx]
        last = float(np.max(np.abs(R)))
        if last < tol:
            return V
        Jff = assemble_laplacian(edges, V, gdyn)[:, fidx[:, None], fidx[None, :]]
        V[:, fidx] += np.linalg.solve(Jff, (-R)[..., None])[..., 0]
    raise RuntimeError(f"solve_inject did not converge in {max_iter} iters (max|residual|={last:.2e})")


def free_voltages(topo, g, X, phi, dphi, V_drive):
    """Free-phase equilibrium node voltages (inputs clamped, hidden+outputs floating) -- same solve as
    crossbar.forward, returning ALL node voltages (P, n_nodes)."""
    edges = edges_from_params(topo, g)
    V0 = assemble_node_volts(topo, X, V_drive)
    return solve_batched(edges, V0, topo.floating, phi, dphi)


def nudged_solve(topo, g, X, y, phi, dphi, V_drive, beta, target, Vf=None):
    """Nudged-phase equilibrium via a net-zero current-injection pair: +beta*err at outp, -beta*err at
    outn, err = target*y - V_out_free (so the common mode relaxes freely). beta=0 -> the free solve."""
    outp, outn = topo.readout
    if Vf is None:
        Vf = free_voltages(topo, g, X, phi, dphi, V_drive)
    X = np.asarray(X, float)
    err = np.asarray(target, float) * np.asarray(y, float) - (Vf[:, outp] - Vf[:, outn])
    edges = edges_from_params(topo, g)
    V0 = assemble_node_volts(topo, X, V_drive)
    inject = np.zeros((X.shape[0], topo.n_nodes))
    inject[:, outp] = beta * err
    inject[:, outn] = -beta * err
    return solve_inject(edges, V0, topo.floating, phi, dphi, inject)


def coupled_update(topo, g, X, y, phi, dphi, V_drive, s_fn, beta=0.1, target=0.3):
    """Local contrastive update Delta g = _SIGMA/beta * sum_patterns[ s_fn(dV^N) - s_fn(dV^F) ]
    per trainable edge, with s_fn the co-content density (make_cocontent). Returns the DESCENT direction
    (n_params,). No CPU gradient -- only per-edge voltage drops from the two physical relaxations."""
    Vf = free_voltages(topo, g, X, phi, dphi, V_drive)
    Vn = nudged_solve(topo, g, X, y, phi, dphi, V_drive, beta, target, Vf=Vf)
    dg = np.zeros(topo.n_params)
    for (a, b, idx) in topo.param_edges:
        sf = np.asarray(s_fn(Vf[:, a] - Vf[:, b]), float)
        sn = np.asarray(s_fn(Vn[:, a] - Vn[:, b]), float)
        dg[idx] = _SIGMA * float(np.sum(sn - sf)) / beta
    return dg


def coupled_step(topo, g, X, y, phi, dphi, V_drive, cfg, rng, s_fn, beta=0.1, target=0.3):
    """One full-batch coupled-learning update. dg is the descent direction (sign in _SIGMA), so
    potentiate where dg>0, depress where dg<0. Same device path as the Manhattan rule."""
    dg = coupled_update(topo, g, X, y, phi, dphi, V_drive, s_fn, beta=beta, target=target)
    s = np.sign(dg)
    g_new = g.copy()
    pot = s > 0
    dep = s < 0
    if np.any(pot):
        g_new[pot] = apply_pulse(g[pot], +1, cfg, rng)       # physics: programming pulse
    if np.any(dep):
        g_new[dep] = apply_pulse(g[dep], -1, cfg, rng)
    audit = {"rule": "coupled_learning", "error_site": "on_array_local", "increment_site": "physical_pulse"}
    return g_new, audit


def train_coupled(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=300, seed=0, target=0.3, beta=0.1, s_fn=None):
    """Full-batch coupled-learning training. Records best-so-far and final-state separation. Reported,
    not asserted. Result keys mirror manhattan.train so run_bracket can treat both rules uniformly."""
    if s_fn is None:
        s_fn = make_cocontent(0.4)
    rng = np.random.default_rng(seed)
    g = init_conductances(topo, rng, cfg)
    curve, audit = [], None
    best_sep = -np.inf
    best_g = g.copy()
    for _ in range(n_epochs):
        g, audit = coupled_step(topo, g, X, y, phi, dphi, V_drive, cfg, rng, s_fn, beta=beta, target=target)
        g = apply_drift(g, cfg)
        np.clip(g, cfg.g_min, cfg.g_max, out=g)
        ep_sep = separation(forward(topo, g, X, phi, dphi, V_drive), y)
        curve.append(ep_sep)
        if ep_sep > best_sep:
            best_sep = ep_sep
            best_g = g.copy()
    sep_mean, _ = evaluate_separation(topo, best_g, X, y, phi, dphi, V_drive, cfg, rng)
    final_mean, _ = evaluate_separation(topo, g, X, y, phi, dphi, V_drive, cfg, rng)
    return {"g": best_g, "separation": float(sep_mean), "final_separation": float(final_mean),
            "curve": curve, "audit": audit}


def _bracket_sweep(train_fn, topo, X, y, phi, dphi, V_drive, cfg, n_epochs, seeds, target, **kw):
    """Run train_fn over seeds, evaluate each via evaluate_separation, count SNR≥3 PASSes via verdict.
    Returns {"sep_mean","final_mean","frac_pass","audit"}."""
    seps, finals, hits = [], [], 0
    rng = np.random.default_rng(999)
    audit = None
    for s in seeds:
        res = train_fn(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=n_epochs, seed=int(s),
                       target=target, **kw)
        audit = res["audit"]
        finals.append(res["final_separation"])
        sep, floor = evaluate_separation(topo, res["g"], X, y, phi, dphi, V_drive, cfg, rng)
        seps.append(sep)
        if verdict(sep, floor)["status"] == "PASS":
            hits += 1
    return {"sep_mean": float(np.mean(seps)), "final_mean": float(np.mean(finals)),
            "frac_pass": hits / len(list(seps)), "audit": audit}


def run_bracket(H=8, a=0.4, V_drive=1.5, n_epochs=300, seeds=range(10), cfg=None, beta=0.1, target=0.3):
    """Verdict #3: half-co-located Manhattan vs fully-co-located coupled learning on parity-3, matched
    H/device/epochs/task. Outcome recorded, not asserted."""
    cfg = DeviceConfig() if cfg is None else cfg
    X, y = parity3_data()
    topo = build_full_diff(H)
    phi, dphi = make_read_nonlinearity(a)
    s_fn = make_cocontent(a)
    seeds = list(seeds)
    man = _bracket_sweep(train, topo, X, y, phi, dphi, V_drive, cfg, n_epochs, seeds, target)
    cou = _bracket_sweep(train_coupled, topo, X, y, phi, dphi, V_drive, cfg, n_epochs, seeds, target,
                         beta=beta, s_fn=s_fn)
    return {"H": H, "a": a, "n_epochs": n_epochs, "n_seeds": len(seeds), "beta": beta,
            "manhattan": man, "coupled": cou}


def main():
    res = run_bracket()
    print("=" * 72)
    print("STAGE 4 -- coupled learning vs Manhattan on parity-3 (Verdict #3, the audit bracket)")
    print("=" * 72)
    print(f"{'rule':<12}{'sep_mean':>10}{'final':>10}{'frac_pass':>11}   audit(error_site)")
    for name in ("manhattan", "coupled"):
        r = res[name]
        print(f"{name:<12}{r['sep_mean']:>10.4f}{r['final_mean']:>10.4f}{r['frac_pass']:>11.2f}   "
              f"{r['audit']['error_site']}")
    print(f"beta={res['beta']}  n_seeds={res['n_seeds']}  n_epochs={res['n_epochs']}")
    with open("coupled_parity3.json", "w") as f:
        json.dump(res, f, indent=2)
    print("raw numbers -> coupled_parity3.json")


if __name__ == "__main__":
    main()
