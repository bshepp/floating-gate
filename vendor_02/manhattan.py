# PORTED from 02-memristive-crossbar/manhattan.py @95868e7 — do not edit to diverge; re-vendor on upstream change.
"""Manhattan / sign-rule in-situ learning on the differential memristive crossbar.
Audit stance (HALF co-located): the array computes the activations (physics); the ERROR SIGN is computed
OFF-ARRAY on the CPU via the analytic output Jacobian; the weight INCREMENT is PHYSICAL -- one
fixed-amplitude programming pulse per edge whose dg/dt is the memristor model (with its non-idealities).
Learning OUTCOMES are recorded, not asserted. See the design spec."""
import json

import numpy as np

from crossbar import forward, build_full_diff, make_read_nonlinearity
from jacobian import output_jacobian
from memristor import apply_pulse, read_conductance, apply_drift, DeviceConfig, IDEAL
from task import separation, parity3_data, verdict
from ceilings import de_ceiling, sgd_ceiling


def init_conductances(topo, rng, cfg):
    """Start near mid-range with small jitter (an unprogrammed array)."""
    mid = 0.5 * (cfg.g_min + cfg.g_max)
    span = cfg.g_max - cfg.g_min
    g = mid + 0.02 * span * rng.normal(size=topo.n_params)
    return np.clip(g, cfg.g_min, cfg.g_max)


def monolithic_grad(topo, g, X, y, phi, dphi, V_drive, target=0.3):
    """dE/dg for the monolithic differential net via the analytic output Jacobian (off-array CPU)."""
    out = forward(topo, g, X, phi, dphi, V_drive)
    err = out - np.asarray(y, float) * target
    return err @ output_jacobian(topo, g, X, phi, dphi, V_drive)


def manhattan_step(topo, g, X, y, phi, dphi, V_drive, cfg, rng, target=0.3, grad_fn=None):
    """One full-batch Manhattan update. grad_fn(topo,g,X,y,phi,dphi,V_drive,target)->dE/dg
    (default: monolithic_grad). Returns (g_new, audit)."""
    grad_fn = monolithic_grad if grad_fn is None else grad_fn
    grad = grad_fn(topo, g, X, y, phi, dphi, V_drive, target=target)   # off-array (CPU) error gradient
    sign = np.sign(grad)                                              # Manhattan: only the sign is used
    g_new = g.copy()
    pot = sign < 0
    dep = sign > 0
    if np.any(pot):
        g_new[pot] = apply_pulse(g[pot], +1, cfg, rng)                # physics: programming pulse
    if np.any(dep):
        g_new[dep] = apply_pulse(g[dep], -1, cfg, rng)
    audit = {"rule": "manhattan_sign", "error_site": "off_array_cpu", "increment_site": "physical_pulse"}
    return g_new, audit


def evaluate_separation(topo, g, X, y, phi, dphi, V_drive, cfg, rng, n_reads=64, forward_fn=forward):
    """Mean two-sided separation over noisy reads + the read-noise floor (std across reads)."""
    seps = []
    for _ in range(n_reads):
        g_read = read_conductance(g, cfg, rng)
        np.clip(g_read, cfg.g_min, cfg.g_max, out=g_read)
        out = forward_fn(topo, g_read, X, phi, dphi, V_drive)
        seps.append(separation(out, y))
    seps = np.asarray(seps)
    floor = max(float(np.std(seps)), 1e-9)
    return float(np.mean(seps)), floor


def train(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=300, seed=0, target=0.3,
          forward_fn=forward, grad_fn=None):
    """Full-batch Manhattan training (drives the monolithic net by default; pass forward_fn=tiled_forward
    and grad_fn=tiled_gradient for the tiled net). Outcome reported, not asserted."""
    rng = np.random.default_rng(seed)
    g = init_conductances(topo, rng, cfg)
    curve, audit = [], None
    best_sep = -np.inf
    best_g = g.copy()
    for _ in range(n_epochs):
        g, audit = manhattan_step(topo, g, X, y, phi, dphi, V_drive, cfg, rng, target=target, grad_fn=grad_fn)
        g = apply_drift(g, cfg)
        np.clip(g, cfg.g_min, cfg.g_max, out=g)
        ep_sep = separation(forward_fn(topo, g, X, phi, dphi, V_drive), y)
        curve.append(ep_sep)
        if ep_sep > best_sep:
            best_sep = ep_sep
            best_g = g.copy()
    sep_mean, _ = evaluate_separation(topo, best_g, X, y, phi, dphi, V_drive, cfg, rng, forward_fn=forward_fn)
    final_mean, _ = evaluate_separation(topo, g, X, y, phi, dphi, V_drive, cfg, rng, forward_fn=forward_fn)
    return {"g": best_g, "separation": float(sep_mean), "final_separation": float(final_mean),
            "curve": curve, "audit": audit}


def _sweep(cfg, topo, X, y, phi, dphi, V_drive, n_epochs, seeds, forward_fn=forward, grad_fn=None):
    seps, finals, snr_hits = [], [], 0
    rng = np.random.default_rng(12345)
    audit = None
    for s in seeds:
        res = train(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=n_epochs, seed=int(s),
                    forward_fn=forward_fn, grad_fn=grad_fn)
        audit = res["audit"]
        finals.append(res["final_separation"])
        sep, floor = evaluate_separation(topo, res["g"], X, y, phi, dphi, V_drive, cfg, rng, forward_fn=forward_fn)
        seps.append(sep)
        if verdict(sep, floor)["status"] == "PASS":
            snr_hits += 1
    return {"sep_mean": float(np.mean(seps)), "sep_per_seed": [float(v) for v in seps],
            "final_mean": float(np.mean(finals)), "frac_pass": snr_hits / len(list(seps)), "audit": audit}


def run_experiment(H=8, a=0.4, V_drive=1.5, n_epochs=400, seeds=range(10), cfg=None,
                   de_seeds=range(4), sgd_seeds=range(20)):
    """Stage-2 Verdict #1: does the Manhattan in-situ rule learn parity-3 under device non-idealities,
    vs the DE representability ceiling and the SGD learnability ceiling? Outcome recorded, not asserted."""
    cfg = DeviceConfig() if cfg is None else cfg
    X, y = parity3_data()
    topo = build_full_diff(H)
    phi, dphi = make_read_nonlinearity(a)

    de = de_ceiling(topo, X, y, cfg.g_min, cfg.g_max, phi, dphi, V_drive, seeds=de_seeds)
    sgd = sgd_ceiling(X, y, n_hid=H, seeds=sgd_seeds)
    dev = _sweep(cfg, topo, X, y, phi, dphi, V_drive, n_epochs, list(seeds))
    idl = _sweep(IDEAL, topo, X, y, phi, dphi, V_drive, n_epochs, list(seeds))

    # verdict on the device run: mean separation vs a floor from one representative noisy eval
    rng = np.random.default_rng(0)
    g0 = init_conductances(topo, rng, cfg)
    _, floor = evaluate_separation(topo, g0, X, y, phi, dphi, V_drive, cfg, rng)
    vd = verdict(dev["sep_mean"], floor)
    return {"H": H, "a": a, "V_drive": V_drive, "n_epochs": n_epochs, "n_seeds": len(list(seeds)),
            "de_ceiling": de["separation"], "sgd_ceiling": sgd["frac_solved"],
            "device": dev, "ideal": idl, "verdict": vd, "audit": dev["audit"]}


def main():
    res = run_experiment()
    print("=" * 70)
    print("STAGE 2 -- Manhattan in-situ rule on parity-3 (Verdict #1)")
    print("=" * 70)
    print(f"audit: error={res['audit']['error_site']}  increment={res['audit']['increment_site']}")
    print(f"DE representability ceiling (separation): {res['de_ceiling']:.4f}")
    print(f"SGD learnability ceiling (frac_solved):   {res['sgd_ceiling']:.2f}")
    print(f"{'run':<10}{'sep_mean':>12}{'frac_pass(SNR>=3)':>20}")
    for label, key in (("ideal", "ideal"), ("device", "device")):
        print(f"{label:<10}{res[key]['sep_mean']:>12.4f}{res[key]['frac_pass']:>20.2f}")
    print(f"VERDICT (device): {res['verdict']['status']}  (SNR={res['verdict']['snr']:.2f})")
    with open("manhattan_parity3.json", "w") as f:
        json.dump(res, f, indent=2)
    print("raw numbers -> manhattan_parity3.json")


if __name__ == "__main__":
    main()
