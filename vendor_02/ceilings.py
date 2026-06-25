# PORTED from 02-memristive-crossbar/ceilings.py @95868e7 — do not edit to diverge; re-vendor on upstream change.
"""Two ceilings for the in-situ learning verdict.
- de_ceiling: offline differential_evolution over the crossbar conductances -> the REPRESENTABILITY
  ceiling (can this topology + read-nonlinearity separate the task at all?). Adapted from
  self-assembly/stage2_diff_net.fit_de.
- sgd_ceiling: an idealized tanh-MLP trained by backprop -> the LEARNABILITY ceiling (what an ideal
  gradient achieves). Adapted from self-assembly/eqprop_core.train_backprop. Crossbar-independent.
- jacobian_ceiling: L-BFGS on the same MSE objective the in-situ rule descends, using the analytic
  output Jacobian -> a self-consistent representability ceiling (replaces the under-converging de_ceiling
  for the verdict).
See docs/superpowers/specs/2026-06-18-memristive-crossbar-design.md."""
import numpy as np
from scipy.optimize import differential_evolution, minimize

from crossbar import forward
from jacobian import output_jacobian
from task import separation


def de_ceiling(topo, X, y, w_lo, w_hi, phi, dphi, V_drive, seeds=range(4), maxiter=150):
    n = topo.n_params
    w_mid = 0.5 * (w_lo + w_hi)
    bounds = [(w_lo, w_hi)] * n

    def neg_sep(p):
        return -separation(forward(topo, p, X, phi, dphi, V_drive), y)

    best = None
    for s in seeds:
        rng = np.random.default_rng(s)
        pop = rng.uniform(w_lo, w_hi, size=(max(n + 2, 15), n))
        pop[0, :] = w_mid                       # seed the null anchor; DE keeps the best
        r = differential_evolution(neg_sep, bounds, seed=int(s), maxiter=maxiter,
                                   polish=False, init=pop, tol=0.01)
        sep = -r.fun
        if best is None or sep > best["separation"]:
            best = {"separation": float(sep), "params": np.asarray(r.x)}
    return best


def sgd_ceiling(X, y, n_hid=8, seeds=range(20), epochs=4000, lr=0.5):
    """Fraction of seeds where a tanh-hidden / sigmoid-output MLP solves the task exactly under backprop.
    Labels y in {-1,+1} are mapped to targets {0,1}."""
    X = np.asarray(X, float)
    t = (np.asarray(y, float) > 0).astype(float)        # {-1,+1} -> {0,1}
    n_in = X.shape[1]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    solved = []
    for s in seeds:
        rng = np.random.default_rng(s)
        W1 = rng.normal(0, 0.5, (n_hid, n_in)); b1 = np.zeros(n_hid)
        W2 = rng.normal(0, 0.5, n_hid); b2 = 0.0
        for _ in range(epochs):
            i = rng.integers(len(X))
            x, tt = X[i], t[i]
            a1 = np.tanh(W1 @ x + b1)
            o = sig(W2 @ a1 + b2)
            d2 = (o - tt) * o * (1 - o)
            d1 = (W2 * d2) * (1 - a1 ** 2)
            W2 -= lr * d2 * a1; b2 -= lr * d2
            W1 -= lr * np.outer(d1, x); b1 -= lr * d1
        preds = np.array([sig(W2 @ np.tanh(W1 @ x + b1) + b2) for x in X])
        solved.append(float(np.mean((preds > 0.5) == (t > 0.5)) == 1.0))
    return {"frac_solved": float(np.mean(solved)), "n_seeds": len(solved)}


def jacobian_ceiling(topo, X, y, w_lo, w_hi, phi, dphi, V_drive, seeds=range(4), target=0.3, maxiter=300):
    """Representability ceiling = the idealized continuous-gradient optimum of the SAME objective the
    Manhattan rule descends, E = 1/2||out - y*target||^2, via L-BFGS-B with the analytic output Jacobian.
    Returns the best separation over seeds. Replaces the under-converging DE ceiling for the verdict."""
    yv = np.asarray(y, float) * target
    bounds = [(w_lo, w_hi)] * topo.n_params

    def fun_and_grad(g):
        out = forward(topo, g, X, phi, dphi, V_drive)
        err = out - yv
        return 0.5 * float(err @ err), err @ output_jacobian(topo, g, X, phi, dphi, V_drive)

    best = None
    for s in seeds:
        rng = np.random.default_rng(int(s))
        g0 = rng.uniform(w_lo, w_hi, topo.n_params)
        r = minimize(fun_and_grad, g0, jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": maxiter})
        sep = separation(forward(topo, r.x, X, phi, dphi, V_drive), y)
        if best is None or sep > best["separation"]:
            best = {"separation": float(sep), "params": np.asarray(r.x)}
    return best
