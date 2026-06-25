# PORTED from 02-memristive-crossbar/solver.py @95868e7 — do not edit to diverge; re-vendor on upstream change.
# VENDORED from F:\science-projects\self-assembly\stage2_solver.py (read-only upstream), 2026-06-18.
# Kept self-contained for parallel 02-05 execution. Do not edit to diverge silently;
# re-vendor if upstream changes. See docs/superpowers/specs/2026-06-18-memristive-crossbar-design.md.
"""Batched Newton Kirchhoff forward solve for a conductance network.
Edges carry conductance g(dV) = w*phi(V_a - V_b); node voltages solve current conservation at floating
nodes. Newton uses the analytic dynamic-Laplacian Jacobian (g_dyn = w*(phi + dV*dphi)), vectorized over
all input patterns at once."""
import numpy as np


def assemble_laplacian(edges, V, weight_fn):
    """Batched conductance Laplacian (P, n, n). weight_fn(u, w) -> (P,) conductance for edge weight w
    and per-pattern bias u = V_a - V_b. (L @ V)[:, n] = sum_b g_nb (V_n - V_b) = net current out of n."""
    P, n = V.shape
    L = np.zeros((P, n, n))
    for (a, b, w) in edges:
        u = V[:, a] - V[:, b]
        g = np.asarray(weight_fn(u, w), float)
        L[:, a, a] += g; L[:, b, b] += g
        L[:, a, b] -= g; L[:, b, a] -= g
    return L


def solve_batched(edges, node_volts, floating, phi, dphi, max_iter=60, tol=1e-11):
    """Node voltages by Newton on the Kirchhoff residual with the analytic dynamic-Laplacian Jacobian,
    batched over all patterns. node_volts: (P, n) with np.nan at floating nodes. Returns (P, n)."""
    V = np.array(node_volts, float)
    P, n = V.shape
    fset = set(int(i) for i in floating)
    fidx = np.array(sorted(fset), int)
    fixed = np.array([i for i in range(n) if i not in fset], int)
    V[:, fidx] = 0.0

    # warm start: one linear solve with g = w (phi == 1)
    L0 = assemble_laplacian(edges, V, lambda u, w: np.full(u.shape, float(w)))
    Lff0 = L0[:, fidx[:, None], fidx[None, :]]
    Lfb0 = L0[:, fidx[:, None], fixed[None, :]]
    rhs0 = -np.einsum('pij,pj->pi', Lfb0, V[:, fixed])
    V[:, fidx] = np.linalg.solve(Lff0, rhs0[..., None])[..., 0]

    g_fn = lambda u, w: float(w) * np.asarray(phi(u), float)
    gdyn_fn = lambda u, w: float(w) * (np.asarray(phi(u), float) + u * np.asarray(dphi(u), float))

    last = np.inf
    for _ in range(max_iter):
        Lg = assemble_laplacian(edges, V, g_fn)
        R = np.einsum('pij,pj->pi', Lg, V)[:, fidx]          # residual current at floating nodes
        last = float(np.max(np.abs(R)))
        if last < tol:
            return V
        Jd = assemble_laplacian(edges, V, gdyn_fn)
        Jff = Jd[:, fidx[:, None], fidx[None, :]]
        dVf = np.linalg.solve(Jff, (-R)[..., None])[..., 0]
        V[:, fidx] += dVf
    raise RuntimeError(f"solve_batched did not converge in {max_iter} iters (max|residual|={last:.2e})")
