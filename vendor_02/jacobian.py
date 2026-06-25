# PORTED from 02-memristive-crossbar/jacobian.py @95868e7 — do not edit to diverge; re-vendor on upstream change.
# VENDORED (adapted) from F:\science-projects\self-assembly\stage2_grad.py (read-only upstream), 2026-06-18.
# Imports retargeted to solver/crossbar. Kept self-contained for parallel 02-05 execution.
# See docs/superpowers/specs/2026-06-18-memristive-crossbar-design.md.
"""Analytic output Jacobian dV_out/d(conductances) via the implicit function theorem, reusing the
Newton solve's dynamic-Laplacian factorization. Supplies the Manhattan rule's off-array error gradient."""
import numpy as np

from solver import assemble_laplacian, solve_batched
from crossbar import edges_from_params, assemble_node_volts


def output_jacobian(topo, params, X, phi, dphi, V_drive):
    """(P, n_params): dV_out[k]/d params[j]. At the solved voltages, J_ff (dV_f/dw) = -(dR_f/dw).
    Only the edge using params[j] contributes to dR/dw_j, at its two endpoints (current phi(u)*u)."""
    params = np.asarray(params, float)
    edges = edges_from_params(topo, params)
    Vnv = assemble_node_volts(topo, X, V_drive)
    V = solve_batched(edges, Vnv, topo.floating, phi, dphi)
    P, n = V.shape
    fidx = np.array(sorted(int(i) for i in topo.floating), int)
    pos = {node: k for k, node in enumerate(fidx)}      # node -> floating row index

    gdyn = lambda u, w: float(w) * (np.asarray(phi(u), float) + u * np.asarray(dphi(u), float))
    Jd = assemble_laplacian(edges, V, gdyn)
    Jff = Jd[:, fidx[:, None], fidx[None, :]]            # (P, nf, nf)

    nf = len(fidx)
    dRdw = np.zeros((P, nf, topo.n_params))              # holds +dR_f/dw_j (current sensitivity); negated at the solve below
    for (a, b, idx) in topo.param_edges:
        u = V[:, a] - V[:, b]
        s = np.asarray(phi(u), float) * u                # d(current_ab)/d w = phi(u)*u
        if a in pos:
            dRdw[:, pos[a], idx] += s              # R_a increases with w (current out of a)
        if b in pos:
            dRdw[:, pos[b], idx] -= s
    # dV_f/dw = -Jff^{-1} (dR_f/dw); we stored +current sensitivity = +(dR/dw) at node a, so solve with -.
    dVf = np.linalg.solve(Jff, -dRdw)                    # (P, nf, n_params)

    sel = np.zeros(nf)
    if len(topo.readout) == 1:
        sel[pos[topo.readout[0]]] = 1.0
    else:
        p_node, n_node = topo.readout
        sel[pos[p_node]] = 1.0; sel[pos[n_node]] = -1.0
    return np.einsum('i,pij->pj', sel, dVf)              # (P, n_params)
