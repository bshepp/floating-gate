# PORTED from 02-memristive-crossbar/crossbar.py @95868e7 — do not edit to diverge; re-vendor on upstream change.
"""Differential memristive-crossbar topologies and forward pass.
Adapted from F:\\science-projects\\self-assembly\\stage2_diff_net.py (read-only upstream): the
trainable params here are the positive edge CONDUCTANCES g (g = w in the solver's g(dV)=w*phi(dV)).
The monolithic graph's nonlinearity is the memristor's even read-I-V curvature phi.
See docs/superpowers/specs/2026-06-18-memristive-crossbar-design.md."""
from dataclasses import dataclass

import numpy as np

from solver import solve_batched


@dataclass
class Topology:
    name: str
    n_nodes: int
    floating: list
    readout: tuple              # (node,) single-rail or (plus, minus) differential
    n_params: int
    param_edges: list           # (a, b, param_index)
    const_edges: list           # (a, b, value)


def build_single_rail(H, g_leak=1.0):
    """Nodes 0,1,2=inputs; 3=ground; 4..3+H=hidden; 4+H=out. Params W1(H*3), b1(H), w2(H). Monotone."""
    out = 4 + H
    pe, c = [], [(out, 3, g_leak)]
    for h in range(H):
        for i in range(3):
            pe.append((i, 4 + h, 3 * h + i))            # W1
        pe.append((3, 4 + h, 3 * H + h))                # b1
        pe.append((4 + h, out, 4 * H + h))              # w2
    return Topology("single_rail", 5 + H, list(range(4, 4 + H)) + [out], (out,), 5 * H, pe, c)


def build_full_diff(H, g_leak=1.0):
    """Two independent positive subnetworks sharing inputs. + subnet: hidden 4..3+H, outp=4+H.
    - subnet: hidden 5+H..4+2H, outn=5+2H. Params: + (W1p,b1p,w2p)=5H, then - (W1n,b1n,w2n)=5H.
    V_out = V[outp] - V[outn]."""
    outp = 4 + H
    h2_base = 5 + H
    outn = 5 + 2 * H
    pe, c = [], [(outp, 3, g_leak), (outn, 3, g_leak)]
    for h in range(H):                                  # + subnet (param base 0)
        for i in range(3):
            pe.append((i, 4 + h, 3 * h + i))
        pe.append((3, 4 + h, 3 * H + h))
        pe.append((4 + h, outp, 4 * H + h))
    b = 5 * H                                           # - subnet (param base 5H)
    for h in range(H):
        hn = h2_base + h
        for i in range(3):
            pe.append((i, hn, b + 3 * h + i))
        pe.append((3, hn, b + 3 * H + h))
        pe.append((hn, outn, b + 4 * H + h))
    floating = list(range(4, 4 + H)) + [outp] + list(range(h2_base, h2_base + H)) + [outn]
    return Topology("full_diff", 6 + 2 * H, floating, (outp, outn), 10 * H, pe, c)


def edges_from_params(topo, params):
    p = np.asarray(params, float)
    return topo.const_edges + [(a, b, float(p[idx])) for (a, b, idx) in topo.param_edges]


def assemble_node_volts(topo, X, V_drive):
    X = np.asarray(X, float)
    P = X.shape[0]
    V = np.full((P, topo.n_nodes), np.nan)
    V[:, 0:3] = X * V_drive
    V[:, 3] = 0.0
    return V


def readout_value(topo, V):
    if len(topo.readout) == 1:
        return V[:, topo.readout[0]]
    p, n = topo.readout
    return V[:, p] - V[:, n]


def forward(topo, params, X, phi, dphi, V_drive):
    edges = edges_from_params(topo, params)
    V = assemble_node_volts(topo, X, V_drive)
    Vsol = solve_batched(edges, V, topo.floating, phi, dphi)
    return readout_value(topo, Vsol)


def make_read_nonlinearity(a):
    """Even memristor read-I-V curvature: g_eff(dV) = g * phi(dV), phi(0)=1. a=0 => Ohmic (linear)."""
    a = float(a)
    phi = lambda u: 1.0 + a * np.asarray(u, float) ** 2
    dphi = lambda u: 2.0 * a * np.asarray(u, float)
    return phi, dphi


def make_cocontent(a):
    """Co-content density s(u) = integral_0^u phi(v) v dv for phi(u)=1+a*u^2, i.e. u^2/2 + a*u^4/4.
    This is d/dg of the per-edge co-content g*s(dV); the eqprop/coupled-learning local quantity (NOT the
    dissipated power phi(u)*u^2). a=0 => u^2/2, the linear-resistor case."""
    a = float(a)
    return lambda u: 0.5 * np.asarray(u, float) ** 2 + 0.25 * a * np.asarray(u, float) ** 4


def null_anchor(topo, X, w_lo, w_hi, phi, dphi, V_drive):
    """All params at the mid-span baseline (no conductance differentiation). Separation ~ 0."""
    w0 = 0.5 * (w_lo + w_hi)
    return forward(topo, np.full(topo.n_params, w0), X, phi, dphi, V_drive)


@dataclass
class TiledTopology:
    """Feedforward two-layer tiled crossbar: closed-form virtual-ground differential MACs with an
    explicit periphery activation. Bias-free (a nonzero voltage bias would break the net's oddness;
    see the design spec). Trainable params are the positive-rail conductances of both layers."""
    name: str
    n_in: int
    H: int
    n_params: int


def build_tiled_periphery(H, n_in=3):
    """3 inputs -> H hidden (tanh periphery) -> 1 differential readout. Params, in order:
    g1p (H*n_in), g1n (H*n_in), g2p (H), g2n (H). Signed weights are g1p-g1n and g2p-g2n."""
    n_params = 2 * H * n_in + 2 * H
    return TiledTopology("tiled_periphery", n_in, H, n_params)


def tiled_unpack(topo, params):
    """Slice the flat conductance vector into (g1p, g1n, g2p, g2n)."""
    H, n_in = topo.H, topo.n_in
    p = np.asarray(params, float)
    i = 0
    g1p = p[i:i + H * n_in].reshape(H, n_in); i += H * n_in
    g1n = p[i:i + H * n_in].reshape(H, n_in); i += H * n_in
    g2p = p[i:i + H]; i += H
    g2n = p[i:i + H]; i += H
    return g1p, g1n, g2p, g2n


def tiled_pass(topo, params, X, phi, V_drive, f=np.tanh):
    """Forward intermediates of the tiled net. Each layer is a virtual-ground MAC: the per-device
    read current is g*phi(V)*V, so the layer input is transconducted by s(V)=phi(V)*V (odd in V)."""
    g1p, g1n, g2p, g2n = tiled_unpack(topo, params)
    drive = np.asarray(X, float) * V_drive               # (P, n_in) row voltages
    s1 = phi(drive) * drive                              # (P, n_in) read-transconducted input
    W1 = g1p - g1n                                       # (H, n_in) signed layer-1 weights
    pre1 = s1 @ W1.T                                     # (P, H)
    a1 = f(pre1)                                         # (P, H) explicit periphery neuron
    s2 = phi(a1) * a1                                    # (P, H) layer-2 transconducted input
    w2 = g2p - g2n                                       # (H,) signed layer-2 weights
    out = s2 @ w2                                        # (P,) differential readout
    return dict(s1=s1, W1=W1, pre1=pre1, a1=a1, s2=s2, w2=w2, out=out)


def tiled_forward(topo, params, X, phi, dphi, V_drive, f=np.tanh):
    """Differential readout of the tiled net. `dphi` is accepted (unused) so the signature matches
    `forward`, making this a drop-in forward_fn for the Manhattan loop."""
    return tiled_pass(topo, params, X, phi, V_drive, f)["out"]
