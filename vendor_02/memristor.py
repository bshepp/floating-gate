# PORTED from 02-memristive-crossbar/memristor.py @95868e7 — do not edit to diverge; re-vendor on upstream change.
"""Compact phenomenological memristor model with INDIVIDUALLY GATEABLE non-idealities, so the ablation
can switch each off and recover an ideal device. State is per-edge conductance g; a programming pulse
changes g through dg/dt. Normalized G = (g - g_min)/(g_max - g_min) in [0,1].
Gates (off-value recovers ideal): nonlin (->0), asym (->1), n_levels (->0/continuous), sigma_write (->0),
sigma_read (->0), drift (->0); on/off ratio = g_max/g_min (bounds always enforced).
See docs/superpowers/specs/2026-06-18-memristive-crossbar-design.md."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DeviceConfig:
    g_min: float = 0.1
    g_max: float = 10.0          # on/off ratio = g_max/g_min
    dG0: float = 0.05            # nominal per-pulse step in normalized G units
    nonlin: float = 2.0          # update-nonlinearity exponent nu; 0 => state-independent (linear)
    asym: float = 0.5            # depression magnitude / potentiation magnitude; 1 => symmetric
    n_levels: int = 0            # conductance quantization levels; 0 => continuous
    sigma_write: float = 0.02    # cycle-to-cycle write noise, in units of dG0 (normalized)
    sigma_read: float = 0.01     # read noise, fraction of the conductance span
    drift: float = 0.0           # per-epoch retention decay toward g_rest
    g_rest: Optional[float] = None   # rest level for drift; None => mid-range


IDEAL = DeviceConfig(nonlin=0.0, asym=1.0, n_levels=0, sigma_write=0.0, sigma_read=0.0, drift=0.0)


def _span(cfg):
    return cfg.g_max - cfg.g_min


def apply_pulse(g, sign, cfg, rng, scale=1.0):
    """One programming pulse per element. sign>0 potentiate, sign<0 depress. `scale` (scalar or
    per-element array) multiplies the deterministic step magnitude; scale=1.0 is the nominal pulse and
    reproduces the fixed-amplitude behavior bit-for-bit. Cycle-to-cycle write noise is NOT scaled by
    `scale` (it models pulse-issue jitter, independent of amplitude). Returns the new g array."""
    g = np.asarray(g, float)
    G = (g - cfg.g_min) / _span(cfg)
    if sign > 0:
        dG = cfg.dG0 * scale * (1.0 - G) ** cfg.nonlin       # saturating potentiation
    else:
        dG = -cfg.dG0 * scale * cfg.asym * G ** cfg.nonlin   # saturating, asymmetric depression
    if cfg.sigma_write > 0:
        dG = dG + cfg.sigma_write * cfg.dG0 * rng.normal(size=G.shape)
    g_new = cfg.g_min + (G + dG) * _span(cfg)
    np.clip(g_new, cfg.g_min, cfg.g_max, out=g_new)          # bounds (on/off ratio)
    if cfg.n_levels and cfg.n_levels > 0:
        levels = np.linspace(cfg.g_min, cfg.g_max, cfg.n_levels)
        g_new = levels[np.argmin(np.abs(g_new[..., None] - levels), axis=-1)]
    return g_new


def read_conductance(g, cfg, rng):
    """Conductance as seen by a forward read: g plus additive read noise scaled by the span."""
    g = np.asarray(g, float)
    if cfg.sigma_read <= 0:
        return g.copy()
    return g + cfg.sigma_read * _span(cfg) * rng.normal(size=g.shape)


def apply_drift(g, cfg):
    """One epoch of retention decay toward g_rest (default mid-range). drift=0 => no change."""
    g = np.asarray(g, float)
    if cfg.drift <= 0:
        return g.copy()
    rest = cfg.g_rest if cfg.g_rest is not None else 0.5 * (cfg.g_min + cfg.g_max)
    return g + cfg.drift * (rest - g)


# --- VTEAM realism cross-check (Kvatinsky et al., IEEE TCAS-II 2015) --------------------------------
# Realism anchor only: confirms the phenomenological defaults (asymmetry, nonlinearity, range) sit in a
# device-realistic regime. NOT used as the training substrate.

_VTEAM_DEFAULT = dict(
    w_on=0.0, w_off=3.0,          # internal state bounds (nm)
    g_on=1.0e-1, g_off=1.0e1,     # conductance at w_on (low state = g_min) / w_off (high state = g_max); potentiation raises w, raising g
    v_on=-0.8, v_off=0.6,         # threshold voltages (V): below v_on depress, above v_off potentiate
    k_on=-4.0, k_off=1.0,         # rate constants
    alpha_on=3.0, alpha_off=3.0,  # nonlinearity exponents
)


def _vteam_dwdt(w, V, p):
    if V >= p["v_off"]:
        return p["k_off"] * (V / p["v_off"] - 1.0) ** p["alpha_off"]
    if V <= p["v_on"]:
        return p["k_on"] * (V / p["v_on"] - 1.0) ** p["alpha_on"]
    return 0.0


def _vteam_g(w, p):
    # linear conductance-in-state map between the two bounds
    frac = (w - p["w_on"]) / (p["w_off"] - p["w_on"])
    return p["g_on"] + frac * (p["g_off"] - p["g_on"])


def vteam_pulse_train(n_pot, n_dep, V_write=1.2, dt=1.0, params=None):
    """Apply n_pot potentiation pulses (+V_write) then n_dep depression pulses (-V_write) to a VTEAM
    state variable; return the conductance after each pulse. Realism anchor for the phenomenological model."""
    p = dict(_VTEAM_DEFAULT if params is None else params)
    w = p["w_on"]
    out = []
    for k in range(n_pot + n_dep):
        V = V_write if k < n_pot else -V_write
        w = float(np.clip(w + dt * _vteam_dwdt(w, V, p), p["w_on"], p["w_off"]))
        out.append(_vteam_g(w, p))
    return np.array(out)
