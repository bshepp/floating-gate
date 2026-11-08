# Design — Phase 1c: the injection-only differential-pair learner at raw FG nonlinearity (H6)

**Date:** 2026-11-08 · **Status:** pre-registered, **unrun** at commit time (constants frozen in
`fg_diffpair.py`; verdict = the already-frozen `classify_transfer`, unchanged).
**Parent:** `2026-11-05-phase1b-device-access-gate-design.md` and its findings addendum (2026-11-08): the
OTC2902E04 is the published RASP 3.0 design, so its floating gates are **injection-only with global erase**
and have **no run-mode fabric-triggered write**.
**Follows:** Phase 1 — half rule PASS at the closed-loop nominal (ν_eff=2, SNR 10.9) but **FORECLOSE at raw
ν=3 (SNR 0.20)**; update nonlinearity is the binding non-ideality.

## 0. Why this sim, now

Two device facts from the papers change what the right write model is:

1. **Injection only raises pFET current; tunneling erase is global.** A single cell has no per-cell
   depression, so `02`'s single-cell `asym` model is not the device. The device-faithful weight is a
   **differential pair** `w = I+ − I−`: *depress* means *inject the minus cell*.
2. **Each in-array update is a single open-loop pulse** (closed-loop program-verify is a maintenance
   operation, not a learning step). The learner therefore sees the **raw** nonlinearity `dG = dG0 (1−G)^ν`,
   which foreclosed the single-cell half rule at ν=3.

Those two facts together produce a hypothesis worth a cheap sim before any purchase or build:

> **H6.** With periodic global erase + re-inject (renormalization: smaller cell to 0), both cells of a pair
> sit near the bottom of their range, where `(1−G)^ν ≈ 1`. The pair therefore **linearizes** the raw update,
> and the half rule that foreclosed at raw ν=3 as a single cell **PASSes** (SNR ≥ 3) as a pair.

A clean refutation is as useful as a confirmation: it would say the in-fabric *and* the inference-grade
open-loop learner both die on raw ν on this family, leaving only closed-loop writes.

## 1. The model (composes vendored physics; `fg_diffpair.py`)

- **Pair state** `(G+, G−)`, each in `[0,1]`; `w = g_min + ½(G+ − G− + 1)·span`. Initialization: `02`'s
  `init_conductances`, converted to the renormalized pair (smaller cell at 0).
- **Pulse** = the vendored `apply_pulse(·, +1, cfg)` potentiation branch on a *cell* (no quantizer,
  `asym` irrelevant since depression never fires). Potentiate `w` → pulse `G+`; depress `w` → pulse `G−`.
- **Renormalization** when **any** cell exceeds `HEADROOM = 0.9`: the *whole array* is erased (tunneling is
  global) and every pair re-injected to its retained `w` with the smaller cell at 0, each cell landing with
  error `renorm_sigma = RMS of an 8-bit quantizer` (`RENORM_LEVELS = 256`, the Phase-1 closed-loop
  precision). This preserves `w`; it computes nothing.
- **Reads** of `w` combine two independent cell reads → effective read noise `σ/√2`. Eval is held at
  `cfg.sigma_read` exactly as in Phase 0/1 (conservative, comparable).
- **Rules**: half (one read, off-array CPU sign) and full (two reads, local contrast), both on the pair.

## 2. Pre-registered operating point and sweeps (frozen in `fg_diffpair.py`)

- `diffpair_nominal()` = Phase-1 `fg_device_config()` with `nonlin = NU_RAW_NOMINAL = 3.0` (the value that
  foreclosed), `asym = 1.0`, `n_levels = 0`. `sigma_fg = 0.01`, 300 epochs, 10 seeds, `β = 0.1`.
- `NU_SWEEP = [0,1,2,3,4,5]` (half rule; single-cell reference at ν=3 is Phase 1's 0.20).
- `HEADROOM_SWEEP = [0.3, 0.6, 0.9]` (more frequent renorm = more linear, more erases).
- `DIFFPAIR_K_GRID = [1, 100, 10⁴]` for the in-fabric robust scan (recorded co-equally).

## 3. Verdict and audit

- **Gate:** the frozen `classify_transfer` on the half rule at the nominal — **no new verdict function**;
  H6 is PASS/BOUNDARY/FORECLOSE on exactly the same bar as Phase 1.
- **Endurance:** `writes_per_cell` = max injections any cell receives + one tunneling stress per global
  erase (both stress the oxide), against `ENDURANCE_BUDGET = 1e5`. Erases per run are recorded.
- **Audit:** learning updates as before (half `(cpu, physical)`; full `(physical, physical)`), **plus** the
  renormalization re-inject recorded as `(cpu, physical)` — a host-*targeted* physical write that computes
  no gradient. So even the "in-fabric" rule is **not both-physical** once an erase cycle occurs; the
  findings must say so rather than let the maintenance op hide inside a PASS.

## 4. What a result means

| outcome at raw ν=3 (half, pair) | reading |
|---|---|
| PASS | H6 confirmed: the pair + renorm linearizes the raw update; an open-loop single-pulse learner is viable on this family (inference-grade, host sign). Count the erases: that is the real cost. |
| BOUNDARY | partial: headroom/dG0 co-design matters; sweep says which. |
| FORECLOSE | H6 refuted: raw ν kills the open-loop learner even as a pair; only closed-loop program-verify writes learn on this family (confirms Phase 1's ν_eff framing). |

## 5. Scope

Code, tests and this spec are committed **before** `python fg_diffpair.py` runs; the run is serial on the
dev box (~10–15 min); the findings note records Observed vs Interpretation. No purchase, no build, no edits
to `vendor_02/`.
