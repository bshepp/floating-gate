# SEED — 08-floating-gate

A brief to be turned into spec → plan → implementation by a fresh agent. The design spec
already exists at `docs/superpowers/specs/2026-06-24-floating-gate-hardware-transfer-design.md`
— read it first; this seed is the one-paragraph origin.

## The brief

Take `02-memristive-crossbar`'s in-fabric learning result toward a **genuine analog
floating-gate array** — the one analog-crossbar device family that *can* keep the update in
the fabric by a physical mechanism (charge written to an isolated gate by tunneling /
hot-carrier injection), as opposed to the owned **switched-capacitor** Anadigm FPAA
(host-written = inference; that is the separate `fpaa-neural-net` project). `02` proved the
half-co-located rule learns parity-3 above the device floor but that the *fully*-co-located
two-read rule is read-noise-fragile (~100× wall). The job here is to find out whether the
**half-co-located** object survives a floating-gate array's read noise, and — the real
floating-gate question — whether its *write side* can realize the update in-fabric.

## Aspirational, by construction

Analog floating-gate transistors are **not buyable as discrete parts** — the analog form
exists only integrated (an academic FG-FPAA, e.g. the Georgia Tech / Hasler line, or a
CMOS/MPW run). So `08` has **no owned device today**: it is the portfolio's *aspirational*
in-fabric build, with a **provisional** operating point until a real array is accessed. "No
accessible floating-gate array at acceptable cost" is itself a valid FORECLOSE.

## Non-negotiables (portfolio method)

1. **Sim gate before silicon.** Phase 0 = the H5 device gate, run before pursuing any device
   access. Compute-first holds; the "compute" reuses `02`'s sim at a provisional `σ_fg`.
2. **The update stays in the fabric.** Two-axis audit (port `commons/audit.py`). A
   host-computed gradient = inference, not learning, and fails the verdict. A floating gate
   *can* satisfy this (physical read + physical charge write); the switched-cap FPAA could
   not — that is why it is a separate project.
3. **Foreclosure is a valid end.** A clean "too noisy" (Phase 0) or "no accessible device"
   (Phase 1) saves cost and is a publishable result.
4. **Pre-registered verdicts** frozen in `verdict.py` before each run; Observed kept separate
   from Interpretation in findings.

## First milestone

Run `02`'s H5 sweep at a provisional `σ_fg` on the half-co-located rule; classify with the
frozen `classify_h5`. Expect an easy PASS (floating-gate read noise is low) — which pins the
operating point and shifts the real question to the **write side** at Phase 1, where device
access must first be secured.
