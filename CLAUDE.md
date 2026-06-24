# CLAUDE.md

Guidance for Claude Code when working in `08-floating-gate`.

## What this project is

The **floating-gate hardware-transfer arm of `02-memristive-crossbar`** — toward a genuine
analog floating-gate array that learns with the update kept *in the fabric* (charge written by
tunneling / hot-carrier injection). It is the *aspirational* cousin of the buyable memristor
arm (`06-crossbar-build`) and the in-fabric-capable counterpart of the switched-capacitor FPAA
(`fpaa-neural-net`, which is inference-only). Read `README.md` and the design spec in
`docs/superpowers/specs/` before doing anything.

## The two hard rules

1. **Sim gate before device access.** Do not pursue a floating-gate array (FG-FPAA
   collaboration or MPW run) until Phase 0 (the H5 read-noise sweep, reusing `02`'s sim)
   classifies PASS through the *frozen* `verdict.py`. The operating point `σ_fg` is provisional
   until a real device confirms it.
2. **The update must stay in the fabric.** The intended update is a floating-gate charge
   write. If a host PC computes the gradient and merely sets the charge, that is **inference,
   not learning** and fails the verdict — the exact failure that disqualified the switched-cap
   FPAA. Record both audit axes (gradient source × write mechanism) on every result; port
   `commons/audit.py`.

## Method discipline (portfolio-wide)

- Pre-register every verdict as a frozen function (`verdict.py`) *before* the run.
- Foreclosure is a valid, cheap outcome — including "no accessible floating-gate array at
  acceptable cost" (Phase 1). A clean "no" at any gate ends the project honestly.
- Keep *Observed* separate from *Interpretation* in `docs/superpowers/findings/`.
- Answer *who computes the update?* explicitly for every learning result.
- Read-noise H5 is expected to PASS easily here; do not over-read it. The floating-gate wall is
  most likely write-side (Phase 1).

## Conventions

Small flat modules; functions return numbers; one pytest module per source module; frequent
commits; **commit/push only when the user asks.**

## Commons

Before starting work, read `../00-shared-harness/BULLETIN.md` and
`../00-shared-harness/synthesis/glossary.md`. Port the two-axis `audit.py` from `commons/` and
reuse `02`'s `readnoise.py` / `coupled.py` as the H5 sim. Report reusable findings up to the
commons via the promotion-branch handoff. Do **not** edit `06` or `fpaa-neural-net` — the
commons↔sibling write boundary.
