# 08 — floating-gate

**The floating-gate hardware-transfer arm of `02-memristive-crossbar`.** It asks whether
`02`'s idealized in-fabric learning PASS can be realized on a **genuine analog floating-gate
array** — the device family that stores an analog weight as non-volatile charge on an
isolated gate and updates it by a *physical* tunneling / hot-carrier injection event.

**Created:** 2026-06-24. **Status:** Phase 0 + Phase 1 sim gates run (provisional).
**Phase 0** (H5 read-noise, 2026-06-24) → **PASS** on the buildable half-co-located rule; the in-fabric
two-read rule forecloses on read noise. **Phase 1** (transfer gate at FG-realistic write-side params,
2026-06-25) → **PASS (inference-grade)**: the buildable rule learns at a high-precision closed-loop FG
operating point (`snr_half=10.9`, endurance fine), so a tapeout is *earned for an FG inference
accelerator* — but the genuinely in-fabric rule **forecloses** (on both read noise *and* the FG write
side; multi-read averaging cannot rescue it), so **in-fabric learning is not achieved**. The binding
non-ideality is the FG **update nonlinearity**, not precision/retention (ν=3 forecloses even at infinite
precision; ν→0 co-design or closed-loop compensation required). The next move is the open
*read-noise-robust **in-fabric** update* problem (or a characterization tapeout to confirm the
provisional params). See `docs/superpowers/findings/2026-06-25-phase1-transfer-sim-gate-findings.md`.
**Aspirational / trailing:** no floating-gate device is owned or buyable (see below), so the operating
point is provisional and nothing is built until access is secured.

## The three crossbar cousins (and why only two can learn)

`02`'s sim has three real-device descendants, split by device physics:

| sibling | device | in-fabric write? | role |
|---|---|---|---|
| `fpaa-neural-net` | Anadigm AN231E04 (switched-capacitor FPAA, **owned**) | **No** — host writes weights over SPI = electronic = inference | representability ceiling, not learning |
| `06-crossbar-build` | discrete memristor (buyable, e.g. Knowm) | Yes — physical pulse write | first *buyable* in-fabric build |
| **`08-floating-gate`** (this) | analog floating-gate transistor array | **Yes** — charge by tunneling / injection | the *aspirational* in-fabric build |

The owned FPAA is switched-capacitor: every weight change is a host write, so by this
project's own audit it is inference, however well it classifies. A floating gate, by contrast,
stores charge that a physical write event changes — so it *can* keep the update in the fabric.
That is `08`'s entire reason to exist as distinct from `fpaa-neural-net`.

## Why aspirational

Analog floating-gate transistors are **not sold as discrete parts** — the analog form is
integrated only (academic FG-FPAAs, or a CMOS/MPW fab run). So unlike `06` (a buyable
memristor), `08` has **no device to characterize today**. Phase 0 runs against a *provisional*
read noise `σ_fg`; the verdict carries `provisional=True` until a real array confirms it. A
clean "no accessible floating-gate array at acceptable cost" is a valid Phase-1 FORECLOSE.

## The throughline (inherited, sharpened for hardware)

> **Who computes the update?** On a floating-gate array the intended answer is in-fabric: the
> weight is the charge on the gate, and the update is a physical tunneling / injection event
> driven by the measured free/nudged contrast. If a host PC computes the gradient and merely
> sets the charge, that is **inference, not learning**, and it fails this project's verdict.

## What `02` established (the starting line)

- **PASS** (parity-3): both a *half*-co-located Manhattan rule and a *fully*-co-located eqprop
  rule learn above the device floor under clean training.
- **Read-noise DIVERGENCE:** under read noise in the *training relaxations*, the
  **fully-co-located** rule (two independent reads, `dg ∝ Σ s(ΔVⁿ) − s(ΔVᶠ)`) forecloses ~100×
  below realistic, while the **half-co-located** rule (one read) stays robust. See `02`'s
  `readnoise.py` + the synthesis lesson `contrastive-update-is-read-noise-fragile`.

## The pre-registered staged verdict

A clean **FORECLOSE at any gate is a valid, money-saving end.** Verdicts are frozen in
`verdict.py` before each run.

- **Phase 0 — H5 device gate (sim, runs first).** Re-run `02`'s H5 sweep at a provisional
  `σ_fg` on the **half-co-located** object. **`classify_h5` PASS iff the half-co-located rule
  holds SNR ≥ 3 at `σ_fg`.** The fully-co-located rule is *expected* to collapse (consistency
  check). **Caveat:** floating-gate read noise is low, so an easy PASS is likely — it means
  "read noise is not the wall," not "the device is fine."
- **Phase 1 — access + transfer verdict (on H5 PASS).** Secure access to a real floating-gate
  array (FG-FPAA collaboration or MPW run), characterize it — read noise (confirming `σ_fg`)
  **and the write side** (programming precision/granularity, tunneling/injection asymmetry,
  endurance, retention) — re-sim at measured parameters, classify transfer.
- **Phase 2 — build (only on Phase-1 PASS).** Program the half-co-located learner on the
  floating-gate array; demonstrate parity-3 / XOR learned **in-fabric**, audited on both axes.

## Why this is a sound (if distant) build target

A floating-gate analog array used as a conductance crossbar is a **gradient flow by
construction** — it minimizes power/co-content — so equilibrium-propagation rigor is
*licensed* (cf. `gradient-flow-is-the-ep-license`). The floating gate only makes the
conductance non-volatile; it does not change the variational story. The open questions are
device access (Phase 1) and the write side (Phase 1/2), not the math.

## Layout

```
verdict.py        # FROZEN Phase-0 classify_h5 (provisional σ_fg; +Phase-1/2 to be frozen later)
tests/            # unit tests that brute-force the frozen decision function
docs/superpowers/
  specs/          # the design spec (2026-06-24)
  findings/       # Observed / Interpretation, filled as phases run
```

## Commons

Before starting work, read `../00-shared-harness/BULLETIN.md` and
`../00-shared-harness/synthesis/glossary.md`. Port the **two-axis `audit.py`** from `commons/`;
reuse `02`'s `readnoise.py` + `coupled.py` as the H5 sim reference. Report reusable findings
(esp. the *sim→hardware transfer* lesson family) up to the commons via the promotion-branch
handoff. Do **not** edit `06` or `fpaa-neural-net` — the commons↔sibling write boundary.
