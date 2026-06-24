# Design — 08-floating-gate: sim→hardware transfer toward an in-fabric floating-gate learner

**Date:** 2026-06-24 · **Status:** approved design (with two fact-forced corrections, §1); Phase 0 pre-registered, unrun.
**Lineage:** the **floating-gate** hardware-transfer arm of `02-memristive-crossbar` — the
*in-fabric-capable* cousin of the switched-capacitor FPAA (`fpaa-neural-net`, inference-only)
and of the buyable-memristor arm (`06-crossbar-build`).

## 0. Why this project exists as a distinct sibling

The portfolio's "analog crossbar hardware" splits **by device physics** (the decomposition
rule: *split where the device model or the scientific claim genuinely differs*). Three cousins
realize `02`'s sim on three real devices, and only two of them can *learn*:

| sibling | device | in-fabric write? | role |
|---|---|---|---|
| `fpaa-neural-net` | Anadigm AN231E04 (switched-capacitor FPAA, **owned**) | **No** — host writes weights over SPI = electronic = inference | representability ceiling, not learning |
| `06-crossbar-build` | discrete memristor (buyable, e.g. Knowm) | Yes — physical pulse write | first *buyable* in-fabric build |
| **`08-floating-gate`** | analog floating-gate transistor array | **Yes** — charge written by tunneling / hot-carrier injection | the *aspirational* in-fabric build |

`08`'s reason to exist: a floating-gate cell stores an analog weight as **non-volatile charge
on an isolated gate**, updated by a *physical* tunneling/injection event. That is a genuine
in-fabric increment — unlike the switched-cap FPAA, which can only be host-written. So `08` is
the floating-gate answer to the portfolio question, *who computes the update?*

## 1. Two fact-forced corrections to the original framing

This project was first imagined (2026-06-23) as the *owned* FPAA, build-first, with a
**measured** `σ_dev`. Two facts (surfaced by `06`'s 2026-06-24 reframe) correct that:

1. **The owned FPAA is switched-capacitor (Anadigm AN231E04), not floating-gate** — its weights
   are capacitor ratios in volatile SRAM, host-written over SPI. Every weight change is an
   electronic write → by the two-axis audit that is `(… × electronic_write)` = inference, not
   learning. It *structurally* cannot keep the update in the fabric and is the separate
   `fpaa-neural-net` project, **not** `08`.
2. **Analog floating-gate transistors are not buyable as discrete parts** — the analog form
   exists only integrated (an academic FG-FPAA, e.g. the Georgia Tech / Hasler line, or a CMOS
   MPW/fab run). So `08` has **no owned device to characterize today**.

Consequently `08` is **aspirational and trailing**, with a **provisional** operating point
(`07`-style `sigma_confirmed` flag), not a device-characterized one. A clean Phase-0/1
foreclosure — including *"no accessible floating-gate array exists at acceptable cost"* — is a
valid, money-saving end (portfolio discipline: *foreclosure is a result*).

## 2. The claim and the verdict-question (staged; foreclosure valid at every gate)

- **Phase 0 — H5 sim gate (runs first).** `classify_h5` (frozen in `verdict.py`) PASSes iff
  `02`'s **half-co-located** rule holds **SNR ≥ 3** at a provisional floating-gate read noise
  `σ_fg`. The fully-co-located two-read rule is *expected* to collapse (`02` already foreclosed
  it ~100× below realistic) — confirming that is a consistency check, not a target. FORECLOSE if
  even the half-co-located rule dies at `σ_fg`.
- **Phase 1 — access + transfer verdict (on H5 PASS).** Gain access to a real floating-gate
  array (academic FG-FPAA collaboration/loan, or an MPW shuttle run), characterize it (read
  noise — now *confirming* `σ_fg` — **and the write side**: programming precision/granularity,
  tunneling/injection asymmetry, endurance, retention), re-sim at measured parameters, classify
  transfer.
- **Phase 2 — build (on Phase-1 PASS).** Program the half-co-located learner on the floating-gate
  array; demonstrate parity-3 / XOR learned **in-fabric**, audited on both axes.

## 3. Why Phase 0 is the highest-value first move — with a floating-gate caveat

Phase 0 costs nothing (it reuses `02`'s sim) and gates everything downstream. But a method
caveat must be recorded so a PASS is not over-read: **floating-gate cells are typically *low*
read-noise and non-volatile (good retention)**, so the H5 *read-noise* gate is **likely to PASS
easily**. For a floating-gate device the binding non-ideality is more likely on the **write
side** — slow, asymmetric, granular tunneling/injection programming and finite write endurance —
which is a **Phase-1** characterization question, not a Phase-0 read-noise one. Running H5 first
is still correct (it forecloses the cheap failure mode and pins the operating point), but the
spec states plainly that an easy Phase-0 PASS means *"read noise is not the wall,"* **not** *"the
device is fine."*

## 4. The two certification axes (both mandatory)

1. **The update stays in the fabric (two-axis audit).** Port `commons/audit.py`. The intended
   in-fabric mechanism is a **floating-gate charge update** (tunneling/injection): a physical
   read of node states + a physical charge write. A host-computed gradient with a host-set charge
   target is `(electronic_gradient × electronic_write)` = inference, and fails the verdict —
   exactly the failure that disqualified the switched-cap FPAA.
2. **Gradient-flow license.** A floating-gate analog array used as a conductance/transconductance
   crossbar is still a **conductance network → minimizes power/co-content → gradient flow by
   construction → EP rigor licensed** (cf. commons lesson `gradient-flow-is-the-ep-license`; same
   construction as `02`/`06`, unlike `05`'s driven net). The floating gate only sets the
   conductance *non-volatilely*; it does not change the variational story. State it in findings.

## 5. Phase 0 — the H5 read-noise gate (the only phase specified in full now)

**Procedure.**
1. **Reuse `02`'s H5 sim** directly — `02/readnoise.py` (`run_readnoise`, `train_*_noisy`,
   `noisy_read`) + `02/coupled.py` — no reimplementation. The half-co-located object is `02`'s
   one-read Manhattan rule; the fully-co-located object is its two-read coupled rule
   (`dg ∝ Σ s(ΔVⁿ) − s(ΔVᶠ)`).
2. **Sweep training read-noise σ** over a fraction-of-span grid to locate the foreclosure wall
   for each rule, and evaluate SNR (separation / noise-floor) at the provisional `σ_fg`.
3. **Classify** with the frozen `classify_h5`: PASS iff `snr_half ≥ 3` at `σ_fg`; record
   `snr_full` and its expected collapse as a consistency check; carry the `provisional` flag
   until a real floating-gate device confirms `σ_fg` at Phase 1.
4. Record **Observed** (SNR curves, wall locations, half-vs-full gap) and **Interpretation**
   (build/redirect decision, plus the read-vs-write caveat from §3) separately.

**Pre-registered constants** (committed in `verdict.py` before the run): `PASS_SNR = 3.0`,
`FORECLOSE_SNR = 1.0`, `SIGMA_REALISTIC_FG = 0.01` (provisional — floating-gate read noise is
expected at or below this; reconciled with the measured device at Phase 1, `sigma_confirmed`).

## 6. Components

- `verdict.py` — FROZEN `classify_h5` now. Structure = `06`-original's half/full object +
  `07`-style provisional handling:

  ```python
  PASS_SNR = 3.0
  FORECLOSE_SNR = 1.0
  SIGMA_REALISTIC_FG = 0.01   # provisional, reconciled at Phase 1

  def classify_h5(snr_half, snr_full,
                  sigma_fg=SIGMA_REALISTIC_FG, sigma_confirmed=False):
      # PASS if snr_half >= PASS_SNR; FORECLOSE if snr_half < FORECLOSE_SNR; else BOUNDARY
      # returns: status, snr_half, snr_full, sigma_fg, sigma_confirmed,
      #          full_rule_collapses_as_expected (snr_full < FORECLOSE_SNR),
      #          provisional (not sigma_confirmed)
      ...
  ```
  `classify_transfer` / `classify_build` raise `NotImplementedError` until their phases'
  protocols are finalized (pre-registering a threshold for an undesigned measurement is theater,
  not discipline).
- `tests/test_verdict.py` — brute-forces the frozen decision function (PASS / FORECLOSE /
  BOUNDARY / threshold edges / provisional + consistency flags / later phases not yet frozen).
- *(deferred to writing-plans / execution)* `h5_sweep.py` — a thin driver over `02`'s
  `run_readnoise` that emits the `classify_h5` inputs + a JSON dump. **Not written this session
  (no runs).**
- Ported, not reimplemented: `commons/audit.py`; `02/readnoise.py` + `02/coupled.py` as the H5
  sim reference.

## 7. Risks / open questions

- **Access is the real gate, not read noise.** The likely outcome is an easy Phase-0 PASS
  followed by a Phase-1 question that is as much *logistical* (can we get a floating-gate array at
  acceptable cost — an FG-FPAA collaboration or an MPW run?) as physical. "No accessible device"
  is an honest FORECLOSE and a portfolio-valid end.
- **The wall is probably write-side.** Programming asymmetry/granularity, endurance, and
  retention are the floating-gate-specific risks; they are Phase-1, and `08` must not claim device
  viability from a read-noise PASS alone (§3).
- **`σ_fg` is provisional.** The Phase-0 verdict carries `provisional=True` until a real device
  confirms it; the device-independent finding (wall location + half-vs-full mechanism) is robust
  regardless.

## 8. Scope this session

The design spec (this document) only. The frozen `verdict.py` + its test, the rest of the
scaffold (SEED, README, CLAUDE with commons pointer, conftest, requirements, .gitignore,
findings/), the GitHub repo, and the H5-sweep runner follow **on approval**; the H5-runner
**writing-plans** is the sibling's own job (it self-initializes from this spec + SEED), as with
`06`/`07`. **No sweeps run.** `08` does **not** edit `06` or `fpaa-neural-net` — the
commons↔sibling write boundary.
