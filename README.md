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
**Phase 1b** (device access, 2026-11-05) → the "not buyable" premise is **corrected**: Okika Devices now sells
the Georgia Tech / Hasler floating-gate SoC FPAA (RASP 3.0 lineage) as the **OTC2902K** board ($2,499 list).
The frozen `classify_access` reads **BUY** for the characterization job (it replaces the characterization
tapeout) with an **inference-grade audit ceiling** `(cpu, physical)` — as published, FG writes go through
program mode under the on-chip MSP430, so in-fabric learning is *not* established by the purchase; the
run-mode local write is **not** available on the published design (answered 2026-11-08 by identity with
the papers: injection is program-mode only, under the programming infrastructure), so the in-fabric path is a
custom cell (MPW) and the board is the characterization instrument + inference-grade host. The family is
**injection-only with global erase**, so the device-faithful weight is a differential pair (Phase 1c,
`fg_diffpair.py`). **Nothing bought yet.** Beware the same vendor's
FlexAnalog boards: they are the Anadigm AN231E04 switched-cap part, not floating gates. See
`docs/superpowers/findings/2026-11-05-phase1b-okika-access-findings.md`.

## The three crossbar cousins (and why only two can learn)

`02`'s sim has three real-device descendants, split by device physics:

| sibling | device | in-fabric write? | role |
|---|---|---|---|
| `fpaa-neural-net` | Anadigm AN231E04 (switched-capacitor FPAA, **owned**; sold since 2025 as Okika *FlexAnalog*) | **No** — host writes weights over SPI = electronic = inference | representability ceiling, not learning |
| `06-crossbar-build` | discrete memristor (buyable, e.g. Knowm) | Yes — physical pulse write | first *buyable* in-fabric build |
| **`08-floating-gate`** (this) | analog floating-gate transistor array (now buyable: Okika OTC2902K SoC FPAA) | **Yes** — charge by tunneling / injection | the in-fabric build; inference-grade host as sold |

The owned FPAA is switched-capacitor: every weight change is a host write, so by this
project's own audit it is inference, however well it classifies. A floating gate, by contrast,
stores charge that a physical write event changes — so it *can* keep the update in the fabric.
That is `08`'s entire reason to exist as distinct from `fpaa-neural-net`.

## Why it was aspirational — and what changed (2026-11-05)

Analog floating-gate transistors are **not sold as discrete parts** — the analog form is
integrated only (FG-FPAAs, or a CMOS/MPW fab run). When this project started (2026-06) no such
array was purchasable, so Phase 0/1 ran against a *provisional* operating point (`provisional=True`
until a real array confirms it), and "no accessible floating-gate array at acceptable cost" was a
valid Phase-1 FORECLOSE.

**That path is now closed:** Okika Devices sells the Georgia Tech SoC FPAA (George et al., TVLSI 2016)
as the OTC2902K board. The access decision is itself a frozen verdict (`classify_access`, spec
`2026-11-05-phase1b-device-access-gate-design.md`), and the bench protocol that turns board
measurements into the vendored `DeviceConfig` lives in `fg_measured.py`
(`python transfer_sweep.py --measured measurements.json` re-runs the frozen Phase-1 gate with
`params_confirmed=True`). What the board does **not** settle is the in-fabric write: as published, FG
programming happens in program mode under the on-chip microprocessor, i.e. `(cpu, physical)`.

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
  **Phase 1b (access, 2026-11-05):** `classify_access` frozen before any purchase — BUY iff floating
  gate ∧ FG programming user-exposed ∧ price ≤ pre-registered budget; records the audit ceiling the
  board can host as sold. Run on the Okika OTC2902K → BUY, ceiling `(cpu, physical)`, in-fabric path `mpw`.
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
verdict.py        # FROZEN classify_h5 (Phase 0), classify_transfer (Phase 1), classify_access (Phase 1b)
fg_params.py      # pre-registered (provisional) FG operating point + sweeps
fg_rules.py       # read-averaging model + endurance write-count
fg_measured.py    # bench measurements -> vendored DeviceConfig (+ CHARACTERIZATION_PROTOCOL)
fg_access.py      # observed Okika inputs through classify_access -> access_okika.json
h5_sweep.py       # Phase-0 runner -> h5_sweep_parity3.json
transfer_sweep.py # Phase-1 runner -> transfer_parity3.json (--measured re-runs on a real device)
vendor_02/        # 02's sim, vendored verbatim
tests/            # one pytest module per source module; the verdicts are brute-forced
docs/superpowers/
  specs/          # design specs (2026-06-24, 06-25, 11-05)
  findings/       # Observed / Interpretation, filled as phases run
```

## Commons

Before starting work, read `../00-shared-harness/BULLETIN.md` and
`../00-shared-harness/synthesis/glossary.md`. Port the **two-axis `audit.py`** from `commons/`;
reuse `02`'s `readnoise.py` + `coupled.py` as the H5 sim reference. Report reusable findings
(esp. the *sim→hardware transfer* lesson family) up to the commons via the promotion-branch
handoff. Do **not** edit `06` or `fpaa-neural-net` — the commons↔sibling write boundary.
