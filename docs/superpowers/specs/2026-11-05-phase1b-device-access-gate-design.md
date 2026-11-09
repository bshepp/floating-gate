# Design — Phase 1b: the device-access gate (`classify_access`) and the bench-characterization protocol

**Date:** 2026-11-05 · **Status:** approved design; `classify_access` frozen and run on observed inputs
(no purchase made). The access half of Phase 1 of `08-floating-gate`.
**Parent specs:** `2026-06-24-floating-gate-hardware-transfer-design.md` (charter; §2 Phase 1 "access +
transfer"), `2026-06-25-phase1-transfer-sim-gate-design.md` (the transfer half; PASS inference-grade,
provisional).
**Trigger:** the user reports that the floating-gate FPAA boards from the academic (Georgia Tech / Hasler)
papers are now sold commercially by Okika. Verified 2026-11-05 (sources in the findings note).

## 0. What changed, and what did not

**Changed.** The charter's premise *"analog floating-gate transistors are not buyable; the analog form is
integrated only (academic FG-FPAA or an MPW run)"* is no longer true in the way that matters. Okika Devices
sells the **OTC2902K** development board carrying the **OTC2902E04 SoC FPAA**, a floating-gate-programmed
mixed-signal FPAA whose block counts (56 CABs, 42 CLBs, on-chip MSP430, 350 nm) match the RASP 3.0 SoC FPAA
of George et al. (TVLSI 2016). List price $2,499. That closes the charter's *"no accessible floating-gate
array at acceptable cost"* foreclosure path and turns Phase 1's **characterization tapeout** milestone
(the Phase-1 findings' named next move) into a **board purchase plus a bench protocol**.

**Not changed.** Two things:

1. **The in-fabric question is still open — and the board does not settle it by existing.** On the
   published RASP infrastructure a floating gate is programmed by switching it *out of* the running
   circuit onto a programming drain line (program mode) and injecting under the on-chip programmer's
   control; the one published learning demonstration on this chip (Shah & Hasler, JETCAS 2018) computes
   the weight updates **digitally on the on-chip MSP430** and then programs the FGs. By this project's
   two-axis audit that is `(cpu, physical)` — the *half-co-located, inference-grade* object, exactly what
   the Phase-1 transfer gate already PASSed. Whether a **fabric-local analog signal can trigger injection
   on a weight FG in run mode** (the `(physical, physical)` write) is not established from the sources
   reachable here. It is the single vendor question that decides the in-fabric path.
2. **The portfolio rule.** Sim earned the device access (Phase 0 PASS, Phase 1 PASS inference-grade), and
   a verdict is still frozen before money moves. This spec freezes that verdict.

**A trap to name.** Okika also sells **FlexAnalog** boards ($156–$299) and a Raspberry Pi HAT. Those carry
the **Anadigm AN231E04** — Okika acquired Anadigm in April 2025 — the *switched-capacitor*, SRAM-weight part
this portfolio already classified inference-only (`fpaa-neural-net`). Same vendor, same "FPAA" word, not a
floating gate. The gate rejects it explicitly (`is_floating_gate=False` → FORECLOSE).

## 1. The two-axis analysis of the SoC FPAA as sold

| object | gradient | write | audit | can the OTC2902K host it as sold? |
|---|---|---|---|---|
| half-co-located (buildable) | CPU / on-chip MSP430 | FG injection via program mode | `(cpu, physical)` | **yes** (this is the published JETCAS-2018 pattern) |
| fully-co-located (in-fabric) | CAB-local analog contrast | injection triggered in run mode by that signal | `(physical, physical)` | **unknown** — needs `run_mode_local_write=True` |

So, as sold, the board's **audit ceiling is inference-grade** until the run-mode write is confirmed. A
purchase buys (a) the **measured operating point** (flips `params_confirmed`), and (b) a Phase-2a
inference-grade learner demo. It does **not** by itself buy in-fabric learning. The verdict records this
split so the purchase cannot be over-read.

## 2. The frozen verdict — `classify_access` (FROZEN 2026-11-05, before any purchase)

```
classify_access(is_floating_gate, price_usd, fg_programming_user_exposed,
                run_mode_local_write=None, budget_usd=ACCESS_BUDGET_USD) -> dict
```

- **FORECLOSE** if `not is_floating_gate` (wrong device), else if `not fg_programming_user_exposed`
  (the toolchain does not let us set individual FGs to our own targets and read them back → the write
  side cannot be measured → the board cannot do the characterization job), else if
  `price_usd > budget_usd`. Precedence in that order; reasons are returned.
- **BUY** otherwise: the board replaces the characterization-tapeout milestone.
- **Elevated, recorded co-equally, not gating:** `audit_ceiling` = `(physical, physical)` iff
  `run_mode_local_write is True`, else `(cpu, physical)`; `in_fabric_path` = `device` / `mpw` /
  `unknown`; `vendor_query_needed` = `run_mode_local_write is None`.

**`ACCESS_BUDGET_USD = 5_000`** — pre-registered as the *cheapest alternative route to the same
measurements*: the low end of an analog-capable MPW characterization-shuttle seat (the Phase-1 spec's own
"$k–$10k+" landscape). A board dearer than that route does not beat it. This is a modeling choice
committed before the decision, **not a purchase authorization** — the user ratifies the number (and the
purchase) explicitly. Tests brute-force the function (`tests/test_verdict.py`).

## 3. The bench-characterization protocol → `DeviceConfig` (`fg_measured.py`)

Each measurement inverts one equation of the vendored `02` device model so a real array feeds the
*unchanged, frozen* `classify_transfer` with `params_confirmed=True`:

| bench measurement | inversion | feeds |
|---|---|---|
| repeated reads of one cell | `std / span` | `sigma_read` (confirms Phase 0's `σ_fg`) |
| open-loop potentiation ladder | `step ∝ (1−G)^ν` → `nonlin_from_pot_steps`, `dG0_from_pot_step` | `nonlin_raw`, `dG0` |
| open-loop depression ladder at matched G | `asym_from_steps` | `asym` |
| repeated nominal pulses at mid-span | `std(step)/dG0` | `sigma_write` |
| closed-loop landing error over a target ladder | `levels_from_rms_error` (RMS = step/√12) | `n_levels`, `nonlin_effective` |
| retention over one training run's wall-clock | `drift_from_retention` | `drift` |
| datasheet / sacrificial-cell cycling | — | `ENDURANCE_BUDGET` |
| run-mode local write test | yes/no | `classify_access.run_mode_local_write` |

**Two write modes, by design.** The Phase-1 findings showed the raw FG nonlinearity forecloses (ν ≥ 3) while
the closed-loop *effective* nonlinearity learns. The bench protocol therefore measures both, and
`device_config_from_measurements(m, write_mode=...)` builds:
- `closed_loop` — each learner update is a program-verify write: effective ν, landing-error quantizer.
  This is the inference-grade `(cpu, physical)` path.
- `open_loop` — each learner update is one raw pulse with no verify: raw ν, no quantizer. This is the
  only write an in-fabric, fabric-triggered update can use. If the measured raw ν ≥ 3, the in-fabric
  object forecloses on this device family *regardless* of the run-mode-write answer, unless ν→0 pulse
  co-design is realizable.

The full protocol is data (`CHARACTERIZATION_PROTOCOL`) so spec, tests and a future runner read one list.
`transfer_sweep.py --measured measurements.json [--write-mode open_loop]` re-runs the frozen Phase-1 gate
on the measured config with `params_confirmed=True`.

**Expectations to test, not parameters.** Published numbers for this chip family: FG charge programmed
to ~10.2-bit accuracy; current sources within 2.2% RMS over 25 dB after calibration, and routinely <1%
over six decades (30 pA–30 µA). If those hold *as a fraction of the learner's span*, the measured
`n_levels` lands well above the 8-bit cliff Phase 1 found. The raw open-loop ν is **unpublished** in a form
this model can use and is the number most likely to bind.

## 4. Components

- `verdict.py` — `classify_access` + `ACCESS_BUDGET_USD` (frozen). Docstring carries the access correction.
- `fg_measured.py` — the inversions, `device_config_from_measurements`, `CHARACTERIZATION_PROTOCOL`.
- `fg_access.py` — the observed Okika inputs with per-input evidence grades (OTC2902K; FlexAnalog as the
  negative control), `run_access`, CLI → `access_okika.json` (committed).
- `transfer_sweep.py` — `run_transfer(cfg=...)` + `--measured` / `--write-mode`.
- `tests/test_fg_measured.py` (round-trips through the vendored model), `tests/test_fg_access.py`,
  additions to `tests/test_verdict.py` and `tests/test_transfer_sweep.py`.
- Findings: `docs/superpowers/findings/2026-11-05-phase1b-okika-access-findings.md`.

## 5. Pre-purchase checklist (user actions; each is one input to the gate)

1. **Individual-FG target API** — does the Okika toolchain release let us program a chosen FG element to
   an arbitrary target and read its current back (not only compiler-generated circuits)? → confirms
   `fg_programming_user_exposed`. (Evidence says yes; grade secondary.)
2. **Run-mode local write** — can a CAB-local analog signal trigger injection on a weight FG during run
   mode with no program-mode switch and no MSP430 in the loop? → `run_mode_local_write`. (Evidence points
   to no; unconfirmed.)
3. **Academic price and lead time** — the vendor advertises academic pricing on request; distributor
   single-unit prices are higher than list. → `price_usd`.
4. **Endurance rating** → `ENDURANCE_BUDGET` (currently the literature low end, 1e5).

## 6. Risks

- **Over-reading a BUY.** A BUY with `audit_ceiling=(cpu, physical)` is a characterization purchase and an
  inference-grade demo, not in-fabric learning. The verdict dict says so; the findings say so.
- **Span mismatch.** Published precision figures are relative to current (decades); the model's
  `n_levels` is relative to the learner's span. Measure in-span; do not import the published bits.
- **Global erase.** If tunneling erase is global on the part, depression is not per-cell; record it as a
  write-side wall for any pulse rule needing per-cell decrement.
- **Toolchain friction.** The open Scilab/Xcos + x2c flow targets circuits; low-level FG access may need
  the programming infrastructure directly. Budget time, not just money.

## 7. Scope this session

Freeze and run `classify_access` on observed inputs; write `fg_measured` + protocol; wire the measured path
into the frozen transfer gate; correct the access premise in README / CLAUDE / `verdict.py`. **No purchase,
no `classify_build` freeze** (Phase 2's protocol depends on the measured transfer verdict). `08` does not
edit `06` or `fpaa-neural-net`; the commons handoff is noted, not performed (harness not in this checkout).
