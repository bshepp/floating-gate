# Design — Phase 1: the transfer sim gate (`classify_transfer`) for a custom floating-gate array

**Date:** 2026-06-25 · **Status:** approved design, unrun. The Phase-1 sim gate of `08-floating-gate`.
**Parent specs:** `2026-06-24-floating-gate-hardware-transfer-design.md` (the staged-verdict charter;
§2 Phase 1) and `2026-06-24-phase0-h5-sweep-runner-design.md` (Phase 0, the read-noise gate).
**Follows:** Phase 0 — PASS (provisional) on the buildable half-co-located rule; the genuinely
in-fabric two-read rule forecloses under read noise (`docs/superpowers/findings/2026-06-24-phase0-h5-readnoise-findings.md`).

## 0. What this is, and the corrected device-access picture

The next step toward a real floating-gate (FG) array is **to design and tape out our own** — not on a
3D printer (a true analog FG is integrated CMOS), but as a **custom chip via an MPW (multi-project-wafer)
shuttle**. A floating gate is standard-CMOS (a poly gate with no DC path; charge moved by
tunneling/injection), so it is manufacturable on ordinary processes — this is exactly the parent
charter's Phase-1 "MPW shuttle run." The access landscape (verified 2026-06-25):

- **Cheap/open shuttles:** Tiny Tapeout (now on IHP `sg13g2` 130nm and GF180; SKY130 via ChipFoundry;
  supports analog/custom-GDS tiles; ~€70/tile). Note **Efabless/chipIgnite shut down March 2025**, so
  that specific SKY130 route is gone.
- **Academic/commercial MPW:** Europractice (`mini@sic`), IMEC IC-Link, MOSIS — many foundries,
  analog-capable processes (some with characterized analog-EEPROM/FG IP, e.g. X-Fab), $k–$10k+.

**Portfolio rule still binds: sim earns silicon.** Phase 0 PASSed (read noise is not the wall) but
left two things open that a tapeout depends on — the **write side** (programming granularity,
tunneling/injection asymmetry, update nonlinearity, write noise, retention, **endurance**) and the
**in-fabric-update problem** (the only genuinely in-fabric rule forecloses on read noise). This phase
**re-parameterizes `02`'s vendored device model with FG-realistic values, freezes `classify_transfer`,
re-sims, and classifies whether a tapeout is earned** — before any money is spent. A clean FORECLOSE
(the write side or endurance kills even the buildable learner) is a valid, money-saving end.

**Decision (hybrid, from brainstorming):** the gate keys on the **buildable** object (objective 1 —
"does the write side transfer?"); the run **also records, co-equally**, the in-fabric rule's fate and
one **read-noise-robust in-fabric candidate** (objective 2 — "could the tapeout be a genuine in-fabric
learner, or only an inference accelerator?"). This continues Phase 0's primary+elevated verdict pattern.

## 1. Parameter sourcing — literature ranges, process-agnostic, committed before the run

We are **not** pinning a process yet (no tapeout is committed). The useful question is *device-independent*:
**does any realistic FG write-side regime transfer?** So FG parameters are **ranges grounded in the
floating-gate literature** (Diorio/Mead synaptic transistor; Hasler floating-gate analog / RASP-FPAA;
analog-EEPROM datasheets), **swept** — mirroring how Phase 0 swept σ. The concrete ranges + citations are
**committed in `fg_params.py` before the sweep** (the pre-registered operating point, exactly as
`SIGMA_REALISTIC_FG` was committed before Phase 0). A specific process is pinned only *after* a PASS.

The verdict carries `params_confirmed=False` (literature values) → `provisional=True`; it flips to
confirmed only when a real device (a characterization tapeout, or a vendor-characterized FG process)
measures the parameters. The robust, device-independent output is *whether the write side transfers at
all, and at what cost* — not a single committed operating point.

## 2. The frozen verdict — `classify_transfer` (pre-registered, FROZEN before the run)

`classify_transfer` currently raises `NotImplementedError`. It is frozen in `verdict.py` before the
re-sim, with brute-force tests over its input space (as `classify_h5` has). It reuses the portfolio bar
(`PASS_SNR = 3.0`, `FORECLOSE_SNR = 1.0`) and is a pure function of reported summary statistics.

```
classify_transfer(snr_half, snr_full, snr_robust,
                  writes_per_cell, endurance_budget,
                  robust_reads_K, params_confirmed=False) -> dict
```

**Gate (keys on the buildable half-co-located object, subject to a hard endurance constraint):**
- `endurance_ok = writes_per_cell <= endurance_budget`.
- **FORECLOSE** if `not endurance_ok` (a training run cannot physically complete within the device's
  write-cycle budget) **or** `snr_half < FORECLOSE_SNR` (the write side kills even the buildable learner).
- **PASS** iff `endurance_ok and snr_half >= PASS_SNR` (the write side transfers; a tapeout is earned for
  at least an inference-grade array).
- **BOUNDARY** otherwise.

**Elevated records (co-equal, not gating — the in-fabric question):**
- `full_status` = band(`snr_full`) — the two-read in-fabric rule; expected FORECLOSE (read noise; Phase 0).
- `robust_status` = band(`snr_robust`) — the multi-read-averaged in-fabric candidate (§5).
- `in_fabric_achieved = (robust_status == "PASS")` — would the tapeout be a *genuine in-fabric learner*
  (and at what read cost `robust_reads_K`), or only an inference accelerator?
- `band(x)` = the same PASS/BOUNDARY/FORECLOSE thresholds applied to one SNR.

**Returns:** `status`, the three SNRs, `full_status`, `robust_status`, `in_fabric_achieved`,
`endurance_ok`, `writes_per_cell`, `endurance_budget`, `robust_reads_K`, `params_confirmed`,
`provisional = not params_confirmed`. `classify_build` (Phase 2) stays `NotImplementedError`.

## 3. FG parameter mapping — `02`'s `DeviceConfig` knobs ← FG characteristics

`02`'s vendored device model already carries the write-side non-idealities; Phase 1 sets them to
FG-realistic values instead of `02`'s memristor defaults. `fg_params.py` defines the mapping + the
committed ranges (literature-grounded, cited inline):

| FG characteristic | `DeviceConfig` knob | FG-realistic direction (ranges finalized + cited in `fg_params.py`) |
|---|---|---|
| Read noise | `sigma_read` | low — at or below Phase-0's `0.01` (FG read is quiet); swept e.g. `[0.001 … 0.01]` |
| Programming granularity / levels | `dG0`, `n_levels` | moderate; `02` showed ~4-bit (ν→0 co-design) suffices — swept around that floor |
| Update nonlinearity (ν) | `nonlin` | **higher** than `02`'s default — FG injection/tunneling is strongly state-dependent |
| Potentiation/depression asymmetry | `asym` | **strong** — tunneling-erase vs hot-carrier-injection differ in rate/granularity (a key FG non-ideality) |
| Cycle-to-cycle write noise | `sigma_write` | swept; FG injection has shot-noise/variability |
| Retention | `drift` | ≈ 0 — FG holds charge for years (the FG-*favorable* knob) |

The exact swept grids and point values are **committed in `fg_params.py` before the sweep** with their
literature citations; this spec fixes the *mapping and method*, not the numbers (pre-registration).

## 4. Endurance — the one genuinely new knob (a write-budget check)

`DeviceConfig` does not model endurance. Phase 1 adds a **lightweight write-budget check**, not
degradation dynamics: count `writes_per_cell` = the **maximum** program/erase pulses any *single* cell
receives over a training run (the worst-case cell hits the endurance limit first, so the max — not the
sum or mean — is the binding quantity), and compare against an `ENDURANCE_BUDGET` (FG program/erase
endurance, ~1e5–1e6 cycles, cited in `fg_params.py`). If `writes_per_cell > ENDURANCE_BUDGET`, the gate
FORECLOSEs (the device physically cannot complete the training). Rationale: a clean physical constraint
with minimal new assumptions; **degradation dynamics** (dG0 shrink / `sigma_write` growth past N cycles)
are a deferred refinement, noted in findings. Per-cell pulse counts are tallied from the vendored rules'
pulse activity (which cells `apply_pulse` actually fires each step), not modeled anew.

## 5. The read-noise-robust in-fabric candidate — multi-read averaging

Phase 0's sharpest open question: **is the in-fabric read-noise wall buyable?** The cheapest, most
quantifiable candidate is **multi-read averaging** — average `K` independent noisy reads per relaxation
before forming the two-read contrast, so the per-read noise scales as `σ/√K` (trading write energy/time
for read SNR). `fg_rules.py` implements this by **composing the vendored primitives** (`free_voltages`,
`nudged_solve`, `apply_pulse`, and an averaged `noisy_read`) in a new 08-authored rule — it does **not**
edit the frozen `vendor_02/` ports. The run reports `snr_robust` at the smallest `K` that clears the bar
(or that K is impractically large). This directly quantifies the cost: e.g. lifting the two-read rule from
its wall (~1e-4, per `02`'s 5.x) to `σ_fg ≈ 1e-2` needs `K ≈ (1e-2 / 1e-4)² = 1e4` reads/step — the
verdict states the number honestly, so "in-fabric is buyable but at K≈10⁴ reads" is a real, recorded
finding (a novel *single-read* estimator is the open alternative, deferred).

## 6. Components

- `verdict.py` — **FREEZE `classify_transfer`** (§2); `classify_build` stays `NotImplementedError`.
- `tests/test_verdict.py` — extend with brute-force tests for `classify_transfer` (PASS / FORECLOSE via
  SNR / FORECLOSE via endurance / BOUNDARY / elevated `full`/`robust`/`in_fabric_achieved` flags /
  provisional). Do not alter the frozen `classify_h5` tests.
- `fg_params.py` (new) — the FG→`DeviceConfig` mapping, the committed swept ranges + point values, the
  `ENDURANCE_BUDGET`, all with inline literature citations. Committed before the sweep.
- `fg_rules.py` (new) — the multi-read-averaged in-fabric rule, composing vendored primitives; plus the
  writes-per-cell counter. **Never edits `vendor_02/`.**
- `transfer_sweep.py` (new) — the Phase-1 runner, a sibling of `h5_sweep.py`: drives the vendored
  `train_*_noisy` at FG-parameterized `DeviceConfig` over the committed ranges + the multi-read rule,
  computes `snr_half`/`snr_full`/`snr_robust` + `writes_per_cell`, calls the frozen `classify_transfer`,
  records the two-axis audit per rule, emits `transfer_parity3.json` + a printed summary.
- `tests/test_transfer_sweep.py` (new) — structure/wiring/determinism tests on a tiny config (no science
  numbers), as for `h5_sweep`.
- `docs/superpowers/findings/` — Observed (transfer SNR vs FG params, the K-cost of multi-read, endurance
  margin, both audits) kept separate from Interpretation (silicon earned? inference-grade vs in-fabric?
  read-vs-write, provisional caveat). Written after the gated run.
- **Reuse, not reimplement:** the `vendor_02/` sim (`DeviceConfig`/`apply_pulse`/`train_*_noisy`/
  `evaluate_separation`/`task.verdict`), `audit.py`, and the `h5_sweep.py` driver idiom.

## 7. Risks / open questions

- **The gate may PASS shallowly.** FG retention is excellent and granularity is forgiving (`02`), so the
  buildable object likely transfers — greenlighting an *inference-grade* array. The elevated in-fabric
  record (§5) is what keeps a PASS honest about whether the tapeout serves 08's *actual* (in-fabric)
  question. Report both; do not let a write-side PASS read as "in-fabric learning works."
- **Parameters are literature estimates, not measured.** `provisional=True` until a real device confirms
  them (a characterization tapeout — milestone B — or a vendor-characterized FG process). The
  device-independent finding (does the write side transfer *at all*, and the K-cost of in-fabric) is
  robust regardless.
- **Endurance is modeled as a budget check, not degradation dynamics** (§4) — conservative and simple; a
  run that *barely* fits the budget but degrades mid-training is not yet modeled (deferred).
- **Multi-read averaging may price in-fabric out of reach** (K≈10⁴). That is itself the finding: it
  quantifies *how far* from buyable the in-fabric wall is, and licenses (or forecloses) the single-read
  estimator as the next idea.

## 8. Scope this session and after

This design spec only. On approval: the implementation plan (`writing-plans`) covers freezing
`classify_transfer` + tests, `fg_params.py` (sourced + committed), `fg_rules.py`, `transfer_sweep.py` +
tests. The **full transfer sweep run + findings are gated** behind an explicit go-ahead (as Phase 0 was).
`08` does not edit `06` or `fpaa-neural-net`, and never edits the `vendor_02/` ports.
