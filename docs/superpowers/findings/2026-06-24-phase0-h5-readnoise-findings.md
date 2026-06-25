# Findings — Phase 0: H5 read-noise gate on a (provisional) floating-gate array

**Date:** 2026-06-24
**Scope:** Phase 0 of `08-floating-gate` — the pre-registered read-noise gate. Re-runs `02`'s
noisy-training read-noise sweep (vendored verbatim from `02 @95868e7`) at a **provisional**
floating-gate read noise `σ_fg = 0.01`, expressed in SNR units, and classifies with the **frozen**
`classify_h5` (`verdict.py`, committed before the run). The question: **does `02`'s buildable
*half*-co-located rule hold SNR ≥ 3 at `σ_fg`** on a floating-gate readout — and what does the
genuinely in-fabric *fully*-co-located rule do at the same point?
**Artifact:** `h5_sweep_parity3.json` (committed). Reproduce with `python h5_sweep.py`
(`run_h5(sigmas=DEFAULT_SIGMAS, H=8, a=0.4, V_drive=1.5, n_epochs=300, seeds=range(10), beta=0.1)`,
default `DeviceConfig`, `sigma_fg=0.01`). Serial run, ~11 min wall-clock on the dev box.

This document keeps **Observed** and **Interpretation** strictly separate (portfolio convention).
Each rule carries its two-axis audit. **half** (Manhattan, one read): `error_site=off_array_cpu`
(the error sign is an off-array gradient from **one** physical read) → audit `(cpu, physical)`,
`is_physical=False`. **full** (coupled, two reads): `error_site=on_array_local` (a local contrastive
difference of **two** independent physical relaxations) → audit `(physical, physical)`,
`is_physical=True`. `increment_site=physical_pulse` for both; sign pulse throughout.

> **Provisional, by construction.** No floating-gate array is owned or characterized; `σ_fg=0.01`
> is pre-registered, not measured. The verdict carries `provisional=True` and is reconciled with a
> real device at Phase 1 (`sigma_confirmed`). The device-independent content — *where each wall is*
> and *that the in-fabric rule is the fragile one* — is robust regardless of the exact `σ_fg`.

---

## Observed

### Consistency anchor: the vendored sim reproduces `02`'s Stage-5 numbers

`08` ran no new physics — it drove the verbatim port of `02`'s sim. The clean-training column and
the coupled collapse reproduce `02`'s Stage-5 `sep_mean` to the digit (built-in port check, on
production numbers):

| rule | `sep_mean` (train_σ=0) | `sep_mean` (train_σ=0.01) | `02` Stage-5 value |
|---|---|---|---|
| half (Manhattan, one read) | **0.083** | 0.109 | 0.083 / 0.109 |
| full (coupled, two reads)  | **0.092** | **−0.0072** | 0.092 / −0.007 |

The port is faithful: same code, same seeds, same outputs. `08`'s contribution is the *SNR framing*
and the *floating-gate re-keying*, not a new measurement.

### SNR vs training read noise (10 seeds, default β=0.1, eval held clean at `cfg.sigma_read=0.01`)

`SNR = separation / read-noise floor` (`02`'s `task.verdict`); the only swept variable is the
*training* read noise. `σ_fg = 0.01` is the pre-registered operating point (bold column).

| train_σ | 0.0 | 0.005 | **0.01** | 0.02 | 0.05 | 0.10 | 0.20 | wall (SNR<1) |
|---|---|---|---|---|---|---|---|---|
| **half** SNR  | 8.37 | 14.22 | **15.45** | 16.10 | 11.45 | 0.00 | 0.00 | **0.10** |
| **half** sep  | 0.083 | 0.108 | 0.109 | 0.106 | 0.070 | −0.019 | −0.023 | |
| **full** SNR  | 9.68 | 0.00 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 | **0.005** |
| **full** sep  | 0.092 | −0.008 | −0.007 | −0.007 | −0.007 | −0.007 | −0.008 | |

- **half:** SNR stays well above the PASS bar (3.0) across train_σ ∈ [0, 0.05] — and *rises* with a
  little training noise (8.37 at σ=0 → peak 16.10 at σ=0.02) — then collapses to 0 at σ=0.10. Wall
  at **σ=0.10** (≈10× `σ_fg`). Clean baseline (σ=0) already PASSes (8.37).
- **full:** SNR=9.68 at σ=0 (PASSes clean), then collapses to **0 at the first nonzero point
  (σ=0.005)** and stays there. Wall at **σ=0.005** — *below* `σ_fg`. The negative `sep` (≈−0.007)
  means the classes overlap (separation ≤ 0 → SNR 0 → FORECLOSE).

### half-vs-full gap and per-seed spread

- At `σ_fg=0.01`: **half SNR 15.45 (PASS) vs full SNR 0.00 (FORECLOSE)** — a clean split at the
  operating point.
- Wall ratio full/half = **0.005 / 0.10 = 0.05** — the in-fabric rule forecloses at ~1/20th the
  training read noise the half rule tolerates.
- Per-seed (10 seeds): clean (σ=0) both rules are seed-dependent but every seed PASSes — half SNR
  range 3.81–16.06, full 4.44–14.69. At each foreclosed point all 10 seeds read exactly 0 (total
  collapse, not a marginal average).

### Audit (recorded for both rules)

| rule | gradient source | write mechanism | `is_physical` |
|---|---|---|---|
| half (buildable; verdict keys on it) | `cpu` (off-array Jacobian sign, one read) | `physical` (pulse) | **False** |
| full (genuinely in-fabric) | `physical` (on-array local contrast, two reads) | `physical` (pulse) | **True** |

---

## Interpretation

**Headline: the read-noise gate PASSes on the buildable half-co-located object (SNR 15.4 ≫ 3 at
`σ_fg`), exactly the easy PASS the spec predicted — and it means "read noise is not the wall," not
"the device is fine."** The half rule tolerates ~10× the provisional `σ_fg` before its wall (0.10),
so a floating-gate readout's low read noise is comfortably survivable for the single-read object.
Phase 0 has done its job: it forecloses the *cheap* failure mode and pins the (provisional)
operating point. The floating-gate question is **not** decided here.

**The crux the elevated verdict makes explicit: the rule that survives read noise is *not* in-fabric,
and the rule that *is* in-fabric is the one that forecloses.** The half rule PASSes, but its gradient
is computed off-array on a CPU (`is_physical=False`) — by `08`'s own hard rule #2, a host-computed
gradient is *inference, not learning*. The fully-co-located coupled rule is the only `(physical,
physical)` object — a genuine in-fabric update by a local contrast of two physical reads — and at
`σ_fg` it **FORECLOSEs** (SNR 0), with its wall at σ=0.005, *below* the provisional realistic read
noise. So **a Phase-0 PASS does not certify in-fabric learning.** "Who computes the update?" on this
device model: the robust answer is *the CPU* (half); the in-fabric answer (full) does not survive
realistic read noise.

**This reproduces and re-keys `02`'s read-noise-fragility lesson, it does not add a new mechanism.**
The collapse is `02`'s `contrastive-update-is-read-noise-fragile`: the on-array update is the sign of
`s(ΔVⁿ) − s(ΔVᶠ)`, a difference of two nearly-equal noisy reads, which is catastrophically
read-noise-sensitive; the off-array update is the sign of one full-magnitude gradient, which is
robust (and even mildly dither-helped — hence the half rule's SNR *rising* at small training noise).
`08`'s only addition is to evaluate it at a floating-gate operating point and in SNR units.

**The full rule's reported wall (0.005) is grid-floor-limited — the true tolerance is finer.** `08`'s
grid floors at σ=0.005 and the full rule collapses at that first nonzero point, so 0.005 is an
*upper bound* on its tolerance, not the wall itself. `02`'s Stage-5x finer-σ sweep (down to 1e-5)
pinned the coupled rule's true tolerance at ~1e-4 — **~100× below realistic read noise** — i.e.
in-fabric contrastive learning is foreclosed *in practice* on this substrate. `08` inherits that
sharpening; it does not re-resolve it (no finer grid was run here).

**The binding floating-gate question is write-side, and it is Phase 1.** An easy read-noise PASS says
nothing about the floating-gate-specific non-idealities — programming asymmetry/granularity,
tunneling/injection write precision, endurance, retention. Those are the likely wall, and they
require a real device (or a measured write model), which Phase 0 deliberately does not touch.

**Caveats.**
1. **Provisional `σ_fg=0.01`** — pre-registered, not measured; verdict carries `provisional=True`.
   Floating-gate read noise is expected at or below this, so the half PASS is, if anything,
   conservative; but the operating point is not confirmed until Phase 1.
2. **parity-3 only**, default `DeviceConfig`, `a=0.4`, sign pulse, β=0.1. The comparison isolates
   *who computes the update*; it varies neither task nor device nor β (the spec dropped the β grid —
   `02`'s Stage-5/5x already showed β provides no rescue within range).
3. **The two-read asymmetry is intrinsic and intended** — eqprop *needs* two physical measurements;
   that is precisely why the in-fabric rule is the fragile one. It is the mechanism, not a thumb on
   the scale.
4. **The full wall (0.005) is an upper bound** (grid floor), per the Interpretation above.
5. **Eval held clean** (`cfg.sigma_read=0.01`, 64 reads) throughout; only *training* read noise moves.

---

## Verdict (recorded, not asserted)

- **PRIMARY — frozen `classify_h5`, keyed on the buildable half rule: PASS** (`snr_half=15.45 ≥ 3` at
  `σ_fg=0.01`; `provisional=True`; `full_rule_collapses_as_expected=True`). Read noise is **not** the
  wall for the half-co-located object → proceed to Phase 1, where the write side is the real test.
- **ELEVATED — the genuinely in-fabric (both-physical) full rule: FORECLOSE** (`snr_full=0.00` at
  `σ_fg`; wall ≤ 0.005, `is_physical=True`). The in-fabric rule does not survive realistic read noise
  on this device model.
- **Audit:** half `(cpu, physical)` → `is_physical=False`; full `(physical, physical)` →
  `is_physical=True`. The robust rule is not in-fabric; the in-fabric rule is fragile.
- **Net:** Phase 0 PASSes on the cheap (read-noise) axis and **does not** certify in-fabric learning.
  No FORECLOSE of the project — the foreclosure is of the *two-read in-fabric rule under read noise*,
  a known `02` result; the buildable object clears the gate. The decision shifts to Phase 1.

## Methodological notes (reusable)

- **The vendored port is its own consistency anchor.** `08`'s `sep_mean` reproduces `02`'s Stage-5 to
  the digit because it *is* `02`'s code (`vendor_02/ @95868e7`). Re-running a sibling's sim verbatim
  is a free correctness check on the port and keeps `08` from accidentally claiming new physics.
- **Positive control held:** both rules PASS at the clean baseline (σ=0: half 8.37, full 9.68), so
  the walls are genuine read-noise foreclosures, not training failures — the cheap signal that the
  runner trained at all. (The `--quick` smoke could not show this, by design.)
- **Coarse grid floors the in-fabric wall.** The full rule collapses at the first nonzero σ, so its
  reported wall is the grid floor, not the rule's tolerance; the true value (≈1e-4) lives in `02`'s
  finer sweep. When a wall pins to the grid floor, report it as an upper bound.

## What this licenses next

- **Phase 1 — secure device access and characterize the write side.** This is the real gate: an
  FG-FPAA collaboration/loan or an MPW shuttle run, then measure read noise (*confirming* `σ_fg`) **and**
  the write side (programming precision/granularity, tunneling/injection asymmetry, endurance,
  retention), re-sim at measured parameters, and freeze + classify `classify_transfer`. *"No accessible
  floating-gate array at acceptable cost"* is itself a valid FORECLOSE.
- **The open design problem Phase 0 sharpens:** a rule that is *simultaneously* in-fabric
  (`is_physical=True`) **and** read-noise-robust. Neither swept rule is both. Candidates for Phase 1/2:
  a single-read physics-local estimator of the contrastive signal, multi-read averaging (trading
  write energy for read SNR), or a floating-gate write mechanism whose charge update is itself robust
  to the two-read contrast.
- **Commons handoff (optional, sweep-time):** the *sim→hardware read-noise-transfer* observation (the
  in-fabric/robust trade-off re-confirmed at a floating-gate operating point) is a candidate for the
  knowledge commons via a promotion-branch handoff — not done here.
