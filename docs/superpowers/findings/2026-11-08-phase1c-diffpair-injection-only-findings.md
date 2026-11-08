# Findings — Phase 1c: the injection-only differential-pair learner at raw FG nonlinearity (H6)

**Date:** 2026-11-08
**Scope:** Phase 1c of `08-floating-gate`. Pre-registered in
`docs/superpowers/specs/2026-11-08-phase1c-diffpair-injection-only-design.md` (constants frozen in
`fg_diffpair.py`, committed `54640c7` **before** the run); verdict = the unchanged frozen `classify_transfer`.
**Artifact:** `diffpair_parity3.json` (committed). Reproduce with `python fg_diffpair.py` (`run_diffpair(H=8,
a=0.4, V_drive=1.5, beta=0.1, n_epochs=300, seeds=range(10))`, nominal `diffpair_nominal()` = Phase-1 FG nominal
with **raw** `nonlin=3.0`, `asym=1.0`, `n_levels=0`; `sigma_fg=0.01`; `HEADROOM=0.9`; `renorm_sigma=0.0011`
= 8-bit landing error). Serial, ~25 min wall-clock.

Observed and Interpretation are kept separate. Audit per rule as in Phase 0/1 — half `(cpu, physical)`,
full `(physical, physical)` — **plus** every global-erase re-inject recorded as `(cpu, physical)`.

> **Provisional, by construction.** No device measured; the pair model composes the vendored `02` device
> physics (`apply_pulse` potentiation branch) with the injection-only / global-erase facts of the RASP family
> taken from the papers. `params_confirmed=False`.

---

## Observed

### Gate + elevated (10 seeds, 300 epochs, eval clean at `cfg.sigma_read`)

| object | rule / audit | SNR @ raw ν=3, pair | erases / run | status |
|---|---|---|---|---|
| **GATE** half (buildable) | Manhattan sign, `(cpu, physical)` | **7.86** | **0.0** | **PASS** (`≥ 3`) |
| single-cell reference (Phase 1, same ν=3) | Manhattan sign | 0.20 | — | FORECLOSE |
| in-fabric two-read | coupled sign, `(physical, physical)` | 0.00 | 0.0 | FORECLOSE |
| in-fabric multi-read averaged | coupled + K ∈ {1, 100, 10⁴} | 0.00 at every K | — | FORECLOSE |

- `writes_per_cell = 300` (max injections per cell 300 + 0 erases) ≪ `ENDURANCE_BUDGET = 1e5` → `endurance_ok`.
- `in_fabric_achieved = False`. `provisional = True`.
- Audit at the nominal: half `cpu|physical: 24000`, `is_physical=False`; full `physical|physical: 24000`,
  `is_physical=True` (no renorm writes occurred at ν=3, so none were recorded).

### Observed: half-rule SNR (erases/run) vs raw ν — pair model vs Phase-1 single cell

| raw ν | pair SNR (erases/run) | single-cell SNR (Phase 1) |
|---|---|---|
| 0 | 17.45 (284) | 27.06 |
| 1 | 20.25 (258) | 25.87 |
| 2 | 15.82 (137) | 10.92 |
| **3** | **7.86 (0)** | **0.20** |
| 4 | 5.39 (0) | — |
| 5 | 1.64 (0) | — |

### Observed: half-rule SNR (erases/run) vs headroom at raw ν=3

| headroom | SNR | erases / run |
|---|---|---|
| 0.3 | 8.99 | 292 |
| 0.6 | 11.08 | 253 |
| 0.9 (nominal) | 7.86 | 0 |

---

## Interpretation

**1. H6's conclusion holds; H6's stated mechanism does not.** The pair learner PASSes at the raw
nonlinearity that foreclosed the single cell (7.86 vs 0.20), and the raw-ν cliff moves from between ν=2–3
(single cell) to between ν=4–5 (pair). But at ν=3 with `HEADROOM=0.9` **no renormalization ever fired**, so
"renorm keeps cells near G≈0" cannot be what did it. The arithmetic says why: with `dG0=0.05`, ν=3, a cell
pulsed every epoch obeys `(1−G)⁻² = 1 + 0.1·n`, so after 300 pulses `G ≈ 0.82 < 0.9`. What the pair *does*
provide is two things the single cell lacks: **(a) both update directions start at the bottom of the cell
range** (full step `dG0`) instead of mid-range (step `dG0·0.5^ν = dG0/8` at ν=3, in both directions — the
single cell was simply too slow), and **(b) a monotone, symmetric decay of the step with each cell's own
pulse count** — an implicit annealing schedule (final step ≈ `dG0·0.18³ ≈ 3·10⁻⁴`, a 170× decay). The
device sets that schedule, not the designer.

**2. Renormalization is a modest, expensive help, not the enabler.** Forcing erases (headroom 0.3/0.6)
raises SNR from 7.9 to 9.0/11.1 — at **250–290 global erase + re-inject cycles per 300 epochs**, i.e. the
whole array erased and re-programmed almost every epoch. On a real chip each cycle is a closed-loop
reprogram of every pair (seconds to minutes, tunneling-oxide stress each time). Within the 1e5 budget, but
it turns a "learner" into a host-reprogrammed array most of the time: every renorm write is `(cpu,
physical)`, so the in-fabric rule would lose both-physical status the moment one fires.

**3. At low raw ν the pair is *worse* than the single cell, and erases nearly every epoch.** With ν ≤ 1 the
step never decays, cells hit the headroom in ~18 pulses, and the array is erased 258–284 times per run; SNR
17–20 vs the single cell's 26–27 (the 8-bit landing error on every re-inject, applied to every cell ~every
epoch, is the likely cost). So the pair is the right model for a **strongly nonlinear raw device** (which the
FG family is) and a poor choice for a nearly-linear one.

**4. The in-fabric rule still forecloses, and the pair does not touch why.** Two-read contrast on the pair:
SNR 0 at every K up to 10⁴ — the same read-noise + write-side fragility Phases 0/1 found. The pair changes
the *write* model; it does not change the contrastive rule's fragility.

**5. "Who computes the update?"** Half: the CPU (on the board, the MSP430) computes the sign; the pair is
written by physical injection → `(cpu, physical)`, inference-grade. Full: forecloses. **Unchanged.**

**Caveats.**
1. **The pair's learning rate schedule is `n_epochs`- and `dG0`-dependent.** 300 epochs at `dG0=0.05`
   happen to anneal well; a longer run saturates cells (then renorm fires and cost rises), a larger `dG0`
   saturates sooner. The measured `dG0` and raw ν (bench rows "open-loop potentiation ladder") decide this.
2. **Provisional model.** Injection-only + global erase + raw ν=3 are literature/identity facts; the
   magnitudes are `02`'s defaults re-used. Pair mismatch (plus vs minus cell) is not modeled.
3. **parity-3, default topology, sign pulse, β=0.1**, as throughout.
4. Eval held clean at `cfg.sigma_read`; the pair's read noise `σ/√2` applies only to training reads.

---

## Verdict (recorded, not asserted)

- **PRIMARY — frozen `classify_transfer` on the pair half rule at raw ν=3: PASS (provisional)**
  (`snr_half = 7.86 ≥ 3`, `endurance_ok`). An **open-loop, single-pulse, injection-only** learner is viable
  on this device family as an inference-grade `(cpu, physical)` object — the single-cell FORECLOSE at raw ν=3
  was an artifact of the wrong write model, not of the device.
- **ELEVATED — in-fabric rules: FORECLOSE** (two-read SNR 0.0; robust 0.0 at every K). `in_fabric_achieved
  = False`.
- **Mechanism correction:** the pair wins by starting cells at the bottom of range with a device-set
  annealing decay, **not** by renormalization (0 erases at the nominal). Renormalization adds ~3 SNR at
  ~250–290 global erases per run.
- **Raw-ν cliff for the open-loop pair learner: between ν=4 (5.4, PASS) and ν=5 (1.6, BOUNDARY).**

## What this licenses next

- **The bench priority is unchanged and sharpened:** measure raw ν and `dG0` on the real part. If raw ν ≤ 4
  the open-loop pair learner is viable *without* closed-loop writes; if ν ≈ 5 it is marginal; if ν > 5 only
  closed-loop program-verify learns. Either way `transfer_sweep.py --measured --differential --write-mode
  open_loop` is the re-run.
- **Phase 2a (inference-grade build on the OTC2902K) has two candidate write modes now:** closed-loop
  program-verify (Phase 1 PASS, 10.9) and open-loop single-pulse pair (Phase 1c PASS, 7.9). The open-loop
  mode is faster per update (one pulse vs a measure-inject loop) and is the only one a future in-fabric cell
  could use. Freeze `classify_build` with both as arms, after the measured transfer verdict.
- **The in-fabric problem stays where Phase 1 left it:** a single-read, non-contrastive local update is
  still the open question; the pair model is the write side it would have to use.
- **Commons handoff (pending; harness not in this checkout):** *"on an injection-only FG array model the
  weight as a differential pair: the single-cell asym model FORECLOSES falsely at raw ν=3"*; *"a pair's
  step decays with its own pulse count — a device-set annealing schedule; check n_epochs × dG0 against the
  headroom before trusting a PASS"*; *"renormalization (global erase + re-inject) is a host write; audit it."*

## Methodological notes (reusable)

- **A confirmed hypothesis with the wrong mechanism is a half-result.** H6 predicted PASS *via* renorm; the
  erase count (0) falsified the mechanism while the gate passed. Recording erases/run alongside SNR is what
  caught it — instrument the mechanism, not only the outcome.
- **Write-model fidelity can flip a foreclosure.** Phase 1's "raw ν=3 forecloses" was true of the single-cell
  model and false of the device-faithful pair. Before declaring a device-family foreclosure, check that the
  write model matches the device's actual mechanisms (here: injection-only, global erase).
- **Count the maintenance ops.** 284 erases per 300 epochs is a result in itself; a PASS that needs it is a
  different object from one that does not.
