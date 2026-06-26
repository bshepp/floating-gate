# Findings — Phase 1: the transfer sim gate at floating-gate-realistic parameters

**Date:** 2026-06-25
**Scope:** Phase 1 of `08-floating-gate` — the pre-registered transfer gate. Re-sims `02`'s learning
rules (vendored verbatim from `02 @95868e7`) at a **floating-gate-realistic** `DeviceConfig` and
classifies, via the **frozen** `classify_transfer` (`verdict.py`, committed before the run), whether a
custom-silicon (MPW) tapeout is earned. The gate keys on the buildable half-co-located rule; the
genuinely in-fabric two-read rule and a multi-read-averaged robust candidate are recorded co-equally.
**Artifact:** `transfer_parity3.json` (committed). Reproduce with `python transfer_sweep.py`
(`run_transfer(H=8, a=0.4, V_drive=1.5, beta=0.1, n_epochs=300, seeds=range(10))`, nominal
`fg_device_config()`, `sigma_fg=0.01`). Serial run, ~22 min wall-clock on the dev box.

This document keeps **Observed** and **Interpretation** strictly separate (portfolio convention). Each
rule carries its two-axis audit: **half** (Manhattan, one read, off-array gradient) → `(cpu, physical)`,
`is_physical=False`; **full** (coupled, two reads, on-array local contrast) → `(physical, physical)`,
`is_physical=True`.

> **Provisional, by construction.** No floating-gate device is owned; the FG `DeviceConfig` (and the
> sweep ranges) are pre-registered **literature/modeling choices** (`fg_params.py`), not measured. The
> verdict carries `params_confirmed=False` → `provisional=True`; it is reconciled with a real device
> (a characterization tapeout, or a vendor-characterized FG process) at a later milestone. The
> device-independent content — *which non-ideality binds, and that the in-fabric rule forecloses* — is
> robust regardless of the exact operating point.

## The committed FG operating point (pre-registered in `fg_params.py`)

A **high-precision, closed-loop-programmed** floating gate. The raw FG update (Fowler–Nordheim
tunneling / hot-electron injection) is strongly nonlinear, but closed-loop program-verify compensates
it to land on each target level, so the *effective* update is only mildly nonlinear at high precision:
`nonlin=2.0` (effective), `n_levels=256` (~8-bit), `asym=0.3` (tunneling/injection asymmetry),
`sigma_write=0.03`, `sigma_read=σ_fg=0.01`, `drift=0.0` (FG retention is years). `ENDURANCE_BUDGET=1e5`
program/erase cycles. `g_min/g_max/dG0` keep `02`'s defaults.

---

## Observed

### Gate + elevated verdict (10 seeds, 300 epochs, eval clean at `cfg.sigma_read`)

| object | rule / audit | SNR @ FG nominal | status |
|---|---|---|---|
| **GATE** half (buildable) | Manhattan, `(cpu, physical)` | **10.92** | **PASS** (`≥ 3`) |
| in-fabric, two-read | coupled, `(physical, physical)` | **0.00** | FORECLOSE |
| in-fabric, multi-read averaged | coupled + K-read averaging | **0.00** (at every K up to 10⁴) | FORECLOSE |

- `endurance_ok = True` (`writes_per_cell = 300` ≪ `ENDURANCE_BUDGET = 1e5`).
- `in_fabric_achieved = False`. `provisional = True`.
- **The multi-read robust candidate forecloses at *every* K in the grid `[1 … 10⁴]`** — i.e. even at an
  effective read noise of `σ_fg/√10⁴ = 1e-4` (essentially clean reads), the two-read rule reads SNR 0
  at the FG nominal. (`robust_reads_K=1` is the bookkeeping for "no K cleared the bar; best = 0.0".)

### Observed robustness — half-rule SNR vs each FG knob (one-at-a-time around the nominal)

| knob | sweep (value : SNR) | shape |
|---|---|---|
| **`nonlin` (ν)** | 0.0:**27.06** · 1.0:25.87 · 2.0:10.92 · 3.0:**0.20** | **hard cliff** — forecloses by ν=3 |
| `n_levels` | 32:**0.00** · 64:0.00 · 128:2.16 · 256:10.92 · 512:11.30 | **precision cliff** — needs ~8-bit (256) |
| `asym` | 0.2:7.17 · 0.3:10.92 · 0.5:15.02 · 1.0:17.42 | graded (1.0 = symmetric is best); no wall in range |
| `sigma_write` | 0.02:8.28 · 0.03:10.92 · 0.05:12.28 · 0.1:13.14 | mild **help** (dither), monotone up |
| `sigma_read` | 0.003:**34.74** · 0.01:10.92 · 0.02:4.95 · 0.05:1.10 | graceful decay; near-foreclosure by 5% |

(`asym` is the depression/potentiation ratio per `02`'s device model: `1.0 = symmetric`, `< 1` =
asymmetric.) All rows are the half rule; eval is held clean (`cfg.sigma_read`), so the only variable per
row is the named knob.

---

## Interpretation

**Headline: the gate PASSes on the buildable object — but only as an *inference-grade* array, and only
because the FG's update nonlinearity is tamed.** Two things must both be read together:

**1. The write side transfers for the buildable (half-co-located) learner — a tapeout is earned, but
it is inference-grade.** At a realistic high-precision closed-loop FG operating point the half rule
learns parity-3 well above floor (SNR 10.9), and endurance is a non-issue for this task (300 writes vs a
10⁵-cycle budget). So an FG analog array *can* host the buildable learner. But that learner's gradient is
computed off-array on a CPU (`is_physical=False`) — by `08`'s hard rule #2 that is **inference, not
learning**. The PASS greenlights an FG **inference accelerator** (CPU gradient + non-volatile FG weight
storage), not an in-fabric learner.

**2. The genuinely in-fabric rule forecloses — and read-averaging cannot buy it back.** The
fully-co-located two-read rule (the only `(physical, physical)` object) reads SNR 0 at the FG nominal,
and the multi-read-averaged candidate **forecloses at every K up to 10⁴** — i.e. even with essentially
clean reads (effective `σ ≈ 1e-4`). This is *new* relative to Phase 0: there the in-fabric rule died on
**read noise**; here it dies even with read noise removed, so at the FG operating point it also dies on
the **write side**. The half (one-read) rule survives the FG nominal's 8-bit quantization (10.9) while
the two-read rule does not (0.0) — the contrastive two-read update is **more write-side-fragile** than
the one-read rule, on top of being read-noise-fragile. Multi-read averaging only addresses read noise,
so it is **ruled out as the in-fabric rescue** here.

**3. The binding non-ideality is update nonlinearity, not precision — overturning the Phase-1 premise.**
The spec expected an easy write-side PASS ("FG retention is great, granularity forgiving"). The `nonlin`
sweep refutes that: ν=0 learns superbly (27), ν=3 forecloses (0.2). A *raw* strongly-nonlinear FG update
cannot learn this task by the sign rule; learning requires reducing the **effective** nonlinearity —
either ν→0 pulse co-design (`02`'s Stage-3.x technique) or closed-loop program-verify (which compensates
the device nonlinearity to hit target levels, modeled here as the effective `nonlin=2` nominal). With the
nonlinearity tamed, programming **precision** is then a real but secondary requirement: ~8-bit
(`n_levels=256`) is needed (6-bit forecloses, 7-bit is marginal at 2.16). Asymmetry is a graded cost
(symmetric is best, but strong asymmetry still learns), write noise mildly *helps* (dither, as in `02`),
and the half rule's read-noise tolerance (~5% wall) is consistent with Phase 0.

**4. "Who computes the update?"** half = `(cpu, physical)` (inference-grade, survives the FG write
side); full = `(physical, physical)` (genuine in-fabric, but forecloses on both read noise and the FG
write side). At this operating point the FG array is realizable as an inference accelerator, **not** as
an in-fabric learner.

**Caveats.**
1. **Provisional FG parameters** — the nominal + sweeps are literature/modeling choices
   (`params_confirmed=False`). The whole gate hinges on whether a real closed-loop FG actually achieves
   the modeled effective nonlinearity (~ν=2) and precision (~8-bit) — a characterization tapeout must
   confirm it. The device-independent findings (nonlinearity is the binder; the in-fabric rule
   forecloses) are robust to the exact values.
2. **Only one robust-in-fabric candidate was tested** — multi-read averaging (a *read-side* fix). It is
   now ruled out, but a single-read physics-local / non-contrastive in-fabric update (a *different*
   candidate) was deferred and remains open (§ licenses).
3. **parity-3 only**, default `02` task/topology, sign pulse, β=0.1. Endurance is a budget check
   (`writes_per_cell = n_epochs`), not degradation dynamics; non-binding for this short task.
4. **Eval held clean** throughout; only the named knob (or training read noise, for the robust scan)
   moves per measurement.

---

## Verdict (recorded, not asserted)

- **PRIMARY — frozen `classify_transfer`, keyed on the buildable half rule: PASS (provisional)**
  (`snr_half = 10.9 ≥ 3` at the FG nominal; `endurance_ok = True`; `provisional = True`). The write side
  transfers for the buildable object → a tapeout is **earned for at least an inference-grade FG array**.
- **ELEVATED — the genuinely in-fabric rules: FORECLOSE.** Two-read `snr_full = 0.0`; the multi-read
  robust candidate `snr_robust = 0.0` at every K up to 10⁴ → `in_fabric_achieved = False`. The in-fabric
  rule dies on **both** read noise (Phase 0) **and** the FG write side (Phase 1); read-averaging cannot
  rescue it.
- **Binding non-ideality: update nonlinearity** (ν=3 forecloses even the half rule; ν→0 / closed-loop
  compensation required), with programming precision (~8-bit) a secondary requirement. **Not** retention
  or endurance (both FG-favorable here).
- **Audit:** half `(cpu, physical)` → `is_physical=False`; full `(physical, physical)` →
  `is_physical=True` but foreclosed. The viable FG learner at these params is **inference-grade**, not
  in-fabric.
- **Net:** Phase 1 PASSes the *cheap, inference-grade* question and **does not** earn an
  in-fabric-learning build. No project FORECLOSE — the buildable object clears the gate — but the
  in-fabric ambition (08's reason to exist) is **not** met at this operating point, and the obvious
  read-side rescue is ruled out.

## Methodological notes (reusable)

- **A pre-registered operating point can be a dead zone — verify it learns before the run.** The
  originally-committed FG nominal (`n_levels=32, nonlin=3`) foreclosed *vacuously* (half SNR 0) with the
  entire sweep grid inside the SNR=0 region; the final whole-branch review caught it, and direct sim
  runs confirmed it (`n_levels=32` floors even `02`'s default 10.2→0.0; `nonlin=3` forecloses even at
  *continuous* precision). The operating point was corrected **pre-run** (no findings yet, so not
  retuning-to-a-result) to a learning regime with sweeps that span the cliffs. Lesson: a pre-registered
  device operating point needs a σ=0 / nominal positive-control check that it learns at all, exactly as
  a learning-rate or budget needs a convergence gate.
- **The vendored sim is its own anchor.** `08` ran no new physics — `02`'s code re-parameterized.
- **Read-averaging is a read-side fix only.** When an in-fabric rule forecloses for a write-side reason,
  multi-read averaging (σ→σ/√K) cannot rescue it — confirmed here (FORECLOSE at K=10⁴). Distinguish
  read-side from write-side foreclosure before proposing a read-side remedy.

## What this licenses next

- **A build is earned only for an inference-grade FG array, not an in-fabric learner.** Two honest
  paths: **(a)** accept the inference-grade object (half-co-located, FG weight storage + physical write,
  CPU gradient) and tape *that* out — a real, modest result that drops 08's in-fabric thesis; or **(b)**
  keep the in-fabric ambition and attack the **open robust-update problem** — a *single-read*
  physics-local / non-contrastive in-fabric update that survives **both** read noise and the FG write
  side. Multi-read averaging is now ruled out as that rescue.
- **A characterization tapeout (the cheap-shuttle milestone) to confirm the provisional FG parameters** —
  specifically whether a real closed-loop FG achieves effective ν≈2 and ≥~8-bit precision, since the
  entire gate hinges on it. This flips `params_confirmed → True`.
- **Commons handoff (optional, sweep-time):** candidate lessons — *"for an in-fabric FG learner the
  binding write-side non-ideality is update nonlinearity, not precision/retention"* and *"multi-read
  averaging cannot rescue an in-fabric contrastive rule that forecloses on the write side."* Not done
  here.
