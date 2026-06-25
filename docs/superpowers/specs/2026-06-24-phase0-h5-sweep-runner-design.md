# Design — Phase-0 H5 sweep runner (`h5_sweep.py`)

**Date:** 2026-06-24 · **Status:** approved design, unrun.
**Parent spec:** `2026-06-24-floating-gate-hardware-transfer-design.md` (§5 specifies the Phase-0
H5 read-noise gate; this document designs the runner that §6 deferred to writing-plans).
**Lineage:** drives `02-memristive-crossbar`'s read-noise sim (`@95868e7`) to compute the inputs
to the **frozen** `classify_h5`.

## 0. Scope

Build and test `h5_sweep.py` — the Phase-0 driver that produces `snr_half` / `snr_full` for the
frozen `classify_h5`, plus the *Observed* curves the parent spec requires. Plus the test module
and the vendored sim it drives. **A quick smoke run validates the runner; the full Phase-0 sweep
and its findings are a separate, gated follow-on** (do not commit compute to the full grid without
the user's go-ahead). No change to the frozen `verdict.py`.

## 1. Reuse mechanism — PORT, not import

Portfolio law (`00-shared-harness/commons/README.md`): shared code is *"consumed by **port, not
import**, because each project is an independent repo."* The siblings are separate git repos
(each remote-backed; `02` = `bshepp/memristive-crossbar`) that merely sit co-located on disk. A
cross-repo `sys.path` import would couple `08` to a checkout of `02` that may not exist. So `08`
**vendors** the sim it needs, exactly as `02/task.py` was itself vendored from `self-assembly`.

**Vendored set — the transitive closure of `02`'s noisy-training sweep** (`readnoise` + `coupled`
+ `manhattan`), copied **byte-identical from `02 @95868e7`** into `vendor_02/`:

```
vendor_02/
  readnoise.py   # the H5 sim: train_{manhattan,coupled}_noisy, noisy_read, run_readnoise
  coupled.py     # fully-co-located two-read rule (free/nudged co-content difference)
  manhattan.py   # half-co-located one-read rule + evaluate_separation + init_conductances
  crossbar.py    # differential crossbar topology + forward + read nonlinearity
  solver.py      # Laplacian assembly / batched node-voltage solve
  jacobian.py    # analytic output Jacobian (off-array gradient for Manhattan)
  memristor.py   # DeviceConfig + apply_pulse / read_conductance / apply_drift
  task.py        # parity3_data + separation + verdict (SNR = separation / floor, k=3)
  ceilings.py    # vendored only so manhattan.py stays byte-identical; never called by the H5 path
  README.md      # provenance: ported from 02 @95868e7, file list, "re-vendor on upstream change"
```

Each `.py` gets a one-line provenance header:
`# PORTED from 02-memristive-crossbar/<file> @95868e7 — do not edit to diverge; re-vendor on upstream change.`
`ceilings.py` is included **solely** to keep `manhattan.py` unmodified (the H5 path never calls
`de_ceiling` / `sgd_ceiling`); this buys **zero adaptation** — every vendored file is a clean copy,
so re-vendoring on an upstream change is a straight overwrite.

**Audit hook**, vendored from the *commons* (the canonical source), not from `02`:
`commons/audit.py @e572b0f` → `08/audit.py`, same provenance-header convention.

**Wiring.** `conftest.py` gains one line putting `vendor_02/` on `sys.path` (alongside the repo
root it already adds), so the vendored modules resolve each other's bare imports
(`from crossbar import …`) unchanged, and `h5_sweep.py` / tests import them as flat modules. No
name collisions: `08`'s own modules are `verdict.py`, `h5_sweep.py`, `audit.py`, `conftest.py` —
disjoint from the vendored set. Outcome: `08` runs its sim hermetically with no `02` checkout.

## 2. `h5_sweep.py` — an SNR-curve driver

A thin orchestration layer; **no physics is reimplemented** — every model call lives in the
vendored modules. The verdict consumes SNR = `separation / floor`, but `02`'s `run_readnoise`
reports `sep_mean` / `frac_pass` and discards the floor, so the driver computes SNR itself from
`02`'s lower-level functions (which it already exposes).

**Per `(rule, train_sigma)` cell** — for `rule ∈ {half = Manhattan, full = coupled}` and each
`train_sigma` on a fraction-of-span grid that **includes `σ_fg = SIGMA_REALISTIC_FG = 0.01`**:

1. For each seed: train at that `train_sigma` via the vendored `train_manhattan_noisy` /
   `train_coupled_noisy` (default `02` config: `H=8`, `a=0.4`, `V_drive=1.5`, `beta=0.1`,
   `DeviceConfig()`), returning the trained `g`.
2. Evaluate clean-convention via the vendored `evaluate_separation` → `(sep, floor)`; `floor` is
   the std of two-sided separation across 64 noisy reads at `cfg.sigma_read = 0.01` (the eval read
   noise). `snr = sep / floor` (using `02.task.verdict`'s definition, `floor` clamped ≥ 1e-9).
3. Aggregate `snr_mean` over seeds; record the per-seed SNRs.

Training read noise = the swept `train_sigma`; eval is held clean — exactly `02`'s Stage-5
convention (the H5 hypothesis is about read noise *in the training relaxations*). Determinism: the
eval RNG is a fresh fixed-seed generator (mirroring `02`'s `_readnoise_sweep`); training seeds are
explicit. Same inputs ⇒ identical numbers.

**Outputs computed once over the grid (the *Observed* layer, parent spec §5.4):**
- **SNR curves** — `snr_mean` vs `train_sigma` for each rule.
- **Foreclosure wall** per rule — the first `train_sigma` whose `snr_mean < FORECLOSE_SNR (=1)`
  (`None` if it never forecloses in-range), computed by a small pure helper.
- **half-vs-full gap** — half and full SNR side by side at `σ_fg` and the ratio of their walls.

## 3. Verdicts — frozen primary + elevated in-fabric, both recorded

- **Frozen primary verdict (keys on the buildable object).** Call the frozen
  `classify_h5(snr_half, snr_full, sigma_fg=0.01, sigma_confirmed=False)` with the `σ_fg`-row SNRs.
  Returns PASS / BOUNDARY / FORECLOSE on `snr_half`, with `provisional=True` and
  `full_rule_collapses_as_expected`. `verdict.py` is **unchanged and frozen**.

- **Elevated in-fabric verdict (co-equal record).** The fully-co-located rule — the only
  *both-physical*, physics-local one — gets a **first-class** recorded read-noise verdict, not just
  the existing boolean: its `snr_full` at `σ_fg`, its foreclosure-wall `train_sigma`, and its
  `02.task.verdict` status. This foregrounds the project's crux: the read-noise-robust rule is
  **not** in-fabric, and the genuinely in-fabric rule is the fragile one.

- **Two-axis audit (ported `audit.py`), recorded for BOTH rules.** Mapping `02`'s audit dicts onto
  the canonical poles (glossary: *physics-local* = derivable only from quantities present at the
  element):
  - half / Manhattan: `error_site=off_array_cpu`, `increment_site=physical_pulse`
    → `Audit.record("cpu", "physical")` ⇒ `is_physical() == False`.
  - full / coupled: `error_site=on_array_local`, `increment_site=physical_pulse`
    → `Audit.record("physical", "physical")` ⇒ `is_physical() == True`.
  Both `breakdown()` and `is_physical()` go into the JSON, making "robust ≠ in-fabric"
  machine-checkable rather than prose-only.

**Emitted result** — `h5_sweep.py:main()` writes `h5_sweep_parity3.json`
(config · per-rule SNR curves · wall locations · `σ_fg` SNRs · the frozen verdict dict · the
elevated in-fabric verdict · both audits) and prints a summary table (per-rule SNR-vs-σ, both
verdicts, both `is_physical` flags). CLI: `--quick`, `--sigmas`, `--seeds`, `--epochs`, `--nproc`,
`--out`, and `--beta` (single coupled nudge, default `0.1`). No `--betas` grid — the Phase-0 gate
sweeps only `train_sigma`; `02`'s β-scaling (Effect A) is not part of this read-noise verdict.

## 4. Tests (`tests/test_h5_sweep.py`, TDD)

Written before the runner. Fast and deterministic:

- **Pure helpers** — the SNR aggregation (sep/floor mean over seeds) and the foreclosure-wall
  finder, on small synthetic inputs (e.g. a hand-built SNR-vs-σ list → known wall).
- **Audit mapping** — half ⇒ `is_physical() is False`; full ⇒ `is_physical() is True`; both axes
  recorded.
- **Verdict wiring** — the `σ_fg` row is the one fed to `classify_h5`, and the elevated in-fabric
  verdict block is present and keyed on `snr_full`.
- **End-to-end smoke** — a tiny `--quick` config (few σ, 1–2 seeds, ~20 epochs) runs the real
  vendored sim, asserts the JSON carries both verdicts + both audits and that a re-run is
  bit-identical. Full-grid runs stay out of the unit suite.

Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` (a broken global pytest plugin in
this environment crashes collection otherwise; the project needs only numpy/scipy/pytest).

## 5. What this runner does and does not establish

- It answers the **Phase-0 read-noise question only**: does the buildable (half-co-located) object
  hold SNR ≥ 3 at the provisional `σ_fg`? An easy PASS is expected (floating-gate read noise is low)
  and means *"read noise is not the wall,"* **not** *"the device is fine"* (parent spec §3).
- The elevated in-fabric verdict + the two audits make explicit that a Phase-0 PASS does **not**
  certify in-fabric learning — that needs a both-physical rule, which is the read-noise-fragile one.
  Closing that gap (a single-read *and* physics-local rule, or a write mechanism that moves the
  gradient into the fabric) is a **Phase-1/2** problem, out of scope here.
- `σ_fg` is provisional; the verdict carries `provisional=True` until a real device confirms it
  (Phase 1). The device-independent finding — wall location + the half-vs-full mechanism — is
  robust regardless.

## 6. Components summary

| Component | Status | Note |
|---|---|---|
| `vendor_02/*.py` (9 files) | to add | byte-identical port from `02 @95868e7` + provenance headers |
| `vendor_02/README.md` | to add | provenance + re-vendor instruction |
| `audit.py` | to add | ported from `commons/audit.py @e572b0f` |
| `conftest.py` | edit (1 line) | add `vendor_02/` to `sys.path` |
| `h5_sweep.py` | to add | the SNR-curve driver + dual verdict + audits + JSON/CLI |
| `tests/test_h5_sweep.py` | to add | pure-helper + audit + wiring + smoke tests |
| `verdict.py`, `tests/test_verdict.py` | unchanged | frozen Phase-0 verdict |

## 7. Risks / open questions

- **Vendoring drift.** Nine vendored files can silently diverge from `02`. Mitigation: byte-identical
  copies + provenance headers + a `vendor_02/README.md` re-vendor note; never edit them in place.
- **The wall is probably write-side.** Read noise is the cheap failure mode; the floating-gate-specific
  risks (programming asymmetry / granularity / endurance / retention) are Phase-1. This runner must
  not be read as device viability (§5).
- **Full run is gated.** Building the runner is this work; the full Phase-0 sweep + Observed/
  Interpretation findings await explicit go-ahead.
