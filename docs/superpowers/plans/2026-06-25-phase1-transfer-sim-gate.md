# Phase-1 Transfer Sim Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase-1 transfer sim gate — freeze `classify_transfer`, commit a floating-gate (FG) operating point, and a runner (`transfer_sweep.py`) that re-sims `02`'s learning rules at FG-realistic device parameters and classifies whether a custom-silicon tapeout is earned.

**Architecture:** Reuse the vendored `02` sim (`vendor_02/`, already on `sys.path`) — it already models the write side via `DeviceConfig`. Add a frozen pre-registered verdict (`classify_transfer`), a committed FG operating point (`fg_params.py`), a tiny read-averaging model (`fg_rules.py`), and a runner that drives the vendored `train_*_noisy` at an FG `DeviceConfig`, computes three SNRs (buildable half rule = the gate; in-fabric two-read rule; multi-read-averaged robust candidate), checks a per-cell endurance budget, and emits the verdict + JSON.

**Tech Stack:** Python 3.13, numpy, scipy, pytest. Flat modules; functions return numbers; `main()` prints a table + dumps JSON.

## Global Constraints

- **`verdict.py` thresholds are frozen and shared:** `PASS_SNR = 3.0`, `FORECLOSE_SNR = 1.0`. `classify_transfer` reuses them. Do NOT change `classify_h5` or its tests; do NOT retune thresholds.
- **Never edit `vendor_02/*` or `audit.py`** — they are frozen ports. New rules compose vendored primitives; they do not modify them.
- **Pre-registration:** `classify_transfer` (the verdict) and `fg_params.py` (the operating point) are committed BEFORE any transfer-sweep run. `params_confirmed=False` ⇒ `provisional=True` (literature/modeling values, not measured).
- **Outcomes recorded, not asserted.** Tests assert structure, determinism, verdict/audit wiring — never specific science numbers (no "snr_half must be ≥ 3").
- **Run tests with** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` (a broken global pytest plugin in this environment crashes collection otherwise).
- **Commit directly to `main`** (portfolio convention) after each task.
- **The full transfer-sweep run + findings are out of scope** — this plan builds and smoke-tests the runner only.

---

### Task 1: Freeze `classify_transfer`

**Files:**
- Modify: `verdict.py` (replace the `classify_transfer` stub)
- Test: `tests/test_verdict.py` (append tests; do not touch existing tests)

**Interfaces:**
- Consumes: `PASS_SNR`, `FORECLOSE_SNR` (already in `verdict.py`).
- Produces: `classify_transfer(snr_half, snr_full, snr_robust, writes_per_cell, endurance_budget, robust_reads_K, params_confirmed=False) -> dict` with keys `status` (PASS/BOUNDARY/FORECLOSE), `snr_half`, `snr_full`, `snr_robust`, `full_status`, `robust_status`, `in_fabric_achieved` (bool), `endurance_ok` (bool), `writes_per_cell` (int), `endurance_budget` (int), `robust_reads_K` (int), `params_confirmed` (bool), `provisional` (bool). Gate: FORECLOSE if endurance blown OR `snr_half < FORECLOSE_SNR`; PASS if endurance ok AND `snr_half >= PASS_SNR`; else BOUNDARY.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_verdict.py`:

```python
from verdict import classify_transfer


def test_transfer_pass_when_half_transfers_and_endurance_ok():
    res = classify_transfer(snr_half=8.0, snr_full=0.0, snr_robust=0.0,
                            writes_per_cell=300, endurance_budget=100_000, robust_reads_K=10_000)
    assert res["status"] == "PASS"
    assert res["endurance_ok"] is True
    assert res["provisional"] is True


def test_transfer_forecloses_when_half_rule_dies():
    res = classify_transfer(snr_half=0.4, snr_full=0.0, snr_robust=0.0,
                            writes_per_cell=300, endurance_budget=100_000, robust_reads_K=10_000)
    assert res["status"] == "FORECLOSE"


def test_transfer_forecloses_when_endurance_blown_even_if_half_passes():
    res = classify_transfer(snr_half=8.0, snr_full=0.0, snr_robust=0.0,
                            writes_per_cell=200_000, endurance_budget=100_000, robust_reads_K=10_000)
    assert res["status"] == "FORECLOSE"
    assert res["endurance_ok"] is False


def test_transfer_boundary_in_between():
    res = classify_transfer(snr_half=2.0, snr_full=0.0, snr_robust=0.0,
                            writes_per_cell=300, endurance_budget=100_000, robust_reads_K=1)
    assert res["status"] == "BOUNDARY"


def test_transfer_records_elevated_in_fabric_verdicts():
    res = classify_transfer(snr_half=8.0, snr_full=0.0, snr_robust=4.0,
                            writes_per_cell=300, endurance_budget=100_000, robust_reads_K=10_000)
    assert res["full_status"] == "FORECLOSE"      # two-read rule dead on read noise
    assert res["robust_status"] == "PASS"         # multi-read averaging clears the bar
    assert res["in_fabric_achieved"] is True
    assert res["robust_reads_K"] == 10_000


def test_transfer_in_fabric_not_achieved_when_robust_subthreshold():
    res = classify_transfer(snr_half=8.0, snr_full=0.0, snr_robust=0.5,
                            writes_per_cell=300, endurance_budget=100_000, robust_reads_K=10_000)
    assert res["in_fabric_achieved"] is False


def test_transfer_flips_to_final_when_params_confirmed():
    res = classify_transfer(snr_half=8.0, snr_full=0.0, snr_robust=0.0,
                            writes_per_cell=300, endurance_budget=100_000, robust_reads_K=1,
                            params_confirmed=True)
    assert res["provisional"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_verdict.py -q`
Expected: FAIL — `classify_transfer` raises `NotImplementedError` (the current stub).

- [ ] **Step 3: Replace the `classify_transfer` stub in `verdict.py`**

Find the existing stub:

```python
def classify_transfer(*args, **kwargs):
    """Phase-1 transfer verdict — NOT YET FROZEN. Freeze before the Phase-1 run, once the
    floating-gate device-characterization protocol (read noise — confirming sigma_fg — plus
    the write side: programming precision/granularity, tunneling/injection asymmetry,
    endurance, retention) is finalized."""
    raise NotImplementedError("classify_transfer is frozen before Phase 1, not before Phase 0.")
```

Replace it with:

```python
def classify_transfer(snr_half, snr_full, snr_robust, writes_per_cell, endurance_budget,
                      robust_reads_K, params_confirmed=False):
    """Phase-1 transfer gate for 02's rules on an FG array at FG-realistic params. FROZEN 2026-06-25.

    Inputs (reported by transfer_sweep, which re-sims 02's vendored rules at an FG DeviceConfig):
      snr_half          SNR of the BUILDABLE half-co-located rule (one read, off-array gradient) at the
                        FG operating point. The gate keys on this.
      snr_full          SNR of the genuinely in-fabric two-read rule. Expected FORECLOSE (read noise;
                        Phase 0). Recorded co-equally, not gating.
      snr_robust        SNR of the multi-read-averaged in-fabric candidate at robust_reads_K reads.
      writes_per_cell   worst-case program/erase pulses any single cell sees over a training run.
      endurance_budget  the FG program/erase endurance budget (committed in fg_params).
      robust_reads_K    the read-averaging factor K at which snr_robust was achieved.
      params_confirmed  True once measured against a real FG device; else the verdict is provisional.

    Verdict (gate on the buildable object, subject to a hard endurance constraint):
      FORECLOSE  endurance blown OR snr_half < FORECLOSE_SNR  -> don't tape out (write side / endurance
                                                                 kills even the buildable learner).
      PASS       endurance ok AND snr_half >= PASS_SNR        -> the write side transfers; a tapeout is
                                                                 earned (>= inference-grade array).
      BOUNDARY   otherwise.
    Elevated (the in-fabric question, recorded co-equally): full_status, robust_status, in_fabric_achieved.
    """
    def _band(x):
        if x >= PASS_SNR:
            return "PASS"
        if x < FORECLOSE_SNR:
            return "FORECLOSE"
        return "BOUNDARY"

    endurance_ok = writes_per_cell <= endurance_budget
    if (not endurance_ok) or snr_half < FORECLOSE_SNR:
        status = "FORECLOSE"
    elif snr_half >= PASS_SNR:
        status = "PASS"
    else:
        status = "BOUNDARY"
    robust_status = _band(snr_robust)
    return {
        "status": status,
        "snr_half": float(snr_half),
        "snr_full": float(snr_full),
        "snr_robust": float(snr_robust),
        # elevated, co-equal in-fabric records (Phase-0 pattern):
        "full_status": _band(snr_full),            # expected FORECLOSE (two-read read-noise fragility)
        "robust_status": robust_status,            # does multi-read averaging buy in-fabric learning?
        "in_fabric_achieved": robust_status == "PASS",
        "endurance_ok": bool(endurance_ok),
        "writes_per_cell": int(writes_per_cell),
        "endurance_budget": int(endurance_budget),
        "robust_reads_K": int(robust_reads_K),
        "params_confirmed": bool(params_confirmed),
        "provisional": not bool(params_confirmed),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_verdict.py -q`
Expected: PASS (the 8 frozen `classify_h5` tests + the 7 new `classify_transfer` tests).

- [ ] **Step 5: Commit**

```bash
git add verdict.py tests/test_verdict.py
git commit -m "feat(08): freeze classify_transfer (Phase-1 transfer gate)"
```

---

### Task 2: Commit the FG operating point — `fg_params.py`

**Files:**
- Create: `fg_params.py`
- Test: `tests/test_fg_params.py`

**Interfaces:**
- Consumes: `memristor.DeviceConfig` (vendored).
- Produces: `SIGMA_FG` (float, ≤ 0.01); `ENDURANCE_BUDGET` (int, ~1e5); `fg_device_config() -> DeviceConfig` (the committed nominal FG operating point: `nonlin>2`, `asym<0.5`, `n_levels=32`, `drift=0.0`, `sigma_read=SIGMA_FG`); `FG_SWEEPS` (dict knob→list of values for one-at-a-time Observed sweeps, keys are `DeviceConfig` field names); `ROBUST_K_GRID` (ascending list of read-averaging factors, starts at 1).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fg_params.py`:

```python
"""The committed FG operating point: structural sanity (no science numbers asserted)."""
import dataclasses

from memristor import DeviceConfig
from fg_params import SIGMA_FG, ENDURANCE_BUDGET, fg_device_config, FG_SWEEPS, ROBUST_K_GRID


def test_sigma_fg_at_or_below_phase0_provisional():
    assert 0.0 < SIGMA_FG <= 0.01


def test_endurance_budget_is_a_positive_cycle_count():
    assert isinstance(ENDURANCE_BUDGET, int) and ENDURANCE_BUDGET > 0


def test_nominal_cfg_is_fg_directional():
    cfg = fg_device_config()
    assert isinstance(cfg, DeviceConfig)
    assert cfg.sigma_read == SIGMA_FG
    assert cfg.drift == 0.0          # FG retention is excellent
    assert cfg.nonlin > 2.0          # FG update is strongly nonlinear
    assert cfg.asym < 0.5            # tunneling/injection asymmetry


def test_fg_sweeps_keys_are_deviceconfig_fields():
    fields = {f.name for f in dataclasses.fields(DeviceConfig)}
    assert set(FG_SWEEPS).issubset(fields)
    for vals in FG_SWEEPS.values():
        assert len(vals) >= 2


def test_robust_k_grid_ascending_from_one():
    assert ROBUST_K_GRID[0] == 1
    assert ROBUST_K_GRID == sorted(ROBUST_K_GRID)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_fg_params.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fg_params'`.

- [ ] **Step 3: Create `fg_params.py`**

```python
"""Floating-gate operating point for the Phase-1 transfer gate — the PRE-REGISTERED, committed-before-
the-run FG device parameters. These are modeling choices grounded in the FG literature (qualitative
direction + rough magnitudes), exactly as 02's DeviceConfig defaults were modeling choices. The verdict
stays provisional (params_confirmed=False) until a real device — a characterization tapeout or a
vendor-characterized FG process — measures them.

Literature basis:
 - FG synaptic transistor programmed by Fowler-Nordheim tunneling (erase) + hot-electron injection
   (program): the two mechanisms differ in rate/granularity -> strong potentiation/depression ASYMMETRY,
   and the update is strongly state-dependent -> NONLINEAR. (Diorio, Hasler, Minch, Mead, "A single-
   transistor silicon synapse," IEEE Trans. Electron Devices 43(11), 1996.)
 - FG analog computing / RASP-FPAA: an FG stores a non-volatile analog weight; closed-loop programming
   reaches high precision, open-loop pulse programming is coarser (~5-8 bit). (Hasler et al., floating-
   gate analog circuits; Georgia Tech RASP / SoC-FPAA line.)
 - FG READ is non-destructive and quiet -> low sigma_read (at/below Phase-0's provisional 0.01).
 - FG RETENTION is excellent (years) -> drift ~ 0 (the FG-favorable knob).
 - FG ENDURANCE (program/erase cycles through the tunneling oxide) is finite, ~1e5-1e6 (flash/EEPROM
   reliability range); we take the conservative low end, 1e5, as the budget.

The exact swept values below are committed here BEFORE the run (the pre-registered operating point);
they are modeling choices, not measured device specs.
"""
from memristor import DeviceConfig   # vendored 02 device model

SIGMA_FG = 0.01            # FG read noise (fraction of conductance span); at/below Phase-0's value
ENDURANCE_BUDGET = 100_000  # ~1e5 program/erase cycles (conservative low end of the FG/flash range)


def fg_device_config():
    """The committed nominal FG operating point: 02's DeviceConfig re-valued for a floating gate —
    strong update nonlinearity + tunneling/injection asymmetry, ~5-bit programming granularity,
    excellent retention (drift=0), quiet read. g_min/g_max/dG0 keep 02's defaults."""
    return DeviceConfig(
        nonlin=3.0,        # > 02's 2.0: FG injection/tunneling is strongly state-dependent
        asym=0.3,          # < 02's 0.5: tunneling-erase vs injection-program are more asymmetric
        n_levels=32,       # ~5-bit programming granularity (02 showed ~4-bit suffices via nu->0)
        sigma_write=0.03,  # cycle-to-cycle injection variability (~02's 0.02, slightly higher)
        sigma_read=SIGMA_FG,
        drift=0.0,         # FG charge retention is years -> no decay over a training run
    )


# One-at-a-time Observed robustness sweeps around the nominal (keys are DeviceConfig field names).
FG_SWEEPS = {
    "nonlin":      [2.0, 3.0, 4.0],
    "asym":        [0.2, 0.3, 0.5],
    "n_levels":    [16, 32, 64],
    "sigma_write": [0.02, 0.03, 0.05],
    "sigma_read":  [0.003, 0.01, 0.02],
}

# Read-averaging factors K for the robust-in-fabric search (objective 2). Ascending, starts at 1.
ROBUST_K_GRID = [1, 4, 16, 100, 1_000, 10_000]
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_fg_params.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add fg_params.py tests/test_fg_params.py
git commit -m "feat(08): commit FG operating point (fg_params, pre-registered)"
```

---

### Task 3: Read-averaging model + endurance count — `fg_rules.py`

**Files:**
- Create: `fg_rules.py`
- Test: `tests/test_fg_rules.py`

**Interfaces:**
- Consumes: `readnoise.noisy_read` (vendored), `memristor.DeviceConfig` (vendored).
- Produces: `effective_sigma(sigma, K) -> float` (= `sigma / sqrt(K)`); `averaged_noisy_read(g, sigma, cfg, rng, K) -> np.ndarray` (mean of `K` vendored `noisy_read` draws); `writes_per_cell(n_epochs) -> int` (= `n_epochs`, the conservative worst-case bound).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fg_rules.py`:

```python
"""Read-averaging model: the K-averaged read has std ~ sigma/sqrt(K) (validates the effective-sigma
equivalence the transfer sweep relies on)."""
import numpy as np

from memristor import DeviceConfig
from readnoise import noisy_read
from fg_rules import effective_sigma, averaged_noisy_read, writes_per_cell


def test_effective_sigma_scales_as_one_over_sqrt_k():
    assert effective_sigma(0.01, 1) == 0.01
    assert abs(effective_sigma(0.01, 100) - 0.001) < 1e-12


def test_averaged_read_std_matches_effective_sigma():
    # Many independent averaged reads of a mid-range conductance (clipping negligible there):
    # the empirical std should track sigma/sqrt(K).
    cfg = DeviceConfig()
    rng = np.random.default_rng(0)
    g = np.full(2000, 0.5 * (cfg.g_min + cfg.g_max))   # mid-range, away from bounds
    sigma, K = 0.05, 16
    reads = np.array([averaged_noisy_read(g, sigma, cfg, rng, K)[0] for _ in range(4000)])
    span = cfg.g_max - cfg.g_min
    expected = effective_sigma(sigma, K) * span
    assert abs(reads.std() - expected) / expected < 0.1   # within 10%


def test_averaged_read_k1_is_a_single_read():
    cfg = DeviceConfig()
    g = np.full(5, 0.5 * (cfg.g_min + cfg.g_max))
    a = averaged_noisy_read(g, 0.02, cfg, np.random.default_rng(1), 1)
    b = np.asarray(noisy_read(g, 0.02, cfg, np.random.default_rng(1)))
    assert np.allclose(a, b)


def test_writes_per_cell_is_n_epochs():
    assert writes_per_cell(300) == 300
    assert isinstance(writes_per_cell(300), int)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_fg_rules.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fg_rules'`.

- [ ] **Step 3: Create `fg_rules.py`**

```python
"""Read-averaging model for the robust-in-fabric candidate (objective 2), plus the endurance write-
count. 08-authored code composing the vendored 02 primitives; it NEVER edits vendor_02/.

Multi-read averaging: averaging K independent reads of a cell reduces the read-noise std by sqrt(K)
(the mean of K i.i.d. draws). So a K-averaged two-read contrastive update at read noise sigma is
distributionally the vendored two-read rule at an EFFECTIVE read noise sigma/sqrt(K) (exact up to the
clip at the conductance bounds, negligible near mid-range -- validated in tests). transfer_sweep uses
that equivalence to compute the robust SNR by running the vendored full rule at effective_sigma; this
module supplies effective_sigma + an averaged_noisy_read that validates the equivalence."""
import numpy as np

from readnoise import noisy_read   # vendored 02 (clipped, span-scaled, explicit-sigma read)


def effective_sigma(sigma, K):
    """K-read averaging reduces read-noise std by sqrt(K): the averaged read noise is sigma/sqrt(K)."""
    return float(sigma) / float(np.sqrt(K))


def averaged_noisy_read(g, sigma, cfg, rng, K):
    """Mean of K independent vendored noisy reads of g. The faithful multi-read read; its empirical
    std ~ sigma/sqrt(K) validates the effective_sigma equivalence the sweep relies on."""
    g = np.asarray(g, float)
    K = int(K)
    if K <= 1:
        return noisy_read(g, sigma, cfg, rng)
    acc = np.zeros_like(g)
    for _ in range(K):
        acc = acc + np.asarray(noisy_read(g, sigma, cfg, rng), float)
    return acc / float(K)


def writes_per_cell(n_epochs):
    """Worst-case program/erase pulses any single cell receives over a training run. The sign rules
    fire a pulse on (almost) every step where the per-edge gradient sign is nonzero, so the conservative
    upper bound is one write per cell per epoch = n_epochs. Conservative is correct for an endurance
    budget check (over-counting only tightens the margin)."""
    return int(n_epochs)
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_fg_rules.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add fg_rules.py tests/test_fg_rules.py
git commit -m "feat(08): fg_rules — read-averaging model + endurance write-count"
```

---

### Task 4: SNR at an FG config + the robust K-scan — `transfer_sweep.py` core

**Files:**
- Create: `transfer_sweep.py`
- Test: `tests/test_transfer_sweep.py`

**Interfaces:**
- Consumes: vendored `build_full_diff`, `make_read_nonlinearity`, `make_cocontent`, `parity3_data`, `evaluate_separation`, `train_manhattan_noisy`, `train_coupled_noisy`, `task.verdict`; `fg_rules.effective_sigma`; `fg_params.ROBUST_K_GRID`; `verdict.PASS_SNR`.
- Produces: `_snr_at_cfg(rule, cfg, train_sigma, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3, n_epochs=300, seeds=range(10)) -> float` (mean SNR over seeds of `rule ∈ {"half","full"}` trained at `cfg`+`train_sigma`, clean-eval); `min_reads_to_pass(cfg, sigma_fg, *, K_grid=None, ...) -> {"snr_robust": float, "robust_reads_K": int}` (smallest `K` in the grid where the full rule at `effective_sigma(sigma_fg,K)` reaches `PASS_SNR`, else the best achieved + its K).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transfer_sweep.py`:

```python
"""Phase-1 transfer runner: SNR-at-config, robust K-scan, assembly, CLI. Structure/wiring/determinism
only (no science numbers)."""
from fg_params import fg_device_config, SIGMA_FG
from transfer_sweep import _snr_at_cfg, min_reads_to_pass


def test_snr_at_cfg_returns_nonneg_and_is_deterministic():
    cfg = fg_device_config()
    a = _snr_at_cfg("half", cfg, SIGMA_FG, n_epochs=10, seeds=range(2))
    b = _snr_at_cfg("half", cfg, SIGMA_FG, n_epochs=10, seeds=range(2))
    assert a >= 0.0
    assert a == b


def test_min_reads_to_pass_reports_snr_and_K_from_grid():
    cfg = fg_device_config()
    res = min_reads_to_pass(cfg, SIGMA_FG, K_grid=[1, 4], n_epochs=10, seeds=range(2))
    assert set(res) == {"snr_robust", "robust_reads_K"}
    assert res["robust_reads_K"] in (1, 4)
    assert res["snr_robust"] >= 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_transfer_sweep.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'transfer_sweep'`.

- [ ] **Step 3: Create `transfer_sweep.py` with imports, `_build`, `_snr_at_cfg`, `min_reads_to_pass`**

```python
"""Phase-1 transfer sim gate runner. Re-sims 02's vendored learning rules at an FG-realistic
DeviceConfig and classifies whether a custom-silicon tapeout is earned (the frozen classify_transfer).
A thin driver: no physics is reimplemented -- every model call lives in vendor_02/. Outcomes recorded,
not asserted. See docs/superpowers/specs/2026-06-25-phase1-transfer-sim-gate-design.md."""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

# Make the vendored 02 sim importable when run as a script (conftest only covers pytest).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor_02"))

from crossbar import build_full_diff, make_read_nonlinearity, make_cocontent
from manhattan import evaluate_separation
from readnoise import train_manhattan_noisy, train_coupled_noisy
from task import parity3_data, verdict as task_verdict
from verdict import classify_transfer, PASS_SNR
from fg_params import (fg_device_config, SIGMA_FG, ENDURANCE_BUDGET, FG_SWEEPS, ROBUST_K_GRID)
from fg_rules import effective_sigma, writes_per_cell
from h5_sweep import rule_audit   # reuse the two-axis audit mapping (half=cpu,phys; full=phys,phys)


def _build(H=8, a=0.4):
    """Rebuild the parity-3 task + differential crossbar topology + read nonlinearity + co-content
    (orchestration only; all builders are vendored from 02)."""
    topo = build_full_diff(H)
    X, y = parity3_data()
    phi, dphi = make_read_nonlinearity(a)
    s_fn = make_cocontent(a)
    return topo, X, y, phi, dphi, s_fn


def _snr_at_cfg(rule, cfg, train_sigma, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
                n_epochs=300, seeds=range(10)):
    """Mean SNR (separation/floor, 02's task.verdict) over seeds for `rule` trained at DeviceConfig
    `cfg` and training read noise `train_sigma`; clean-convention eval. Fresh fixed-seed eval RNG ->
    deterministic."""
    topo, X, y, phi, dphi, s_fn = _build(H, a)
    eval_rng = np.random.default_rng(12345)
    snrs = []
    for s in seeds:
        if rule == "half":
            res = train_manhattan_noisy(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=n_epochs,
                                        seed=int(s), target=target, train_sigma=train_sigma)
        elif rule == "full":
            res = train_coupled_noisy(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=n_epochs,
                                      seed=int(s), target=target, beta=beta, s_fn=s_fn,
                                      train_sigma=train_sigma, mode="sign")
        else:
            raise ValueError(f"unknown rule {rule!r}")
        sep, floor = evaluate_separation(topo, res["g"], X, y, phi, dphi, V_drive, cfg, eval_rng)
        snrs.append(float(task_verdict(sep, floor)["snr"]))
    return float(np.mean(snrs))


def min_reads_to_pass(cfg, sigma_fg, *, K_grid=None, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
                      n_epochs=300, seeds=range(10)):
    """Robust-in-fabric search: the smallest read-averaging K (ascending grid) at which the full
    (two-read) rule reaches PASS_SNR at effective read noise sigma_fg/sqrt(K). If none clears the bar,
    return the best SNR seen and the K that produced it. Models multi-read averaging via the
    effective_sigma equivalence (fg_rules)."""
    K_grid = ROBUST_K_GRID if K_grid is None else K_grid
    best_snr, best_K = -1.0, sorted(K_grid)[-1]
    for K in sorted(K_grid):
        snr = _snr_at_cfg("full", cfg, effective_sigma(sigma_fg, K), H=H, a=a, V_drive=V_drive,
                          beta=beta, target=target, n_epochs=n_epochs, seeds=seeds)
        if snr > best_snr:
            best_snr, best_K = snr, K
        if snr >= PASS_SNR:
            return {"snr_robust": float(snr), "robust_reads_K": int(K)}
    return {"snr_robust": float(best_snr), "robust_reads_K": int(best_K)}
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_transfer_sweep.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add transfer_sweep.py tests/test_transfer_sweep.py
git commit -m "feat(08): transfer_sweep core — SNR-at-FG-config + robust K-scan"
```

---

### Task 5: Assemble the transfer verdict — `run_transfer`

**Files:**
- Modify: `transfer_sweep.py` (append `_knob_sweeps`, `run_transfer`)
- Test: `tests/test_transfer_sweep.py` (append)

**Interfaces:**
- Consumes: `_snr_at_cfg`, `min_reads_to_pass`, `fg_device_config`, `SIGMA_FG`, `ENDURANCE_BUDGET`, `FG_SWEEPS`, `writes_per_cell`, `classify_transfer`, `rule_audit`, `dataclasses.replace`.
- Produces: `run_transfer(*, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3, n_epochs=300, seeds=range(10), sigma_fg=SIGMA_FG, endurance_budget=ENDURANCE_BUDGET, params_confirmed=False, K_grid=None) -> dict` with keys `config`, `nominal_cfg` (dict), `snr_half`, `snr_full`, `snr_robust`, `robust_reads_K`, `writes_per_cell`, `verdict` (frozen `classify_transfer` dict), `sweeps` (`{knob: [{"value", "snr_half"}]}`), `audit` (`{"half"/"full": {"breakdown","is_physical"}}`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transfer_sweep.py`:

```python
import json
from transfer_sweep import run_transfer
from fg_params import SIGMA_FG


def test_run_transfer_assembles_gate_elevated_and_audits():
    res = run_transfer(n_epochs=10, seeds=range(2), K_grid=[1, 4])
    assert res["verdict"]["status"] in {"PASS", "BOUNDARY", "FORECLOSE"}
    assert res["verdict"]["snr_half"] == res["snr_half"]
    assert res["verdict"]["provisional"] is True
    # elevated in-fabric records present
    assert "full_status" in res["verdict"] and "robust_status" in res["verdict"]
    assert res["robust_reads_K"] in (1, 4)
    # two-axis audit: half not both-physical, full both-physical
    assert res["audit"]["half"]["is_physical"] is False
    assert res["audit"]["full"]["is_physical"] is True
    # Observed knob sweeps present and JSON-safe
    assert set(res["sweeps"]) and json.dumps(res)


def test_run_transfer_endurance_recorded():
    res = run_transfer(n_epochs=10, seeds=range(1), K_grid=[1])
    assert res["verdict"]["endurance_ok"] is True       # 10 writes << 1e5 budget
    assert res["verdict"]["writes_per_cell"] == 10
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_transfer_sweep.py -q`
Expected: FAIL — `ImportError: cannot import name 'run_transfer'`.

- [ ] **Step 3: Append `_knob_sweeps` and `run_transfer` to `transfer_sweep.py`**

```python
def _knob_sweeps(nominal_cfg, sigma_fg, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
                 n_epochs=300, seeds=range(10)):
    """Observed robustness: vary each FG knob one-at-a-time around the nominal cfg and record the
    half-rule SNR at each value (the gate object's sensitivity to each write-side non-ideality)."""
    out = {}
    for knob, vals in FG_SWEEPS.items():
        rows = []
        for v in vals:
            cfg = dataclasses.replace(nominal_cfg, **{knob: v})
            rows.append({"value": v,
                         "snr_half": _snr_at_cfg("half", cfg, sigma_fg, H=H, a=a, V_drive=V_drive,
                                                 beta=beta, target=target, n_epochs=n_epochs,
                                                 seeds=seeds)})
        out[knob] = rows
    return out


def run_transfer(*, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3, n_epochs=300, seeds=range(10),
                 sigma_fg=SIGMA_FG, endurance_budget=ENDURANCE_BUDGET, params_confirmed=False,
                 K_grid=None):
    """Drive the gate (half rule at the nominal FG cfg), the elevated in-fabric records (the two-read
    rule, and the multi-read-averaged robust candidate), the endurance check, and the Observed knob
    sweeps; classify with the frozen classify_transfer; record the two-axis audit for both rules."""
    seeds = list(seeds)
    cfg = fg_device_config()
    kw = dict(H=H, a=a, V_drive=V_drive, beta=beta, target=target, n_epochs=n_epochs, seeds=seeds)

    snr_half = _snr_at_cfg("half", cfg, sigma_fg, **kw)            # GATE (buildable object)
    snr_full = _snr_at_cfg("full", cfg, sigma_fg, **kw)            # in-fabric two-read at FG op point
    robust = min_reads_to_pass(cfg, sigma_fg, K_grid=K_grid, **kw)  # multi-read-averaged candidate
    wpc = writes_per_cell(n_epochs)

    verdict = classify_transfer(snr_half, snr_full, robust["snr_robust"], wpc, endurance_budget,
                                robust["robust_reads_K"], params_confirmed=params_confirmed)

    sweeps = _knob_sweeps(cfg, sigma_fg, **kw)

    audit = {}
    for rule in ("half", "full"):
        a_rec = rule_audit(rule)
        audit[rule] = {"breakdown": {f"{g}|{w}": c for (g, w), c in a_rec.breakdown().items()},
                       "is_physical": a_rec.is_physical()}

    return {
        "config": {"H": H, "a": a, "V_drive": V_drive, "beta": beta, "target": target,
                   "n_epochs": n_epochs, "n_seeds": len(seeds), "sigma_fg": float(sigma_fg),
                   "endurance_budget": int(endurance_budget)},
        "nominal_cfg": dataclasses.asdict(cfg),
        "snr_half": float(snr_half), "snr_full": float(snr_full),
        "snr_robust": float(robust["snr_robust"]), "robust_reads_K": int(robust["robust_reads_K"]),
        "writes_per_cell": int(wpc),
        "verdict": verdict, "sweeps": sweeps, "audit": audit,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_transfer_sweep.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add transfer_sweep.py tests/test_transfer_sweep.py
git commit -m "feat(08): transfer_sweep run_transfer — gate + elevated + audits + sweeps"
```

---

### Task 6: CLI, summary, and JSON dump — `main`

**Files:**
- Modify: `transfer_sweep.py` (append `_QUICK`, `_print_summary`, `main`, `__main__` guard)
- Test: `tests/test_transfer_sweep.py` (append)

**Interfaces:**
- Consumes: `run_transfer`.
- Produces: `main(argv=None) -> dict` (parses args, runs, prints, writes JSON to `--out`, returns the result). Flags: `--out` (default `transfer_parity3.json`), `--epochs` (300), `--seeds` (10), `--beta` (0.1), `--quick` (tiny smoke: 2 seeds, 20 epochs, `K_grid=[1, 4]`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transfer_sweep.py`:

```python
from transfer_sweep import main


def test_main_quick_writes_json_with_verdict(tmp_path):
    out = tmp_path / "transfer.json"
    res = main(["--quick", "--out", str(out)])
    assert out.exists()
    data = json.loads(out.read_text())
    assert set(data) >= {"config", "nominal_cfg", "verdict", "sweeps", "audit",
                         "snr_half", "snr_full", "snr_robust", "robust_reads_K", "writes_per_cell"}
    assert data["verdict"]["provisional"] is True
    assert res["config"]["n_seeds"] == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_transfer_sweep.py -q`
Expected: FAIL — `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Append `_QUICK`, `_print_summary`, `main`, and the `__main__` guard to `transfer_sweep.py`**

```python
_QUICK = {"seeds": range(2), "n_epochs": 20, "K_grid": [1, 4]}


def _print_summary(res):
    c = res["config"]
    v = res["verdict"]
    print("=" * 78)
    print("PHASE-1 transfer sim gate — 02's rules on an FG array at FG-realistic params (parity-3)")
    print("=" * 78)
    print(f"sigma_fg={c['sigma_fg']}  n_seeds={c['n_seeds']}  n_epochs={c['n_epochs']}  "
          f"endurance_budget={c['endurance_budget']}")
    print(f"\nGATE (frozen classify_transfer, keys on the buildable HALF rule): {v['status']}")
    print(f"  snr_half={v['snr_half']:.3f}  writes_per_cell={v['writes_per_cell']}  "
          f"endurance_ok={v['endurance_ok']}  provisional={v['provisional']}")
    print(f"IN-FABRIC (elevated): two-read full={v['full_status']} (snr={v['snr_full']:.3f}); "
          f"multi-read robust={v['robust_status']} (snr={v['snr_robust']:.3f} at K={v['robust_reads_K']}); "
          f"in_fabric_achieved={v['in_fabric_achieved']}")
    print(f"audit: half is_physical={res['audit']['half']['is_physical']}  "
          f"full is_physical={res['audit']['full']['is_physical']}")
    print("\n-- Observed: half-rule SNR vs each FG knob (one-at-a-time) --")
    for knob, rows in res["sweeps"].items():
        cells = "  ".join(f"{r['value']}:{r['snr_half']:.2f}" for r in rows)
        print(f"  {knob:<12} {cells}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Phase-1 transfer sim gate for 02's rules on a floating-gate array.")
    ap.add_argument("--out", default="transfer_parity3.json")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--quick", action="store_true", help="tiny local smoke (few seeds/epochs/K)")
    args = ap.parse_args(argv)

    if args.quick:
        res = run_transfer(seeds=_QUICK["seeds"], n_epochs=_QUICK["n_epochs"],
                           beta=args.beta, K_grid=_QUICK["K_grid"])
    else:
        res = run_transfer(seeds=range(args.seeds), n_epochs=args.epochs, beta=args.beta)

    _print_summary(res)
    with open(args.out, "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"\nraw numbers -> {args.out}")
    return res


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the CLI test passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_transfer_sweep.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Run the full suite green**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`
Expected: PASS (Phase-0 suite + `classify_transfer` tests + `fg_params` + `fg_rules` + `transfer_sweep`).

- [ ] **Step 6: Validate the runner end-to-end with a real quick smoke run**

Run (exercises the `__main__`/argparse/script path the in-process test does not):
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python transfer_sweep.py --quick --out transfer_quick_smoke.json && rm transfer_quick_smoke.json
```
Expected: prints the gate + elevated in-fabric verdicts + the Observed knob sweeps + both `is_physical` flags, and writes/removes the disposable JSON. This is the validation smoke; it is **not** the gated full Phase-1 run.

- [ ] **Step 7: Commit**

```bash
git add transfer_sweep.py tests/test_transfer_sweep.py
git commit -m "feat(08): transfer_sweep CLI + summary + JSON; quick-smoke validated"
```

---

## After the plan

The runner is built and smoke-validated. **The full Phase-1 transfer-sweep run + Observed/Interpretation findings (`docs/superpowers/findings/`) are a separate, gated step** — do not run the full grid or write findings without the user's explicit go-ahead (spec §8).
