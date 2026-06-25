# Phase-0 H5 Sweep Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `h5_sweep.py`, the Phase-0 driver that produces `snr_half` / `snr_full` for the frozen `classify_h5`, plus the Observed read-noise curves, by driving a vendored copy of `02`'s sim.

**Architecture:** Port (vendor) the 9-file transitive closure of `02`'s noisy-training sweep into `vendor_02/` and the commons audit hook into `audit.py`; then write a thin serial driver that, per `(rule, train_sigma)`, calls the vendored `train_*_noisy` → `evaluate_separation` → `task.verdict` to get a mean SNR, assembles the frozen primary verdict (on the half rule) plus an elevated co-equal in-fabric verdict (on the full rule) plus a two-axis audit for both rules, and dumps JSON.

**Tech Stack:** Python 3.13, numpy, scipy, pytest. Flat modules; functions return numbers; `main()` prints a table + dumps JSON.

## Global Constraints

- **Port, not import.** Vendored files are copied **from `02 @95868e7`** (sim) and **`commons/audit.py @e572b0f`** (audit), with a one-line provenance header prepended and **no code edits**. Never edit them to diverge; re-vendor on upstream change.
- **`verdict.py` is frozen.** Do not modify `verdict.py` or `tests/test_verdict.py`. `classify_h5` keys the primary verdict on `snr_half`.
- **Outcomes recorded, not asserted.** Tests assert structure, determinism, and audit/verdict wiring — never specific science numbers (no "SNR must be ≥ 3").
- **Run tests with** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` (a broken global pytest plugin in this environment crashes collection otherwise).
- **Two-axis audit, both rules.** half = `(cpu, physical)` ⇒ `is_physical()==False`; full = `(physical, physical)` ⇒ `is_physical()==True`.
- **Commit directly to `main`** (portfolio convention) after each task.
- **The full Phase-0 sweep run + findings are out of scope** — this plan builds and smoke-tests the runner only.

---

### Task 1: Port the commons two-axis audit hook

**Files:**
- Create: `audit.py` (copied from `../00-shared-harness/commons/audit.py @e572b0f`)
- Test: `tests/test_audit.py`

**Interfaces:**
- Consumes: nothing (leaf).
- Produces: `Audit` class with `.record(gradient_source, write_mechanism, n_updates=1)`, `.breakdown() -> {(g,w): count}`, `.is_physical() -> bool`, `.total() -> int`; module constants `PHYSICAL = "physical"`, `CPU = "cpu"`, `ELECTRONIC = "electronic"`.

- [ ] **Step 1: Copy the commons audit module verbatim**

Run:
```bash
cp ../00-shared-harness/commons/audit.py audit.py
```

- [ ] **Step 2: Prepend the provenance header**

Insert this as the very first line of `audit.py` (above the existing module docstring), changing no other line:

```python
# PORTED from 00-shared-harness/commons/audit.py @e572b0f — do not edit to diverge; re-vendor on upstream change.
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_audit.py`:

```python
"""Confirm the two-axis audit behaves as 08 relies on: half rule is NOT both-physical,
full rule IS both-physical."""
from audit import Audit, PHYSICAL, CPU


def test_half_rule_audit_is_not_physical():
    a = Audit()
    a.record(CPU, PHYSICAL)            # off-array CPU gradient + physical write
    assert a.is_physical() is False
    assert a.total() == 1


def test_full_rule_audit_is_physical():
    a = Audit()
    a.record(PHYSICAL, PHYSICAL)       # in-fabric gradient + physical write
    assert a.is_physical() is True


def test_breakdown_keys_on_the_pair():
    a = Audit()
    a.record(CPU, PHYSICAL, 3)
    assert a.breakdown() == {(CPU, PHYSICAL): 3}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_audit.py -q`
Expected: PASS (3 passed). The module is a verbatim copy of a tested commons module, so the port is correct iff these pass.

- [ ] **Step 5: Commit**

```bash
git add audit.py tests/test_audit.py
git commit -m "feat(08): port two-axis audit hook from commons @e572b0f"
```

---

### Task 2: Vendor 02's noisy-training sim and wire it onto the path

**Files:**
- Create: `vendor_02/readnoise.py`, `vendor_02/coupled.py`, `vendor_02/manhattan.py`, `vendor_02/crossbar.py`, `vendor_02/solver.py`, `vendor_02/jacobian.py`, `vendor_02/memristor.py`, `vendor_02/task.py`, `vendor_02/ceilings.py` (each copied from `../02-memristive-crossbar/<file> @95868e7`)
- Create: `vendor_02/README.md`
- Modify: `conftest.py` (add `vendor_02/` to `sys.path`)
- Test: `tests/test_vendor_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces (the subset 08 drives): from `readnoise` — `train_manhattan_noisy(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=300, seed=0, target=0.3, train_sigma=0.0) -> {"g","separation","final_separation","curve","audit"}` and `train_coupled_noisy(topo, X, y, phi, dphi, V_drive, cfg, n_epochs=300, seed=0, target=0.3, beta=0.1, s_fn=None, train_sigma=0.0, mode="sign") -> {...}`; from `crossbar` — `build_full_diff(H)`, `make_read_nonlinearity(a) -> (phi, dphi)`, `make_cocontent(a) -> s_fn`; from `manhattan` — `evaluate_separation(topo, g, X, y, phi, dphi, V_drive, cfg, rng, n_reads=64) -> (sep_mean, floor)`; from `task` — `parity3_data() -> (X, y)`, `verdict(sep, floor, k=3.0) -> {"status","snr","separation","floor"}`; from `memristor` — `DeviceConfig`. Manhattan audit dict has `error_site="off_array_cpu"`, `increment_site="physical_pulse"`; coupled has `error_site="on_array_local"`, `increment_site="physical_pulse"`.

- [ ] **Step 1: Copy the 9 vendored files**

Run:
```bash
mkdir -p vendor_02
for f in readnoise coupled manhattan crossbar solver jacobian memristor task ceilings; do
  cp "../02-memristive-crossbar/$f.py" "vendor_02/$f.py"
done
```

- [ ] **Step 2: Prepend the provenance header to each vendored file**

Prepend this single line (with the matching `<file>`) as the first line of each `vendor_02/*.py`, changing no other line:

```python
# PORTED from 02-memristive-crossbar/<file>.py @95868e7 — do not edit to diverge; re-vendor on upstream change.
```

- [ ] **Step 3: Write the provenance README**

Create `vendor_02/README.md`:

```markdown
# vendor_02 — ported sim from 02-memristive-crossbar @95868e7

These are the **transitive closure** of `02`'s noisy-training read-noise sweep
(`readnoise` + `coupled` + `manhattan`), copied **byte-for-byte** (only a one-line
provenance header prepended) so `08` runs the H5 sim hermetically — the portfolio is
**port, not import** (each project is a separate repo).

Files: `readnoise coupled manhattan crossbar solver jacobian memristor task ceilings`.
`ceilings.py` is included **only** so `manhattan.py` stays unmodified; the H5 path never
calls it.

**Do not edit these to diverge.** On an upstream `02` change, re-vendor with a clean copy:
re-run the copy + header step from `docs/superpowers/plans/2026-06-24-phase0-h5-sweep-runner.md`,
bumping the sha.
```

- [ ] **Step 4: Wire `vendor_02/` onto `sys.path`**

Modify `conftest.py` — after the existing `sys.path.insert(0, os.path.dirname(__file__))`, add:

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor_02"))
```

- [ ] **Step 5: Write the failing smoke test**

Create `tests/test_vendor_smoke.py`:

```python
"""Smoke-test the vendored 02 sim: it imports, runs a tiny noisy training for each rule,
and carries the expected (rule -> audit) signatures."""
from crossbar import build_full_diff, make_read_nonlinearity, make_cocontent
from task import parity3_data
from memristor import DeviceConfig
from readnoise import train_manhattan_noisy, train_coupled_noisy


def _setup():
    topo = build_full_diff(8)
    X, y = parity3_data()
    phi, dphi = make_read_nonlinearity(0.4)
    s_fn = make_cocontent(0.4)
    return topo, X, y, phi, dphi, s_fn


def test_vendored_manhattan_runs_and_audits_half():
    topo, X, y, phi, dphi, s_fn = _setup()
    res = train_manhattan_noisy(topo, X, y, phi, dphi, 1.5, DeviceConfig(),
                                n_epochs=8, seed=0, train_sigma=0.01)
    assert set(res) >= {"g", "separation", "audit"}
    assert res["audit"]["error_site"] == "off_array_cpu"
    assert res["audit"]["increment_site"] == "physical_pulse"


def test_vendored_coupled_runs_and_audits_full():
    topo, X, y, phi, dphi, s_fn = _setup()
    res = train_coupled_noisy(topo, X, y, phi, dphi, 1.5, DeviceConfig(),
                              n_epochs=8, seed=0, beta=0.1, s_fn=s_fn, train_sigma=0.01)
    assert res["audit"]["error_site"] == "on_array_local"
    assert res["audit"]["increment_site"] == "physical_pulse"
```

- [ ] **Step 6: Run the smoke test**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_vendor_smoke.py -q`
Expected: PASS (2 passed). If an `ImportError` appears, the `sys.path` wiring (Step 4) is wrong; if a `from ceilings import` error appears, `ceilings.py` was not copied.

- [ ] **Step 7: Confirm the whole suite is still green**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`
Expected: PASS (the 8 frozen-verdict tests + Task-1 + Task-2 tests).

- [ ] **Step 8: Commit**

```bash
git add vendor_02 conftest.py tests/test_vendor_smoke.py
git commit -m "feat(08): vendor 02's noisy-training sim @95868e7 into vendor_02/"
```

---

### Task 3: Create `h5_sweep.py` with constants and pure helpers

**Files:**
- Create: `h5_sweep.py`
- Test: `tests/test_h5_sweep.py`

**Interfaces:**
- Consumes: `audit.Audit`, `audit.PHYSICAL`, `audit.CPU`; `verdict.FORECLOSE_SNR`.
- Produces: `RULES = ("half", "full")`; `DEFAULT_SIGMAS` (list, includes `0.01`); `rule_audit(rule: str) -> Audit` (half→`(CPU,PHYSICAL)`, full→`(PHYSICAL,PHYSICAL)`, else `ValueError`); `foreclosure_wall(sigmas, snr_means, thresh=FORECLOSE_SNR) -> float | None` (first train_sigma, scanned ascending, whose SNR `< thresh`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_h5_sweep.py`:

```python
"""Phase-0 H5 runner: pure helpers, sweep integration, verdict assembly, and CLI."""
from h5_sweep import foreclosure_wall, rule_audit


def test_foreclosure_wall_finds_first_subthreshold():
    sigmas = [0.0, 0.01, 0.05, 0.1]
    snr = [5.0, 4.0, 0.5, 0.1]
    assert foreclosure_wall(sigmas, snr) == 0.05


def test_foreclosure_wall_none_when_robust():
    assert foreclosure_wall([0.0, 0.1], [5.0, 3.2]) is None


def test_foreclosure_wall_scans_in_sigma_order():
    # input order is scrambled; the wall is still found by ascending train_sigma
    assert foreclosure_wall([0.1, 0.0, 0.05], [0.2, 5.0, 4.0]) == 0.1


def test_rule_audit_half_is_not_physical():
    assert rule_audit("half").is_physical() is False


def test_rule_audit_full_is_physical():
    assert rule_audit("full").is_physical() is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_h5_sweep.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'h5_sweep'`.

- [ ] **Step 3: Create `h5_sweep.py` with the imports, constants, and pure helpers**

```python
"""Phase-0 H5 read-noise gate for 02's in-fabric rules on a floating-gate array.

A THIN DRIVER over the vendored 02 sim (vendor_02/, ported @95868e7): no physics is
reimplemented. For each (rule, train_sigma) it trains the vendored noisy rule, evaluates
clean-convention separation/floor, and forms SNR = separation/floor (02's task.verdict
definition). It records the FROZEN primary verdict (classify_h5, keyed on the buildable
half-co-located rule) PLUS an elevated co-equal in-fabric verdict (the both-physical
fully-co-located rule) PLUS the two-axis audit for both rules. Outcomes recorded, not asserted.
See docs/superpowers/specs/2026-06-24-phase0-h5-sweep-runner-design.md."""
import argparse
import json
import os
import sys

import numpy as np

# Make the vendored 02 sim importable when run as a script (conftest only covers pytest).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor_02"))

from crossbar import build_full_diff, make_read_nonlinearity, make_cocontent
from manhattan import evaluate_separation
from readnoise import train_manhattan_noisy, train_coupled_noisy
from memristor import DeviceConfig
from task import parity3_data, verdict as task_verdict
from audit import Audit, PHYSICAL, CPU
from verdict import classify_h5, SIGMA_REALISTIC_FG, PASS_SNR, FORECLOSE_SNR

RULES = ("half", "full")
# fraction-of-span training read-noise grid; includes sigma_fg=0.01 and brackets it below/above.
DEFAULT_SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def rule_audit(rule):
    """Two-axis provenance for a rule. half = off-array CPU gradient + physical write
    (NOT both-physical); full = in-fabric local gradient + physical write (both-physical)."""
    a = Audit()
    if rule == "half":
        a.record(CPU, PHYSICAL)
    elif rule == "full":
        a.record(PHYSICAL, PHYSICAL)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return a


def foreclosure_wall(sigmas, snr_means, thresh=FORECLOSE_SNR):
    """First train_sigma (scanned in ascending sigma order) whose mean SNR < thresh — the
    foreclosure cliff — or None if SNR never falls below thresh in the swept range."""
    for s, snr in sorted(zip(sigmas, snr_means), key=lambda p: p[0]):
        if snr < thresh:
            return float(s)
    return None
```

- [ ] **Step 4: Run to verify the helper tests pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_h5_sweep.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add h5_sweep.py tests/test_h5_sweep.py
git commit -m "feat(08): h5_sweep scaffold — constants + rule_audit + foreclosure_wall"
```

---

### Task 4: Per-cell SNR computation and the sweep

**Files:**
- Modify: `h5_sweep.py` (append `_build`, `_cell`, `sweep`)
- Test: `tests/test_h5_sweep.py` (append sweep tests)

**Interfaces:**
- Consumes: vendored `build_full_diff`, `make_read_nonlinearity`, `make_cocontent`, `parity3_data`, `DeviceConfig`, `train_manhattan_noisy`, `train_coupled_noisy`, `evaluate_separation`, `task_verdict`.
- Produces: `sweep(rule, sigmas, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3, n_epochs=300, seeds=range(10)) -> list[row]` where each `row = {"rule", "train_sigma": float, "snr_mean": float, "snr_per_seed": list[float], "sep_mean": float, "audit": dict}`, one per sigma in input order.

- [ ] **Step 1: Write the failing sweep tests**

Append to `tests/test_h5_sweep.py`:

```python
from h5_sweep import sweep


def test_sweep_returns_one_snr_row_per_sigma():
    rows = sweep("half", [0.0, 0.1], n_epochs=10, seeds=range(2))
    assert [r["train_sigma"] for r in rows] == [0.0, 0.1]
    for r in rows:
        assert r["snr_mean"] >= 0.0
        assert len(r["snr_per_seed"]) == 2
        assert r["audit"]["error_site"] == "off_array_cpu"


def test_sweep_full_rule_carries_in_fabric_audit():
    rows = sweep("full", [0.01], n_epochs=10, seeds=range(2))
    assert rows[0]["audit"]["error_site"] == "on_array_local"


def test_sweep_is_deterministic():
    a = sweep("half", [0.05], n_epochs=10, seeds=range(2))
    b = sweep("half", [0.05], n_epochs=10, seeds=range(2))
    assert a[0]["snr_per_seed"] == b[0]["snr_per_seed"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_h5_sweep.py -q`
Expected: FAIL with `ImportError: cannot import name 'sweep'`.

- [ ] **Step 3: Append `_build`, `_cell`, and `sweep` to `h5_sweep.py`**

```python
def _build(H=8, a=0.4):
    """Rebuild the parity-3 task + differential crossbar topology + read nonlinearity +
    co-content (orchestration only; all builders are vendored from 02)."""
    topo = build_full_diff(H)
    X, y = parity3_data()
    phi, dphi = make_read_nonlinearity(a)
    s_fn = make_cocontent(a)
    return topo, X, y, phi, dphi, s_fn


def _cell(rule, train_sigma, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
          n_epochs=300, seeds=range(10)):
    """One (rule, train_sigma) cell. Trains each seed's noisy rule, evaluates clean-convention
    (sep, floor), and forms per-seed SNR via 02's task.verdict; returns the aggregated row.
    The eval RNG is a fresh fixed-seed generator -> pure/deterministic (mirrors 02's sweep)."""
    topo, X, y, phi, dphi, s_fn = _build(H, a)
    cfg = DeviceConfig()
    eval_rng = np.random.default_rng(12345)
    snrs, seps, audit = [], [], None
    for s in seeds:
        if rule == "half":
            res = train_manhattan_noisy(topo, X, y, phi, dphi, V_drive, cfg,
                                        n_epochs=n_epochs, seed=int(s), target=target,
                                        train_sigma=train_sigma)
        elif rule == "full":
            res = train_coupled_noisy(topo, X, y, phi, dphi, V_drive, cfg,
                                      n_epochs=n_epochs, seed=int(s), target=target,
                                      beta=beta, s_fn=s_fn, train_sigma=train_sigma, mode="sign")
        else:
            raise ValueError(f"unknown rule {rule!r}")
        audit = res["audit"]
        sep, floor = evaluate_separation(topo, res["g"], X, y, phi, dphi, V_drive, cfg, eval_rng)
        snrs.append(float(task_verdict(sep, floor)["snr"]))
        seps.append(float(sep))
    return {"rule": rule, "train_sigma": float(train_sigma),
            "snr_mean": float(np.mean(snrs)), "snr_per_seed": snrs,
            "sep_mean": float(np.mean(seps)), "audit": audit}


def sweep(rule, sigmas, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
          n_epochs=300, seeds=range(10)):
    """Run `rule` over the train_sigma grid (serial); one SNR row per sigma, input order."""
    seeds = list(seeds)
    return [_cell(rule, s, H=H, a=a, V_drive=V_drive, beta=beta, target=target,
                  n_epochs=n_epochs, seeds=seeds) for s in sigmas]
```

- [ ] **Step 4: Run to verify the sweep tests pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_h5_sweep.py -q`
Expected: PASS (8 passed — 5 helper + 3 sweep).

- [ ] **Step 5: Commit**

```bash
git add h5_sweep.py tests/test_h5_sweep.py
git commit -m "feat(08): h5_sweep per-cell SNR + sweep over train_sigma"
```

---

### Task 5: Assemble the dual verdict + audits (`run_h5`)

**Files:**
- Modify: `h5_sweep.py` (append `run_h5`)
- Test: `tests/test_h5_sweep.py` (append assembly tests)

**Interfaces:**
- Consumes: `sweep`, `foreclosure_wall`, `rule_audit`, `classify_h5`, `task_verdict`, `PASS_SNR`, `FORECLOSE_SNR`, `SIGMA_REALISTIC_FG`.
- Produces: `run_h5(sigmas=None, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3, n_epochs=300, seeds=range(10), sigma_fg=SIGMA_REALISTIC_FG, sigma_confirmed=False) -> dict` with keys `config`, `curves` (`{"half":[rows],"full":[rows]}` sorted by sigma), `walls` (`{"half":float|None,"full":float|None}`), `snr_half`, `snr_full`, `verdict` (frozen `classify_h5` dict), `in_fabric_verdict` (dict keyed on `snr_full`), `audit` (`{"half":{...},"full":{...}}` JSON-safe).

- [ ] **Step 1: Write the failing assembly tests**

Append to `tests/test_h5_sweep.py`:

```python
from h5_sweep import run_h5
from verdict import SIGMA_REALISTIC_FG


def test_run_h5_assembles_both_verdicts_and_audits():
    res = run_h5(sigmas=[0.0, 0.01, 0.1], seeds=range(2), n_epochs=10)
    # frozen primary verdict keyed on the half rule
    assert res["verdict"]["status"] in {"PASS", "BOUNDARY", "FORECLOSE"}
    assert res["verdict"]["snr_half"] == res["snr_half"]
    assert res["verdict"]["provisional"] is True
    assert res["config"]["sigma_fg"] == SIGMA_REALISTIC_FG
    # elevated co-equal in-fabric verdict keyed on the full rule
    assert res["in_fabric_verdict"]["snr_full"] == res["snr_full"]
    assert res["in_fabric_verdict"]["status"] in {"PASS", "BOUNDARY", "FORECLOSE"}
    assert res["in_fabric_verdict"]["is_physical"] is True
    # two-axis audit: half NOT both-physical, full both-physical
    assert res["audit"]["half"]["is_physical"] is False
    assert res["audit"]["full"]["is_physical"] is True


def test_run_h5_inserts_sigma_fg_if_missing():
    res = run_h5(sigmas=[0.0, 0.1], seeds=range(1), n_epochs=8)
    assert SIGMA_REALISTIC_FG in res["config"]["sigmas"]


def test_run_h5_audit_breakdown_is_json_safe():
    import json
    res = run_h5(sigmas=[0.01], seeds=range(1), n_epochs=8)
    json.dumps(res)  # must not raise (tuple keys would)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_h5_sweep.py -q`
Expected: FAIL with `ImportError: cannot import name 'run_h5'`.

- [ ] **Step 3: Append `run_h5` to `h5_sweep.py`**

```python
def _status_from_snr(snr):
    """The same PASS/BOUNDARY/FORECLOSE bands classify_h5 uses, applied to the full rule
    so its in-fabric verdict is recorded co-equally (not just as a collapse boolean)."""
    if snr >= PASS_SNR:
        return "PASS"
    if snr < FORECLOSE_SNR:
        return "FORECLOSE"
    return "BOUNDARY"


def run_h5(sigmas=None, *, H=8, a=0.4, V_drive=1.5, beta=0.1, target=0.3,
           n_epochs=300, seeds=range(10), sigma_fg=SIGMA_REALISTIC_FG, sigma_confirmed=False):
    """Drive both rules over the train_sigma grid; assemble the frozen primary verdict, the
    elevated in-fabric verdict, the foreclosure walls, and the two-axis audits."""
    seeds = list(seeds)
    sigmas = list(DEFAULT_SIGMAS if sigmas is None else sigmas)
    if sigma_fg not in sigmas:
        sigmas = sorted(set(sigmas) | {sigma_fg})

    curves = {rule: sorted(sweep(rule, sigmas, H=H, a=a, V_drive=V_drive, beta=beta,
                                 target=target, n_epochs=n_epochs, seeds=seeds),
                           key=lambda r: r["train_sigma"])
              for rule in RULES}

    def _snr_at(rule, s):
        for r in curves[rule]:
            if r["train_sigma"] == s:
                return r["snr_mean"]
        raise KeyError(s)

    snr_half = _snr_at("half", sigma_fg)
    snr_full = _snr_at("full", sigma_fg)

    primary = classify_h5(snr_half, snr_full, sigma_fg=sigma_fg, sigma_confirmed=sigma_confirmed)

    walls = {rule: foreclosure_wall([r["train_sigma"] for r in curves[rule]],
                                    [r["snr_mean"] for r in curves[rule]])
             for rule in RULES}

    in_fabric = {"rule": "full_coupled_two_read", "snr_full": float(snr_full),
                 "status": _status_from_snr(snr_full), "wall": walls["full"],
                 "is_physical": True,
                 "note": "both-physical / physics-local; the read-noise-fragile rule"}

    audit = {}
    for rule in RULES:
        a_rec = rule_audit(rule)
        audit[rule] = {"breakdown": {f"{g}|{w}": c for (g, w), c in a_rec.breakdown().items()},
                       "is_physical": a_rec.is_physical()}

    return {
        "config": {"H": H, "a": a, "V_drive": V_drive, "beta": beta, "target": target,
                   "n_epochs": n_epochs, "n_seeds": len(seeds), "sigmas": sigmas,
                   "sigma_fg": float(sigma_fg)},
        "curves": curves, "walls": walls,
        "snr_half": float(snr_half), "snr_full": float(snr_full),
        "verdict": primary, "in_fabric_verdict": in_fabric, "audit": audit,
    }
```

- [ ] **Step 4: Run to verify the assembly tests pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_h5_sweep.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add h5_sweep.py tests/test_h5_sweep.py
git commit -m "feat(08): h5_sweep run_h5 — frozen + elevated in-fabric verdicts + audits"
```

---

### Task 6: CLI, summary print, and JSON dump (`main`)

**Files:**
- Modify: `h5_sweep.py` (append `_print_summary`, `main`, `__main__` guard)
- Test: `tests/test_h5_sweep.py` (append CLI test)

**Interfaces:**
- Consumes: `run_h5`.
- Produces: `main(argv=None) -> dict` (parses args, runs, prints, writes JSON to `--out`, returns the result dict). Flags: `--out` (default `h5_sweep_parity3.json`), `--epochs` (300), `--seeds` (10), `--beta` (0.1), `--sigmas` (comma list, overrides grid), `--quick` (tiny smoke: sigmas `[0.0, 0.01, 0.1]`, 2 seeds, 20 epochs).

- [ ] **Step 1: Write the failing CLI test**

Append to `tests/test_h5_sweep.py`:

```python
import json
from h5_sweep import main


def test_main_quick_writes_json_with_both_verdicts(tmp_path):
    out = tmp_path / "h5.json"
    res = main(["--quick", "--out", str(out)])
    assert out.exists()
    data = json.loads(out.read_text())
    assert set(data) >= {"config", "curves", "walls", "verdict",
                         "in_fabric_verdict", "audit", "snr_half", "snr_full"}
    assert data["verdict"]["provisional"] is True
    assert data["audit"]["half"]["is_physical"] is False
    assert data["audit"]["full"]["is_physical"] is True
    assert res["config"]["n_seeds"] == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_h5_sweep.py -q`
Expected: FAIL with `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Append `_print_summary`, `main`, and the `__main__` guard to `h5_sweep.py`**

```python
_QUICK = {"sigmas": [0.0, 0.01, 0.1], "seeds": range(2), "n_epochs": 20}


def _print_summary(res):
    c = res["config"]
    half = {r["train_sigma"]: r["snr_mean"] for r in res["curves"]["half"]}
    full = {r["train_sigma"]: r["snr_mean"] for r in res["curves"]["full"]}
    print("=" * 78)
    print("PHASE-0 H5 read-noise gate — 02's rules on a floating-gate array (parity-3)")
    print("=" * 78)
    print(f"sigma_fg={c['sigma_fg']}  n_seeds={c['n_seeds']}  n_epochs={c['n_epochs']}  beta={c['beta']}")
    print("\n-- mean SNR vs train_sigma --")
    print(f"  {'train_sigma':>12} {'half(SNR)':>12} {'full(SNR)':>12}")
    for s in c["sigmas"]:
        print(f"  {s:>12} {half.get(s, float('nan')):>12.3f} {full.get(s, float('nan')):>12.3f}")
    print(f"\nforeclosure walls (train_sigma): half={res['walls']['half']}  full={res['walls']['full']}")
    v = res["verdict"]
    print(f"\nPRIMARY (frozen classify_h5, keys on the buildable HALF rule): {v['status']}")
    print(f"  snr_half={v['snr_half']:.3f}  provisional={v['provisional']}  "
          f"full_collapses_as_expected={v['full_rule_collapses_as_expected']}")
    f = res["in_fabric_verdict"]
    print(f"IN-FABRIC (elevated, the both-physical FULL rule): {f['status']}  "
          f"snr_full={f['snr_full']:.3f}  is_physical={f['is_physical']}")
    print(f"audit: half is_physical={res['audit']['half']['is_physical']}  "
          f"full is_physical={res['audit']['full']['is_physical']}  "
          f"(robust rule is NOT in-fabric; in-fabric rule is the fragile one)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Phase-0 H5 read-noise gate for 02's rules on a floating-gate array.")
    ap.add_argument("--out", default="h5_sweep_parity3.json")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sigmas", default=None,
                    help="comma-separated train_sigma grid (overrides DEFAULT_SIGMAS)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny local smoke (few sigmas/seeds/epochs)")
    args = ap.parse_args(argv)

    sigmas = [float(x) for x in args.sigmas.split(",")] if args.sigmas else None
    if args.quick:
        res = run_h5(sigmas=_QUICK["sigmas"], seeds=_QUICK["seeds"],
                     n_epochs=_QUICK["n_epochs"], beta=args.beta)
    else:
        res = run_h5(sigmas=sigmas, seeds=range(args.seeds), n_epochs=args.epochs, beta=args.beta)

    _print_summary(res)
    with open(args.out, "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"\nraw numbers -> {args.out}")
    return res


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the CLI test passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_h5_sweep.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Run the full suite green**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`
Expected: PASS (frozen-verdict 8 + audit 3 + vendor smoke 2 + h5_sweep 12).

- [ ] **Step 6: Validate the runner end-to-end with a real quick smoke run**

Run (executes the `__main__`/argparse/script path the in-process test does not):
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python h5_sweep.py --quick --out h5_quick_smoke.json && rm h5_quick_smoke.json
```
Expected: prints the summary table (SNR-vs-σ for both rules, both verdicts, both `is_physical` flags) and writes/removes the disposable JSON. This is the validation smoke; it is **not** the gated full Phase-0 run.

- [ ] **Step 7: Commit**

```bash
git add h5_sweep.py tests/test_h5_sweep.py
git commit -m "feat(08): h5_sweep CLI + summary + JSON dump; quick-smoke validated"
```

---

## After the plan

The runner is built and smoke-validated. **The full Phase-0 sweep + Observed/Interpretation findings (`docs/superpowers/findings/`) are a separate, gated step** — do not run the full grid or write findings without the user's explicit go-ahead (parent spec §8; this plan's Global Constraints).
