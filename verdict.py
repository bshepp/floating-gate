"""Pre-registered verdicts for 08-floating-gate — FROZEN decision functions.

Portfolio discipline (cf. 03's `classify_*`): the verdict is a pure function of summary
statistics, with thresholds + operating point fixed *in this file, committed, before the
run*. A reviewer can brute-force its input space (see tests/test_verdict.py). The verdict
NEVER reads the raw sweep; it classifies the sweep's reported numbers.

`classify_h5` (Phase 0) and `classify_transfer` (Phase 1) are frozen. `classify_access`
(Phase 1b, the device-access gate) was frozen 2026-11-05 BEFORE any purchase. `classify_build`
(Phase 2) is frozen only once the build protocol is fixed.

Access correction (2026-11-05). The "not buyable" premise below was true when written. Since
2025 Okika Devices sells the Georgia Tech / Hasler floating-gate SoC FPAA (RASP 3.0 lineage) as
the OTC2902K development board, so a floating-gate array IS now purchasable; what remains open
is whether that part can host the genuinely in-fabric (physical, physical) update or only the
inference-grade (cpu, physical) one. `classify_access` records exactly that split. Beware the
same vendor's FlexAnalog boards: they are the Anadigm AN231E04 switched-capacitor part — the
device already classified inference-only by `fpaa-neural-net` — not floating gates.

Operating point. 08 is the *aspirational* floating-gate arm: analog floating-gate
transistors are not buyable as discrete parts (the analog form is integrated only — an
academic FG-FPAA or a CMOS/MPW run), so unlike 06 (a buyable memristor) there is no device
to characterize today. Phase 0 therefore evaluates against a PROVISIONAL `SIGMA_REALISTIC_FG`
and carries `provisional=True` until a real floating-gate array confirms it at Phase 1
(`sigma_confirmed`). The robust, device-independent output of Phase 0 is *where the wall is*
and *that the two-read rule is the fragile one*.

Caveat recorded in the spec (§3): floating-gate cells are typically LOW read-noise and
non-volatile, so this read-noise gate is expected to PASS easily. An easy PASS means "read
noise is not the wall", NOT "the device is fine" — the floating-gate wall is most likely
write-side (programming asymmetry / granularity / endurance), a Phase-1 question.
"""

# --- Frozen thresholds (the portfolio-wide PASS bar; do not retune to a result) ---
PASS_SNR = 3.0        # separation / noise-floor >= 3  -> a usable, above-floor learner
FORECLOSE_SNR = 1.0   # SNR < 1 -> no separation survives; the operating point is dead

# Provisional realistic read noise for a floating-gate readout, as a fraction of conductance
# span. Pre-registered now; MUST be reconciled with the measured device noise before the
# Phase-0 verdict is treated as final (set `sigma_confirmed=True` then). Floating-gate read
# noise is expected at or below this.
SIGMA_REALISTIC_FG = 0.01


def classify_h5(snr_half, snr_full, sigma_fg=SIGMA_REALISTIC_FG, sigma_confirmed=False):
    """Phase-0 read-noise gate for 02's in-fabric rules on a floating-gate array. FROZEN 2026-06-24.

    Inputs (all reported by the H5 sweep, which reuses 02's readnoise.py and is run separately):
      snr_half          SNR (separation / floor) of the HALF-co-located rule (one physical
                        read) evaluated at `sigma_fg` read noise in the TRAINING relaxations.
                        This is the buildable object; the verdict keys on it.
      snr_full          SNR of the FULLY-co-located rule (two independent reads) at sigma_fg.
                        02 already foreclosed this rule ~100x below realistic; its collapse
                        here is a CONSISTENCY CHECK, not a PASS requirement.
      sigma_fg          the operating point used (provisional by default).
      sigma_confirmed   True once measured against a real floating-gate device (Phase 1).

    Verdict (keys ONLY on the buildable half-co-located object):
      PASS       snr_half >= PASS_SNR        -> the floating-gate read noise is survivable;
                                                proceed to Phase 1 (where the write side is
                                                the real test).
      FORECLOSE  snr_half <  FORECLOSE_SNR   -> fragile even for the half-co-located rule;
                                                redirect to a non-contrastive update or stop
                                                before pursuing device access.
      BOUNDARY   otherwise (1 <= SNR < 3)    -> marginal; sharpen before deciding.
    """
    if snr_half >= PASS_SNR:
        status = "PASS"
    elif snr_half < FORECLOSE_SNR:
        status = "FORECLOSE"
    else:
        status = "BOUNDARY"
    return {
        "status": status,
        "snr_half": float(snr_half),
        "snr_full": float(snr_full),
        "sigma_fg": float(sigma_fg),
        "sigma_confirmed": bool(sigma_confirmed),
        # expected True (02's finding): the fully-co-located two-read rule should collapse.
        "full_rule_collapses_as_expected": bool(snr_full < FORECLOSE_SNR),
        # a verdict on a provisional operating point is provisional, by construction.
        "provisional": not bool(sigma_confirmed),
    }


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


# --- Phase-1b device-access gate (FROZEN 2026-11-05, committed before any purchase) ----------------
# The budget is PRE-REGISTERED as the cheapest alternative route to the same measurements: the low
# end of an analog-capable MPW characterization-shuttle seat (the Phase-1 spec's "$k-$10k+" access
# landscape). A board that costs more than that route does not beat it. This number is a modeling
# choice committed before the decision, NOT a purchase authorization: the user ratifies it.
ACCESS_BUDGET_USD = 5_000.0


def classify_access(is_floating_gate, price_usd, fg_programming_user_exposed,
                    run_mode_local_write=None, budget_usd=ACCESS_BUDGET_USD):
    """Phase-1b device-access gate: is a commercially available array worth buying for the
    characterization milestone, and what two-axis audit ceiling can it host AS SOLD? FROZEN 2026-11-05.

    Inputs (read off vendor documentation / primary papers; record an evidence grade beside each):
      is_floating_gate             weights are non-volatile charge on an isolated gate written by
                                   tunneling / hot-carrier injection -- NOT a switched-capacitor /
                                   SRAM-weight part (the FlexAnalog AN231E04 trap).
      price_usd                    delivered price of board + any tools required to program FGs.
      fg_programming_user_exposed  the toolchain lets us set INDIVIDUAL FG elements to our own target
                                   values and read them back. Without it the write side cannot be
                                   measured at all, so the board cannot do the characterization job.
      run_mode_local_write         can a fabric-local analog signal trigger injection on a weight FG
                                   while the circuit RUNS -- no program-mode switch, no
                                   microprocessor in the loop?  True / False / None (unknown).
      budget_usd                   the pre-registered ceiling (ACCESS_BUDGET_USD).

    Verdict (gates the purchase on the characterization job only):
      FORECLOSE  not a floating gate (wrong device), or FG programming not user-exposed (cannot
                 characterize), or price > budget ("no accessible floating-gate array at acceptable
                 cost" -- the charter's honest end for this gate).
      BUY        otherwise: the board replaces the characterization-tapeout milestone; running the
                 protocol on it flips classify_transfer's params_confirmed.
    Elevated (recorded co-equally, NOT gating -- the in-fabric question):
      audit_ceiling    best provenance the board can host as sold: "(physical, physical)" iff
                       run_mode_local_write is True, else "(cpu, physical)" = inference-grade.
      in_fabric_path   "device" (True) / "mpw" (False: a custom cell is needed) / "unknown" (None).
    A BUY with audit_ceiling "(cpu, physical)" buys CHARACTERIZATION and the inference-grade
    learner, not in-fabric learning. Do not over-read it.
    """
    within_budget = float(price_usd) <= float(budget_usd)
    if not is_floating_gate:
        status, reason = "FORECLOSE", "not a floating-gate device"
    elif not fg_programming_user_exposed:
        status, reason = "FORECLOSE", "FG programming not user-exposed; cannot characterize"
    elif not within_budget:
        status, reason = "FORECLOSE", "price exceeds the pre-registered access budget"
    else:
        status, reason = "BUY", "floating gate, FG programming exposed, within budget"
    if run_mode_local_write is True:
        ceiling, path = "(physical, physical)", "device"
    elif run_mode_local_write is False:
        ceiling, path = "(cpu, physical)", "mpw"
    else:
        ceiling, path = "(cpu, physical)", "unknown"
    return {
        "status": status,
        "reason": reason,
        "is_floating_gate": bool(is_floating_gate),
        "price_usd": float(price_usd),
        "budget_usd": float(budget_usd),
        "within_budget": bool(within_budget),
        "fg_programming_user_exposed": bool(fg_programming_user_exposed),
        "run_mode_local_write": run_mode_local_write,
        "audit_ceiling": ceiling,
        "in_fabric_path": path,
        "replaces_characterization_tapeout": status == "BUY",
        "vendor_query_needed": run_mode_local_write is None,
    }


def classify_build(*args, **kwargs):
    """Phase-2 in-fabric build verdict — NOT YET FROZEN. Freeze before the build run."""
    raise NotImplementedError("classify_build is frozen before Phase 2, not before Phase 0.")
