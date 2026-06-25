"""Pre-registered verdicts for 08-floating-gate — FROZEN decision functions.

Portfolio discipline (cf. 03's `classify_*`): the verdict is a pure function of summary
statistics, with thresholds + operating point fixed *in this file, committed, before the
run*. A reviewer can brute-force its input space (see tests/test_verdict.py). The verdict
NEVER reads the raw sweep; it classifies the sweep's reported numbers.

Only `classify_h5` (Phase 0) is frozen now. The Phase-1 / Phase-2 verdicts will be frozen
here before those phases run, once the floating-gate device-characterization protocol is
fixed.

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


def classify_build(*args, **kwargs):
    """Phase-2 in-fabric build verdict — NOT YET FROZEN. Freeze before the build run."""
    raise NotImplementedError("classify_build is frozen before Phase 2, not before Phase 0.")
