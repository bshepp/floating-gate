"""Measured floating-gate parameters -> the vendored DeviceConfig (Phase 1b). Pure inversions of 02's
phenomenological device model (vendor_02/memristor.py), so a bench measurement on a real FG array
feeds classify_transfer with params_confirmed=True. Each function inverts ONE model equation:

  potentiation step   dG = dG0 * (1-G)**nu                 -> nonlin (nu), dG0
  depression step     dG = dG0 * asym * G**nu              -> asym
  write noise         std(dG) = sigma_write * dG0          -> sigma_write
  read noise          std(read) = sigma_read * span        -> sigma_read
  quantizer           n levels over span, snap-to-nearest  -> n_levels from the closed-loop landing error
  drift               g <- g + drift*(g_rest - g) / epoch  -> drift from retention

Two write modes, because the Phase-1 findings showed they differ decisively (raw nu>=3 forecloses):
  closed_loop  each learner update is a program-verify write: the learner sees the EFFECTIVE
               nonlinearity and the landing error becomes the quantizer (the inference-grade path).
  open_loop    each learner update is ONE raw injection/tunneling pulse with no verify (the only
               option for an in-fabric, fabric-triggered write): the learner sees the RAW nonlinearity
               and the pulse's own granularity (no quantizer).
All quantities are normalized: G and errors as fractions of the chosen conductance (current) span.

Injection-only parts (the RASP / OTC2902 family, 2026-11-08): hot-electron injection is the only per-cell
mechanism and it only RAISES the pFET current; tunneling erase is global. A single cell therefore has no
per-cell depression and `asym` is the wrong abstraction. The device-faithful weight is a DIFFERENTIAL PAIR
w = I+ - I-: "depress" = inject the minus cell. Both directions are then the same mechanism (asym -> 1.0),
cells only grow between global erases, and the erase + re-inject cycle (not pulses per cell) is what the
endurance budget counts. `device_config_from_measurements(..., differential_pair=True)` applies the
asym -> 1.0 mapping; fg_diffpair.py (Phase 1c) simulates the pair itself.
08-authored; never edits vendor_02/.
"""
import math

from memristor import DeviceConfig   # vendored 02 device model

_SQRT12 = math.sqrt(12.0)


# --- precision -------------------------------------------------------------------------------------

def rms_error_from_levels(n_levels):
    """RMS snap-to-nearest error (fraction of span) of a uniform n-level quantizer: step/sqrt(12),
    step = 1/(n-1). Continuous (n<=1) -> 0."""
    n = int(n_levels)
    if n <= 1:
        return 0.0
    return 1.0 / ((n - 1) * _SQRT12)


def levels_from_rms_error(rms_error):
    """Quantizer level count whose snap-to-nearest RMS error equals a measured closed-loop LANDING error
    (fraction of span): n = 1 + 1/(eps*sqrt(12)). eps<=0 -> 0 (continuous, 02's convention)."""
    eps = float(rms_error)
    if eps <= 0.0:
        return 0
    return max(2, int(round(1.0 + 1.0 / (eps * _SQRT12))))


def levels_from_bits(bits):
    """2**bits levels (a vendor 'N-bit programming' claim)."""
    return int(round(2.0 ** float(bits)))


def bits_from_levels(n_levels):
    """log2 of a level count (0 for continuous, by convention)."""
    n = int(n_levels)
    return 0.0 if n <= 1 else math.log2(n)


# --- update shape ----------------------------------------------------------------------------------

def nonlin_from_pot_steps(step_lo, step_hi, G_lo, G_hi):
    """nu from two open-loop POTENTIATION steps measured at states G_lo < G_hi:
    step_hi/step_lo = ((1-G_hi)/(1-G_lo))**nu. Steps > 0 as fractions of span. Clipped at 0."""
    if not (0.0 <= G_lo < G_hi < 1.0):
        raise ValueError("need 0 <= G_lo < G_hi < 1")
    if step_lo <= 0 or step_hi <= 0:
        raise ValueError("potentiation steps must be > 0")
    nu = math.log(step_hi / step_lo) / math.log((1.0 - G_hi) / (1.0 - G_lo))
    return max(0.0, float(nu))


def nonlin_from_dep_steps(step_lo, step_hi, G_lo, G_hi):
    """nu from two open-loop DEPRESSION step magnitudes at states 0 < G_lo < G_hi:
    |step_hi|/|step_lo| = (G_hi/G_lo)**nu. Clipped at 0."""
    if not (0.0 < G_lo < G_hi <= 1.0):
        raise ValueError("need 0 < G_lo < G_hi <= 1")
    a, b = abs(step_lo), abs(step_hi)
    if a <= 0 or b <= 0:
        raise ValueError("depression steps must be nonzero")
    nu = math.log(b / a) / math.log(G_hi / G_lo)
    return max(0.0, float(nu))


def dG0_from_pot_step(step, G, nu):
    """Nominal pulse step dG0 (fraction of span) from one potentiation step at state G: dG0 = step/(1-G)**nu."""
    if not (0.0 <= G < 1.0):
        raise ValueError("need 0 <= G < 1")
    return float(step) / (1.0 - G) ** float(nu)


def asym_from_steps(dep_step, pot_step, G, nu):
    """Depression/potentiation ratio at the SAME state G: asym = (|dep|/G**nu) / (pot/(1-G)**nu).
    At G=0.5 the state factors cancel and asym = |dep|/pot."""
    if not (0.0 < G < 1.0):
        raise ValueError("need 0 < G < 1")
    if pot_step <= 0:
        raise ValueError("potentiation step must be > 0")
    return (abs(dep_step) / G ** float(nu)) / (float(pot_step) / (1.0 - G) ** float(nu))


def sigma_write_from_step_std(step_std, dG0):
    """Cycle-to-cycle write noise in 02's units of dG0: std of repeated nominal pulses / dG0."""
    if dG0 <= 0:
        raise ValueError("dG0 must be > 0")
    return float(step_std) / float(dG0)


# --- read + retention ------------------------------------------------------------------------------

def sigma_read_from_std(read_std, span):
    """Read noise as a fraction of the conductance (current) span: std of repeated reads / span."""
    if span <= 0:
        raise ValueError("span must be > 0")
    return float(read_std) / float(span)


def drift_from_retention(G0, Gt, n_epochs, G_rest=0.5):
    """Per-epoch drift from a retention measurement: (Gt - rest) = (G0 - rest) * (1 - drift)**n.
    G0 == rest carries no information -> 0. Result clipped to [0, 1]."""
    if n_epochs <= 0:
        raise ValueError("n_epochs must be > 0")
    d0 = G0 - G_rest
    if d0 == 0:
        return 0.0
    ratio = (Gt - G_rest) / d0
    if ratio <= 0:
        return 1.0          # fully relaxed (or overshot) within the window
    return float(min(1.0, max(0.0, 1.0 - ratio ** (1.0 / float(n_epochs)))))


# --- assembly --------------------------------------------------------------------------------------

REQUIRED_MEASUREMENTS = ("sigma_read", "nonlin_raw", "asym", "sigma_write", "drift")


def device_config_from_measurements(m, write_mode="closed_loop", base=None, differential_pair=False):
    """Build the DeviceConfig a learner would see on the measured array.

    m: dict of normalized measurements -- REQUIRED_MEASUREMENTS plus, for closed_loop,
       `prog_rms_error` (closed-loop landing error, fraction of span) and optionally
       `nonlin_effective` (post-verify nonlinearity; defaults to nonlin_raw, the conservative choice).
       Optional `dG0` overrides the nominal step. `base` seeds g_min/g_max (default: 02's defaults).
    write_mode: "closed_loop" (effective nu + landing-error quantizer) or "open_loop" (raw nu, no
       quantizer) -- see module docstring.
    differential_pair: the weight is I+ - I- on an injection-only part, so both update directions are
       the same mechanism -> asym is forced to 1.0 (the measured `asym` is ignored; on such a part it is
       not measurable per cell anyway). The pair's own dynamics (saturation, global erase) are NOT in a
       single DeviceConfig; fg_diffpair simulates them."""
    missing = [k for k in REQUIRED_MEASUREMENTS if k not in m]
    if missing:
        raise KeyError(f"missing measurements: {missing}")
    base = DeviceConfig() if base is None else base
    asym = 1.0 if differential_pair else float(m["asym"])
    if write_mode == "closed_loop":
        if "prog_rms_error" not in m:
            raise KeyError("closed_loop needs prog_rms_error")
        nonlin = float(m.get("nonlin_effective", m["nonlin_raw"]))
        n_levels = levels_from_rms_error(m["prog_rms_error"])
    elif write_mode == "open_loop":
        nonlin = float(m["nonlin_raw"])
        n_levels = 0
    else:
        raise ValueError(f"unknown write_mode {write_mode!r}")
    return DeviceConfig(
        g_min=base.g_min, g_max=base.g_max,
        dG0=float(m.get("dG0", base.dG0)),
        nonlin=nonlin, asym=asym, n_levels=int(n_levels),
        sigma_write=float(m["sigma_write"]), sigma_read=float(m["sigma_read"]),
        drift=float(m["drift"]),
    )


# The bench protocol, one row per DeviceConfig input (what to measure, how, and which field it feeds).
# Kept as data so the spec, the tests and a future runner all read the same list.
CHARACTERIZATION_PROTOCOL = (
    {"name": "span", "feeds": "normalization",
     "procedure": "Pick the FG current window the learner will use (e.g. one decade inside the "
                  "programmable range); span = I_max - I_min; every quantity below is a fraction of it."},
    {"name": "read noise", "feeds": "sigma_read",
     "procedure": "Program one FG mid-span; take >=100 repeated reads at the learner's read bandwidth; "
                  "sigma_read = std / span. Confirms Phase 0's provisional sigma_fg."},
    {"name": "open-loop potentiation ladder", "feeds": "nonlin_raw, dG0",
     "procedure": "From G~0, apply identical injection pulses one at a time, reading after each; fit "
                  "step vs G to dG0*(1-G)**nu (nonlin_from_pot_steps / dG0_from_pot_step)."},
    {"name": "per-cell depression test", "feeds": "asym (single cell) OR differential_pair=True",
     "procedure": "Try to LOWER one cell's current without touching its neighbours. On the RASP / OTC2902 "
                  "family this is expected to be impossible (tunneling erase is global; injection only "
                  "raises current): record 'injection-only', set differential_pair=True (asym -> 1.0) and "
                  "skip the depression ladder. If per-cell decrement exists, run the mirror ladder from "
                  "G~1 and use asym_from_steps at matched G."},
    {"name": "differential-pair ladder", "feeds": "nonlin_raw of the MINUS cell (consistency), dG0",
     "procedure": "Program a pair; inject the minus cell from G~0 one pulse at a time, reading I+ - I-; "
                  "the step-vs-G fit must match the plus-cell ladder (same mechanism). A mismatch is a "
                  "pair-mismatch term to record."},
    {"name": "global erase + re-inject cycle", "feeds": "renorm_sigma, erase cost (fg_diffpair)",
     "procedure": "Erase the array (tunneling), closed-loop re-inject every pair to its retained "
                  "differential value with the smaller cell at G~0; renorm_sigma = RMS (landed - "
                  "target)/span over all pairs; record wall-clock per cycle. This is the maintenance op an "
                  "injection-only learner needs when cells run out of headroom."},
    {"name": "cycle-to-cycle", "feeds": "sigma_write",
     "procedure": ">=30 repeats of (program to mid-span, one nominal pulse, read); "
                  "sigma_write = std(step)/dG0."},
    {"name": "closed-loop landing error", "feeds": "prog_rms_error -> n_levels, nonlin_effective",
     "procedure": "Program-verify a ladder of >=32 targets across the span; prog_rms_error = RMS "
                  "(landed - target)/span; nonlin_effective from the effective step vs G of successive "
                  "verified targets."},
    {"name": "retention", "feeds": "drift",
     "procedure": "Program cells to G=0.1 and G=0.9; re-read over the wall-clock of one training run; "
                  "drift_from_retention per epoch-equivalent."},
    {"name": "endurance", "feeds": "ENDURANCE_BUDGET",
     "procedure": "Vendor/datasheet program-erase cycle rating, or cycle a sacrificial cell until the "
                  "step size degrades by 2x. On an injection-only part compare it to the number of GLOBAL "
                  "ERASE cycles per training run (fg_diffpair counts them), not to pulses per cell: the "
                  "tunneling oxide sees one stress per erase."},
    {"name": "run-mode local write", "feeds": "classify_access.run_mode_local_write",
     "procedure": "Can a CAB-local analog signal drive injection on a weight FG while in run mode, with "
                  "no program-mode switch and no microprocessor in the loop? Answered False for the "
                  "OTC2902E04 by identity with the published design (2026-11-08); a bench confirmation is "
                  "cheap once the board is in hand. This decides the in-fabric path, not the purchase."},
)
