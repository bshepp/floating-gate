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
    """The committed nominal FG operating point: a HIGH-PRECISION, CLOSED-LOOP-programmed floating gate.
    The raw FG update (Fowler-Nordheim tunneling / hot-electron injection) is strongly nonlinear, but
    closed-loop program-verify compensates it to land on each target level, so the EFFECTIVE update is
    only mildly nonlinear (~nu=2) at ~8-bit precision (n_levels=256). Strong tunneling/injection
    asymmetry and excellent retention (drift=0) are kept; read is quiet. NOTE (verified 2026-06-25): the
    RAW un-compensated nonlinearity (nu>=3) FORECLOSES even at continuous precision -- so UPDATE
    NONLINEARITY, not granularity, is the binding non-ideality; the nonlin sweep (FG_SWEEPS) probes it
    directly (nu->0 = 02's pulse-co-design limit). g_min/g_max/dG0 keep 02's defaults."""
    return DeviceConfig(
        nonlin=2.0,        # EFFECTIVE post-closed-loop nonlinearity (raw FG update is stronger; nu>=3 forecloses)
        asym=0.3,          # < 02's 0.5: tunneling-erase vs injection-program are more asymmetric
        n_levels=256,      # ~8-bit closed-loop programming precision (Hasler FG reaches 10-14 bit w/ feedback)
        sigma_write=0.03,  # cycle-to-cycle injection variability (~02's 0.02, slightly higher)
        sigma_read=SIGMA_FG,
        drift=0.0,         # FG charge retention is years -> no decay over a training run
    )


# One-at-a-time Observed robustness sweeps around the nominal (keys are DeviceConfig field names).
FG_SWEEPS = {
    "nonlin":      [0.0, 1.0, 2.0, 3.0],      # KEY axis: 0 = nu->0 co-design limit (learns); >=3 raw FG (forecloses)
    "asym":        [0.2, 0.3, 0.5, 1.0],      # 1.0 = symmetric
    "n_levels":    [32, 64, 128, 256, 512],   # programming-precision cliff around the nominal
    "sigma_write": [0.02, 0.03, 0.05, 0.1],
    "sigma_read":  [0.003, 0.01, 0.02, 0.05],
}

# Read-averaging factors K for the robust-in-fabric search (objective 2). Ascending, starts at 1.
ROBUST_K_GRID = [1, 4, 16, 100, 1_000, 10_000]
