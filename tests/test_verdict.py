"""Brute-force the frozen Phase-0 decision function over its input space.

The pre-registration audit: confirm `classify_h5` returns the documented verdict on
representative inputs and at the threshold boundaries, and that the consistency / provisional
flags behave as documented.
"""
import pytest

from verdict import (
    classify_h5, classify_transfer, classify_build,
    PASS_SNR, FORECLOSE_SNR, SIGMA_REALISTIC_FG,
)


def test_pass_when_half_rule_tolerates_realistic_noise():
    """snr_half >= 3 at realistic noise -> PASS (the buildable object survives)."""
    res = classify_h5(snr_half=3.5, snr_full=0.4)
    assert res["status"] == "PASS"


def test_foreclose_when_half_rule_fragile():
    """snr_half < 1 -> FORECLOSE (even the robust half-co-located rule dies)."""
    res = classify_h5(snr_half=0.4, snr_full=0.2)
    assert res["status"] == "FORECLOSE"


def test_boundary_in_between():
    res = classify_h5(snr_half=2.0, snr_full=0.5)
    assert res["status"] == "BOUNDARY"


def test_verdict_keys_only_on_half_rule():
    """A healthy half rule PASSes even though the full (two-read) rule has collapsed."""
    res = classify_h5(snr_half=4.0, snr_full=0.1)
    assert res["status"] == "PASS"
    assert res["full_rule_collapses_as_expected"] is True


def test_threshold_edges_are_inclusive_as_documented():
    assert classify_h5(PASS_SNR, 0.0)["status"] == "PASS"
    assert classify_h5(FORECLOSE_SNR, 0.0)["status"] == "BOUNDARY"   # SNR==1 not foreclosed
    assert classify_h5(FORECLOSE_SNR - 1e-9, 0.0)["status"] == "FORECLOSE"


def test_full_rule_collapse_flag_tracks_snr_full():
    """The consistency check: collapse flag is True iff snr_full < FORECLOSE_SNR."""
    assert classify_h5(3.5, 0.5)["full_rule_collapses_as_expected"] is True
    assert classify_h5(3.5, 2.0)["full_rule_collapses_as_expected"] is False


def test_verdict_is_provisional_until_device_noise_confirmed():
    """Default operating point is provisional; flips to final once confirmed."""
    prov = classify_h5(3.5, 0.4)
    assert prov["provisional"] is True
    assert prov["sigma_fg"] == SIGMA_REALISTIC_FG
    final = classify_h5(3.5, 0.4, sigma_fg=0.008, sigma_confirmed=True)
    assert final["provisional"] is False


def test_later_phase_verdicts_are_not_yet_frozen():
    with pytest.raises(NotImplementedError):
        classify_build()


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
    assert res["endurance_ok"] is True   # foreclosed by SNR, not by endurance


def test_transfer_forecloses_when_endurance_blown_even_if_half_passes():
    res = classify_transfer(snr_half=8.0, snr_full=0.0, snr_robust=0.0,
                            writes_per_cell=200_000, endurance_budget=100_000, robust_reads_K=10_000)
    assert res["status"] == "FORECLOSE"
    assert res["endurance_ok"] is False


def test_transfer_boundary_in_between():
    res = classify_transfer(snr_half=2.0, snr_full=0.0, snr_robust=0.0,
                            writes_per_cell=300, endurance_budget=100_000, robust_reads_K=1)
    assert res["status"] == "BOUNDARY"
    assert res["endurance_ok"] is True
    assert res["provisional"] is True


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
