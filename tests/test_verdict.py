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
        classify_transfer()
    with pytest.raises(NotImplementedError):
        classify_build()
