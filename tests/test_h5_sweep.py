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
