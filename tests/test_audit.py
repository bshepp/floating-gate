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
