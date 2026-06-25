# PORTED from 00-shared-harness/commons/audit.py @e572b0f — do not edit to diverge; re-vendor on upstream change.
"""Two-axis provenance audit — "who computes the update?" as a logged, checkable fact.

Canonical hook, promoted from **04-photonic-mzi-mesh** (2026-06-21) per
``synthesis/lessons-ledger.md#audit-is-two-axis``. It records every weight update
on the portfolio-standard **two axes**::

    (gradient / error source)  x  (weight-write mechanism)

This generalizes the earlier one-axis recorder (01's ``physics-local`` / ``offline``),
02's ``error_site`` x ``increment_site``, and 03's 3-way provenance into one model.

Why two axes: a physical gradient written by an electronic controller is fast
**inference**, not a co-located **learner** (a BOUNDARY result). 04's Arm 1 (offline
backprop) and Arm 2 (in-situ adjoint) compute *identical gradient numbers* yet earn
different verdicts — the only difference is the write mechanism. A single
``physics-local`` / ``offline`` flag cannot express that; the two-axis record can.
**A PASS requires physics on BOTH axes** (see ``Audit.is_physical``).

Vocabulary is **open by default** so each substrate supplies its own axis-value
strings — 04: ``("optical", "physical-medium")``; 02: a physical read + physical
pulse but a CPU-computed error sign; 01: ``physical`` vs ``cpu``/``electronic``.
Pass ``gradient_vocab`` / ``write_mechanism_vocab`` (closed sets) to recover 04-style
typo-protection — e.g. ``Audit(gradient_vocab={"cpu","optical"},
write_mechanism_vocab={"electronic","physical-medium"})`` reproduces 04's strict
recorder exactly.

Recommended canonical poles (not enforced unless you pass a vocab):

==================  ==========================  ====================
axis                physical pole               non-physical pole
==================  ==========================  ====================
gradient source     ``PHYSICAL``                ``CPU``
write mechanism     ``PHYSICAL``                ``ELECTRONIC``
==================  ==========================  ====================
"""
from __future__ import annotations

from collections import Counter

# Canonical, substrate-agnostic axis poles. Use them where they fit; a substrate
# is free to use finer-grained value strings (04 uses "optical"/"physical-medium").
PHYSICAL = "physical"      # gradient computed by — or weight written by — the medium's own physics
CPU = "cpu"                # gradient computed off-substrate (offline / adjoint on a CPU)
ELECTRONIC = "electronic"  # weight written by an external electronic controller


class Audit:
    """Two-axis provenance recorder: ``(gradient_source, write_mechanism)``.

    Counts every recorded update keyed by the pair. With no vocab supplied the
    axes accept any non-empty string; supply a vocab set per axis to reject typos.
    """

    def __init__(self, gradient_vocab=None, write_mechanism_vocab=None):
        self._gvocab = None if gradient_vocab is None else set(gradient_vocab)
        self._wvocab = None if write_mechanism_vocab is None else set(write_mechanism_vocab)
        self._counts = Counter()

    @staticmethod
    def _check(value, vocab, axis):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{axis} must be a non-empty string, got {value!r}")
        if vocab is not None and value not in vocab:
            raise ValueError(f"bad {axis} {value!r}; allowed {sorted(vocab)}")

    def record(self, gradient_source, write_mechanism, n_updates=1):
        """Record ``n_updates`` updates with the given gradient source and write mechanism."""
        self._check(gradient_source, self._gvocab, "gradient_source")
        self._check(write_mechanism, self._wvocab, "write_mechanism")
        self._counts[(gradient_source, write_mechanism)] += int(n_updates)

    def breakdown(self):
        """Return ``{(gradient_source, write_mechanism): count}`` for the run summary / JSON."""
        return dict(self._counts)

    def total(self):
        """Return the total number of recorded updates."""
        return sum(self._counts.values())

    def is_pure(self, gradient_source, write_mechanism):
        """True iff every recorded update has exactly the given provenance (and there is at least one)."""
        return (self.total() > 0 and
                set(self._counts) == {(gradient_source, write_mechanism)})

    def is_physical(self, physical_values=(PHYSICAL,)):
        """True iff every recorded update is physical on BOTH axes — the substrate-agnostic PASS gate.

        ``physical_values`` lists the axis-value strings that count as physical;
        override it when a substrate names its physical poles differently
        (e.g. 04: ``physical_values=("optical", "physical-medium")``).
        """
        pv = set(physical_values)
        return (self.total() > 0 and
                all(g in pv and w in pv for (g, w) in self._counts))

    def assert_provenance(self, expected_gradient, expected_write):
        """Raise AssertionError unless every update is exactly ``(expected_gradient, expected_write)``."""
        bad = {k: v for k, v in self._counts.items()
               if k != (expected_gradient, expected_write)}
        if bad or self.total() == 0:
            raise AssertionError(
                f"expected only ({expected_gradient},{expected_write}); "
                f"got {self.breakdown()}")


# --- Legacy one-axis adapter (promoted from 01-coupled-learning-networks, 2026-06-20) -----------
# RECONCILE(00-agent): 01's coupled_learning / mechanical_learning / baselines still call this
# one-axis (gradient-source-only) API. It is kept here as a THIN ADAPTER over the two-axis Audit so
# the Tier-1 callers and their tests stay green and so the one-axis -> two-axis generalization is
# explicit in code. Your curation call: migrate those three callers to Audit.record(grad, write, n)
# and retire Auditor, or keep this shim long-term. (The legacy labels project onto the grid as
# PHYSICS_LOCAL -> (physical, physical); OFFLINE_GRADIENT -> (cpu, electronic).)
PHYSICS_LOCAL = "physics-local"
OFFLINE_GRADIENT = "offline-gradient"

_LEGACY_TO_TWO_AXIS = {
    PHYSICS_LOCAL: (PHYSICAL, PHYSICAL),
    OFFLINE_GRADIENT: (CPU, ELECTRONIC),
}
_TWO_AXIS_TO_LEGACY = {v: k for k, v in _LEGACY_TO_TWO_AXIS.items()}


class Auditor:
    """One-axis (gradient-source) recorder — the legacy 01 API, backed by two-axis ``Audit``.

    Tags each update with a single ``source`` label (``PHYSICS_LOCAL`` / ``OFFLINE_GRADIENT``),
    which is projected onto the canonical two-axis grid internally.
    """

    def __init__(self):
        self._audit = Audit()

    def record(self, source, n_updates=1):
        """Tag ``n_updates`` weight changes with ``source`` (``PHYSICS_LOCAL`` / ``OFFLINE_GRADIENT``)."""
        if source not in _LEGACY_TO_TWO_AXIS:
            raise ValueError(f"bad source {source!r}; allowed {sorted(_LEGACY_TO_TWO_AXIS)}")
        g, w = _LEGACY_TO_TWO_AXIS[source]
        self._audit.record(g, w, n_updates)

    def breakdown(self):
        """Return ``{source: count}`` (one-axis labels) for the run summary / JSON."""
        return {_TWO_AXIS_TO_LEGACY[k]: c for k, c in self._audit.breakdown().items()}

    def assert_all_physics_local(self):
        """Raise if any update was not physics-local (the headline-run gate)."""
        bad = {s: c for s, c in self.breakdown().items() if s != PHYSICS_LOCAL}
        if bad:
            raise AssertionError(f"non-physics-local updates: {bad}")
