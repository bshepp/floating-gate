"""Read-averaging model for the robust-in-fabric candidate (objective 2), plus the endurance write-
count. 08-authored code composing the vendored 02 primitives; it NEVER edits vendor_02/.

Multi-read averaging: averaging K independent reads of a cell reduces the read-noise std by sqrt(K)
(the mean of K i.i.d. draws). So a K-averaged two-read contrastive update at read noise sigma is
distributionally the vendored two-read rule at an EFFECTIVE read noise sigma/sqrt(K) (exact up to the
clip at the conductance bounds, negligible near mid-range -- validated in tests). transfer_sweep uses
that equivalence to compute the robust SNR by running the vendored full rule at effective_sigma; this
module supplies effective_sigma + an averaged_noisy_read that validates the equivalence."""
import numpy as np

from readnoise import noisy_read   # vendored 02 (clipped, span-scaled, explicit-sigma read)


def effective_sigma(sigma, K):
    """K-read averaging reduces read-noise std by sqrt(K): the averaged read noise is sigma/sqrt(K)."""
    return float(sigma) / float(np.sqrt(K))


def averaged_noisy_read(g, sigma, cfg, rng, K):
    """Mean of K independent vendored noisy reads of g. The faithful multi-read read; its empirical
    std ~ sigma/sqrt(K) validates the effective_sigma equivalence the sweep relies on."""
    g = np.asarray(g, float)
    K = int(K)
    if K <= 1:
        return np.asarray(noisy_read(g, sigma, cfg, rng), float)
    acc = np.zeros_like(g)
    for _ in range(K):
        acc = acc + np.asarray(noisy_read(g, sigma, cfg, rng), float)
    return acc / float(K)


def writes_per_cell(n_epochs):
    """Worst-case program/erase pulses any single cell receives over a training run. The sign rules
    fire a pulse on (almost) every step where the per-edge gradient sign is nonzero, so the conservative
    upper bound is one write per cell per epoch = n_epochs. Conservative is correct for an endurance
    budget check (over-counting only tightens the margin)."""
    return int(n_epochs)
