"""Read-averaging model: the K-averaged read has std ~ sigma/sqrt(K) (validates the effective-sigma
equivalence the transfer sweep relies on)."""
import numpy as np

from memristor import DeviceConfig
from readnoise import noisy_read
from fg_rules import effective_sigma, averaged_noisy_read, writes_per_cell


def test_effective_sigma_scales_as_one_over_sqrt_k():
    assert effective_sigma(0.01, 1) == 0.01
    assert abs(effective_sigma(0.01, 100) - 0.001) < 1e-12


def test_averaged_read_std_matches_effective_sigma():
    # Many independent averaged reads of a mid-range conductance (clipping negligible there):
    # the empirical std should track sigma/sqrt(K).
    cfg = DeviceConfig()
    rng = np.random.default_rng(0)
    g = np.full(2000, 0.5 * (cfg.g_min + cfg.g_max))   # mid-range, away from bounds
    sigma, K = 0.05, 16
    reads = np.array([averaged_noisy_read(g, sigma, cfg, rng, K)[0] for _ in range(4000)])
    span = cfg.g_max - cfg.g_min
    expected = effective_sigma(sigma, K) * span
    assert abs(reads.std() - expected) / expected < 0.1   # within 10%


def test_averaged_read_k1_is_a_single_read():
    cfg = DeviceConfig()
    g = np.full(5, 0.5 * (cfg.g_min + cfg.g_max))
    a = averaged_noisy_read(g, 0.02, cfg, np.random.default_rng(1), 1)
    b = np.asarray(noisy_read(g, 0.02, cfg, np.random.default_rng(1)))
    assert np.allclose(a, b)
    assert isinstance(a, np.ndarray)


def test_writes_per_cell_is_n_epochs():
    assert writes_per_cell(300) == 300
    assert isinstance(writes_per_cell(300), int)
