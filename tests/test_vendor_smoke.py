"""Smoke-test the vendored 02 sim: it imports, runs a tiny noisy training for each rule,
and carries the expected (rule -> audit) signatures."""
from crossbar import build_full_diff, make_read_nonlinearity, make_cocontent
from task import parity3_data
from memristor import DeviceConfig
from readnoise import train_manhattan_noisy, train_coupled_noisy


def _setup():
    topo = build_full_diff(8)
    X, y = parity3_data()
    phi, dphi = make_read_nonlinearity(0.4)
    s_fn = make_cocontent(0.4)
    return topo, X, y, phi, dphi, s_fn


def test_vendored_manhattan_runs_and_audits_half():
    topo, X, y, phi, dphi, s_fn = _setup()
    res = train_manhattan_noisy(topo, X, y, phi, dphi, 1.5, DeviceConfig(),
                                n_epochs=8, seed=0, train_sigma=0.01)
    assert set(res) >= {"g", "separation", "audit"}
    assert res["audit"]["error_site"] == "off_array_cpu"
    assert res["audit"]["increment_site"] == "physical_pulse"


def test_vendored_coupled_runs_and_audits_full():
    topo, X, y, phi, dphi, s_fn = _setup()
    res = train_coupled_noisy(topo, X, y, phi, dphi, 1.5, DeviceConfig(),
                              n_epochs=8, seed=0, beta=0.1, s_fn=s_fn, train_sigma=0.01)
    assert res["audit"]["error_site"] == "on_array_local"
    assert res["audit"]["increment_site"] == "physical_pulse"
