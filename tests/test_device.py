"""Non-CPU devices (README §2.7). Tested only where the device exists: MPS on Apple Silicon has been run (see below), CUDA never has
(no CUDA hardware was available during development) -- its tests are skipped here and are the first thing to run on a machine with a GPU."""
import numpy as np
import pytest
import torch

from dcms.utils.wng import k_s_generator_pl

DEVICES = [
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")),
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")),
]


def _data(N=40, seed=42):
    k, s = k_s_generator_pl(N, rho=0.1, seed=seed)
    return [x.numpy().astype(float) for x in (k[:N], k[N:], s[:N], s[N:])]


def _models():
    from dcms.models.dcm import DCMModel
    from dcms.models.decm import DECMModel
    from dcms.models.dwcm import DWCMModel
    from dcms.models.qdecm import qDECMModel
    k_out, k_in, s_out, s_in = _data()
    return {
        "DCM": lambda: DCMModel(k_out, k_in),
        "DWCM": lambda: DWCMModel(s_out, s_in),
        "qDECM": lambda: qDECMModel(k_out, k_in, s_out, s_in),
        "DECM": lambda: DECMModel(k_out, k_in, s_out, s_in),
    }


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["DCM", "DWCM", "qDECM", "DECM"])
def test_solves_on_device_in_float32_and_returns_cpu_float64(name, device):
    model = _models()[name]()
    converged = model.solve_tool(tol=1e-3, max_iter=2000, device=device)
    assert converged
    assert model.sol.best_theta.dtype == np.float64          # SolverResult is always CPU/float64
    cpu = _models()[name]()
    cpu.solve_tool(tol=1e-3, max_iter=2000)
    assert np.allclose(model.sol.best_theta, cpu.sol.best_theta, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("name", ["DCM", "DWCM", "qDECM", "DECM"])
def test_non_cpu_device_requires_the_reduced_path(name):
    """No device is needed: the request itself must raise, not be ignored."""
    with pytest.raises(NotImplementedError):
        _models()[name]().solve_tool(tol=1e-4, device="mps", reduce_degeneracy=False)
