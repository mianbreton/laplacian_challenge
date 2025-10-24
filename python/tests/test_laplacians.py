import pytest
import numpy as np
import numba_impl.laplacian as nl
import taichi_impl.laplacian as tl
import numpy_impl.laplacian as npl
from enum import Enum
import taichi as ti
from test_utils import check_interior, check_exterior

class Kernel(Enum):
    NUMBA_INTERIOR_ONLY = 0
    NUMBA_MODULO = 1
    NUMBA_CONDITIONAL_ADD = 2
    NUMBA_TERNARY = 3
    NUMBA_NEGATIVE_WRAP = 4
    NUMBA_DOMAINS = 5
    TAICHI_LOOPS_MODULO = 6
    TAICHI_NDRANGE_MODULO = 7
    TAICHI_NDRANGE_DOMAIN = 8
    TAICHI_GROUPED_MODULO = 9
    TAICHI_GROUPED_DOMAIN = 10
    NUMPY_BRUTE_FORCE_VEC = 11

@pytest.fixture(scope="module")
def setup_taichi():
    ti.init(arch=ti.cpu)

@pytest.fixture(scope="module")
def setup_gpu():
    try:
        ti.init(arch=ti.gpu)
        return True
    except Exception:
        return False

@pytest.fixture
def sample_data():
    N = 32
    x = np.empty((N, N, N), dtype=np.float32)
    out = np.empty_like(x)
    nl.initialise(x)
    return x, out, N

@pytest.mark.parametrize("kernel", list(Kernel))
def test_kernel_cpu(kernel, sample_data, setup_taichi):
    x, out, N = sample_data
    kernel_map = {
        Kernel.NUMBA_INTERIOR_ONLY: nl.interior_only,
        Kernel.NUMBA_MODULO: nl.modulo,
        Kernel.NUMBA_CONDITIONAL_ADD: nl.conditional_add,
        Kernel.NUMBA_TERNARY: nl.ternary,
        Kernel.NUMBA_NEGATIVE_WRAP: nl.negative_wrap,
        Kernel.NUMBA_DOMAINS: nl.domains,
        Kernel.TAICHI_LOOPS_MODULO: tl.loops_modulo,
        Kernel.TAICHI_NDRANGE_MODULO: tl.ndrange_modulo,
        Kernel.TAICHI_NDRANGE_DOMAIN: tl.ndrange_domain,
        Kernel.TAICHI_GROUPED_MODULO: tl.grouped_modulo,
        Kernel.TAICHI_GROUPED_DOMAIN: tl.grouped_domain,
        Kernel.NUMPY_BRUTE_FORCE_VEC: npl.brute_force_vectorize,
    }
    func = kernel_map[kernel]

    if "NUMBA" in kernel.name or "NUMPY" in kernel.name:
        func(out, x)
        check_interior(out)
        if not "INTERIOR" in kernel.name:
            check_exterior(out)
    elif "TAICHI" in kernel.name:
        x_ti = ti.field(dtype=ti.f32, shape=(N, N, N))
        out_ti = ti.field(dtype=ti.f32, shape=(N, N, N))
        x_ti.from_numpy(x)
        func(out_ti, x_ti, N)
        out = out_ti.to_numpy()
        check_interior(out)
        if not "INTERIOR" in kernel.name:
            check_exterior(out)

@pytest.mark.parametrize("kernel", [k for k in Kernel if "TAICHI" in k.name])
def test_kernel_gpu(kernel, sample_data, setup_gpu):
    if not setup_gpu:
        pytest.skip("GPU not available")
    x, out, N = sample_data
    kernel_map = {
        Kernel.TAICHI_LOOPS_MODULO: tl.loops_modulo,
        Kernel.TAICHI_NDRANGE_MODULO: tl.ndrange_modulo,
        Kernel.TAICHI_NDRANGE_DOMAIN: tl.ndrange_domain,
        Kernel.TAICHI_GROUPED_MODULO: tl.grouped_modulo,
        Kernel.TAICHI_GROUPED_DOMAIN: tl.grouped_domain,
    }
    func = kernel_map[kernel]
    x_ti = ti.field(dtype=ti.f32, shape=(N, N, N))
    out_ti = ti.field(dtype=ti.f32, shape=(N, N, N))
    x_ti.from_numpy(x)
    func(out_ti, x_ti, N)
    out = out_ti.to_numpy()
    check_interior(out)
    if not "INTERIOR" in kernel.name:
        check_exterior(out)
