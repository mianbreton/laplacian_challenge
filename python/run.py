import numpy as np
import numba_impl.laplacian as nl
import taichi_impl.laplacian as tl
import numpy_impl.laplacian as npl
from enum import Enum
import numba
import time
from rich import print
import taichi as ti
import argparse

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

def timing(kernel: Kernel, N, runs, NCPU):
    # Set up the input/output arrays
    x = np.empty((N, N, N), dtype=np.float32)
    out = np.empty_like(x)
    # Initialise input data
    nl.initialise(x)

    # Mapping enum to actual functions
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

    # Timing
    time_array = np.empty(runs)
    if "NUMBA" in kernel.name or "NUMPY" in kernel.name:
        # Warm up (especially important for Numba JIT)
        func(out, x)
        if N == 32:
            print("Checking correctness for N=32")
            nl.check_interior(out)
            if not "INTERIOR" in kernel.name:
                print("Check also boundaries")
                nl.check_exterior(out)
        for i in range(runs):
            start = time.time()
            func(out, x)
            time_array[i] = time.time() - start
    elif "TAICHI" in kernel.name:
        x_ti = ti.field(dtype=ti.f32, shape=(N,N,N))
        out_ti = ti.field(dtype=ti.f32, shape=(N,N,N))
        x_ti.from_numpy(x)
        x = 0
        out = 0
        # Warm up (especially important for Taichi JIT)
        func(out_ti, x_ti, N)
        if N == 32:
            out = out_ti.to_numpy()
            print("Checking correctness for N=32")
            nl.check_interior(out)
            if not "INTERIOR" in kernel.name:
                print("Check also boundaries")
                nl.check_exterior(out)
        for i in range(runs):
            start = time.time()
            func(out_ti, x_ti, N)
            ti.sync()
            time_array[i] = time.time() - start

    else:
        raise ValueError(f"{kernel.name=}, must contain 'NUMBA' or 'TAICHI' or 'NUMPY'")

    median = np.median(time_array)*1e3
    std = np.std(time_array)*1e3
    np.savetxt(f"timings/time_{kernel.name}_ncells1d_{N}_NCPU_{NCPU}.dat", time_array)
    print(f"{kernel.name}: Time: {median:.4f} (median) [+- {std:.4f}] milliseconds  ( {runs} runs )")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Laplacian benchmarks")
    parser.add_argument(
        "--ncells",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256],
        help="Number of cells along each dimension"
    )
    parser.add_argument(
        "--ncpus",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
        help="Number of CPU threads to test"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of repetitions for timing"
    )

    args = parser.parse_args()

    N_list = args.ncells
    NCPUs = args.ncpus
    runs = args.runs

    # CPU runs
    for NCPU in NCPUs:
        ti.init(arch=ti.cpu, cpu_max_num_threads=NCPU)
        numba.set_num_threads(NCPU)
        print(f"CPU runs with {NCPU} threads")
        for N in N_list:
            print("")
            print(f"cells along one direction: {N:{'-'}<{50}}")
            print("")
            for k in Kernel:
                timing(k, N, runs, NCPU)
    # GPU runs
    try:
        ti.init(arch=ti.gpu)
        GPU_AVAILABLE = True
    except Exception as e:
        print("No GPU available, skipping GPU runs:", e)
        GPU_AVAILABLE = False

    if GPU_AVAILABLE:
        print("")
        print("GPU runs")
        for N in N_list:
            print("")
            print(f"cells along one direction: {N:{'-'}<{50}}")
            print("")
            for k in Kernel:
                if "TAICHI" in k.name:
                    timing(k, N, runs, 0)


