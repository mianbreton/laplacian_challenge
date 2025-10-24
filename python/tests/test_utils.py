import numpy as np
from numba import config, njit, prange

# Interior check
@njit(["void(f4[:,:,::1])"], fastmath=True, cache=True)
def check_interior(x: np.ndarray) -> None:
    N = x.shape[0]
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for k in range(1, N - 1):
                if x[i, j, k] != 0.0:
                    raise ValueError("check_interior failed")


# Exterior check
@njit(["void(f4[:,:,::1])"], fastmath=True, cache=True)
def check_exterior(x: np.ndarray) -> None:
    N = x.shape[0]
    N2 = N * N
    N3 = N * N2

    # Faces
    for i in range(1, N - 1):
        for k in range(1, N - 1):
            if x[i, -1, k] != -N2 * N2: raise ValueError("face error")
            if x[i, 0, k] != N2 * N2: raise ValueError("face error")

    for j in range(1, N - 1):
        for k in range(1, N - 1):
            if x[-1, j, k] != -N2 * N3: raise ValueError("face error")
            if x[0, j, k] != N2 * N3: raise ValueError("face error")

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            if x[i, j, -1] != -N2 * N: raise ValueError("face error")
            if x[i, j, 0] != N2 * N: raise ValueError("face error")

    # Edges
    for i in range(1, N - 1):
        if x[i, -1, -1] != N2 * (-N2 - N): raise ValueError("edge error")
        if x[i, -1, 0] != N2 * (-N2 + N): raise ValueError("edge error")
        if x[i, 0, -1] != N2 * (N2 - N): raise ValueError("edge error")
        if x[i, 0, 0] != N2 * (N2 + N): raise ValueError("edge error")

    for j in range(1, N - 1):
        if x[-1, j, -1] != N2 * (-N3 - N): raise ValueError("edge error")
        if x[-1, j, 0] != N2 * (-N3 + N): raise ValueError("edge error")
        if x[0, j, -1] != N2 * (N3 - N): raise ValueError("edge error")
        if x[0, j, 0] != N2 * (N3 + N): raise ValueError("edge error")

    for k in range(1, N - 1):
        if x[-1, -1, k] != N2 * (-N3 - N2): raise ValueError("edge error")
        if x[-1, 0, k] != N2 * (-N3 + N2): raise ValueError("edge error")
        if x[0, -1, k] != N2 * (N3 - N2): raise ValueError("edge error")
        if x[0, 0, k] != N2 * (N3 + N2): raise ValueError("edge error")

    # Corners
    if x[-1, -1, -1] != N2 * (-N3 - N2 - N): raise ValueError("corner error")
    if x[-1, -1, 0]  != N2 * (-N3 - N2 + N): raise ValueError("corner error")
    if x[-1, 0, -1]  != N2 * (-N3 + N2 - N): raise ValueError("corner error")
    if x[-1, 0, 0]   != N2 * (-N3 + N2 + N): raise ValueError("corner error")
    if x[0, -1, -1]  != N2 * (N3 - N2 - N):  raise ValueError("corner error")
    if x[0, -1, 0]   != N2 * (N3 - N2 + N):  raise ValueError("corner error")
    if x[0, 0, -1]   != N2 * (N3 + N2 - N):  raise ValueError("corner error")
    if x[0, 0, 0]    != N2 * (N3 + N2 + N):  raise ValueError("corner error")
