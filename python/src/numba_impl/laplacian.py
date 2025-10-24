import numpy as np
import numpy.typing as npt
from numba import config, njit, prange

@njit(["void(f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def initialise(x: npt.NDArray[np.float32]) -> None:
    x_ravel = x.ravel()
    for i in prange(len(x_ravel)):
        x_ravel[i] = i

@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def interior_only(out: npt.NDArray[np.float32], x: npt.NDArray[np.float32]) -> None:
    N = x.shape[0]
    invh2 = np.float32(N**2)
    six = np.float32(6)
    for i in prange(1, N - 1):
        for j in range(1, N - 1):
            for k in range(1, N - 1):
                out[i, j, k] = (
                      x[i, j, k-1]
                    + x[i, j, k+1]                    
                    + x[i, j-1, k]
                    + x[i, j+1, k]
                    + x[i-1, j, k]
                    + x[i+1, j, k]
                    - six * x[i, j, k]
                ) * invh2

@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def modulo(out: npt.NDArray[np.float32], x: npt.NDArray[np.float32]) -> None:
    N = x.shape[0]
    invh2 = np.float32(N**2)
    six = np.float32(6)
    for i in prange(N):
        im1 = (i - 1) % N
        ip1 = (i + 1) % N
        for j in range(N):
            jm1 = (j - 1) % N
            jp1 = (j + 1) % N
            for k in range(N):
                km1 = (k - 1) % N
                kp1 = (k + 1) % N
                out[i, j, k] = (
                      x[i, j, km1]
                    + x[i, j, kp1]                    
                    + x[i, jm1, k]
                    + x[i, jp1, k]
                    + x[im1, j, k]
                    + x[ip1, j, k]
                    - six * x[i, j, k]
                ) * invh2

@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def conditional_add(out: npt.NDArray[np.float32], x: npt.NDArray[np.float32]) -> None:
    N = x.shape[0]
    Nm1 = N - 1
    invh2 = np.float32(N**2)
    six = np.float32(6)
    for i in prange(N):
        im1 = (i - 1) + N * (i==0)
        ip1 = (i + 1) - N * (i==Nm1)
        for j in range(N):
            jm1 = (j - 1) + N * (j==0)
            jp1 = (j + 1) - N * (j==Nm1)
            for k in range(N):
                km1 = (k - 1) + N * (k==0)
                kp1 = (k + 1) - N * (k==Nm1)
                out[i, j, k] = (
                      x[i, j, km1]
                    + x[i, j, kp1]                    
                    + x[i, jm1, k]
                    + x[i, jp1, k]
                    + x[im1, j, k]
                    + x[ip1, j, k]
                    - six * x[i, j, k]
                ) * invh2

@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def ternary(out: npt.NDArray[np.float32], x: npt.NDArray[np.float32]) -> None:
    N = x.shape[0]
    Nm1 = N - 1
    invh2 = np.float32(N**2)
    six = np.float32(6)
    for i in prange(N):
        im1 = Nm1 if i == 0 else i - 1
        ip1 = 0 if i == Nm1 else i + 1
        for j in range(N):
            jm1 = Nm1 if j == 0 else j - 1
            jp1 = 0 if j == Nm1 else j + 1
            for k in range(N):
                km1 = Nm1 if k == 0 else k - 1
                kp1 = 0 if k == Nm1 else k + 1
                out[i, j, k] = (
                      x[i, j, km1]
                    + x[i, j, kp1]                    
                    + x[i, jm1, k]
                    + x[i, jp1, k]
                    + x[im1, j, k]
                    + x[ip1, j, k]
                    - six * x[i, j, k]
                ) * invh2

@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def negative_wrap(out: npt.NDArray[np.float32], x: npt.NDArray[np.float32]) -> None:
    N = x.shape[0]
    invh2 = np.float32(N**2)
    six = np.float32(6)
    for i in prange(-1, N - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in range(-1, N - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in range(-1, N - 1):
                km1 = k - 1
                kp1 = k + 1
                out[i, j, k] = (
                      x[i, j, km1]
                    + x[i, j, kp1]                    
                    + x[i, jm1, k]
                    + x[i, jp1, k]
                    + x[im1, j, k]
                    + x[ip1, j, k]
                    - six * x[i, j, k]
                ) * invh2


@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=False, parallel=True)
def domains(out, x):
    N = out.shape[0] 
    invh2 = np.float32(N**2)
    six = np.float32(6)
    # Interior
    for i in prange(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                out[i, j, k] = (
                      x[i, j, k-1]
                    + x[i, j, k+1]                    
                    + x[i, j-1, k]
                    + x[i, j+1, k]
                    + x[i-1, j, k]
                    + x[i+1, j, k]
                    - six * x[i, j, k]
                ) * invh2    
    # Faces
    for i in prange(1, N-1):
        for j in range(-1,1):
            for k in range(1, N-1):
                out[i, j, k] = (
                      x[i, j, k-1]
                    + x[i, j, k+1]                    
                    + x[i, j-1, k]
                    + x[i, j+1, k]
                    + x[i-1, j, k]
                    + x[i+1, j, k]
                    - six * x[i, j, k]
                ) * invh2
    for i in range(-1,1):
        for j in prange(1, N-1):
            for k in range(1, N-1):
                out[i, j, k] = (
                      x[i, j, k-1]
                    + x[i, j, k+1]                    
                    + x[i, j-1, k]
                    + x[i, j+1, k]
                    + x[i-1, j, k]
                    + x[i+1, j, k]
                    - six * x[i, j, k]
                ) * invh2
    for i in prange(1, N-1):
        for j in range(1, N-1):
            for k in range(-1,1):
                out[i, j, k] = (
                      x[i, j, k-1]
                    + x[i, j, k+1]                    
                    + x[i, j-1, k]
                    + x[i, j+1, k]
                    + x[i-1, j, k]
                    + x[i+1, j, k]
                    - six * x[i, j, k]
                ) * invh2
    # Edges
    for i in prange(1, N-1):
        for j in range(-1,1):
            for k in range(-1,1):
                out[i, j, k] = (
                      x[i, j, k-1]
                    + x[i, j, k+1]                    
                    + x[i, j-1, k]
                    + x[i, j+1, k]
                    + x[i-1, j, k]
                    + x[i+1, j, k]
                    - six * x[i, j, k]
                ) * invh2
    for i in range(-1,1):
        for j in prange(1, N-1):
            for k in range(-1,1):
                out[i, j, k] = (
                      x[i, j, k-1]
                    + x[i, j, k+1]                    
                    + x[i, j-1, k]
                    + x[i, j+1, k]
                    + x[i-1, j, k]
                    + x[i+1, j, k]
                    - six * x[i, j, k]
                ) * invh2
    for i in range(-1,1):
        for j in range(-1,1):
            for k in prange(1, N-1):
                out[i, j, k] = (
                      x[i, j, k-1]
                    + x[i, j, k+1]                    
                    + x[i, j-1, k]
                    + x[i, j+1, k]
                    + x[i-1, j, k]
                    + x[i+1, j, k]
                    - six * x[i, j, k]
                ) * invh2
    
    # Corners
    for i in range(-1,1):
        for j in range(-1,1):
            for k in range(-1,1):
                out[i, j, k] = (
                      x[i, j, k-1]
                    + x[i, j, k+1]                    
                    + x[i, j-1, k]
                    + x[i, j+1, k]
                    + x[i-1, j, k]
                    + x[i+1, j, k]
                    - six * x[i, j, k]
                ) * invh2
