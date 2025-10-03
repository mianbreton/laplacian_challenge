import taichi as ti
import numpy as np
import numba
import time

@ti.func
def laplacian(out: ti.template(), x: ti.template(), i, im1, ip1, j, jm1, jp1, k, km1, kp1, six, invh2):
    out[i,j,k] = (
                      x[i, j, km1]
                    + x[i, j, kp1]                    
                    + x[i, jm1, k]
                    + x[i, jp1, k]
                    + x[im1, j, k]
                    + x[ip1, j, k]
                    - six * x[i, j, k]
                ) * invh2

@ti.kernel
def loops_modulo(out: ti.template(), x: ti.template(), N: ti.i32):
    invh2 = ti.f32(N * N)
    six = ti.f32(6.0)
    for i in range(N):
        im1 = (i - 1) % N
        ip1 = (i + 1) % N
        for j in range(N):
            jm1 = (j - 1) % N
            jp1 = (j + 1) % N
            for k in range(N):
                km1 = (k - 1) % N
                kp1 = (k + 1) % N
                laplacian(out, x, i, im1, ip1, j, jm1, jp1, k, km1, kp1, six, invh2)


@ti.kernel
def ndrange_modulo(out: ti.template(), x: ti.template(), N: ti.i32):
    invh2 = ti.f32(N * N)
    six = ti.f32(6.0)
    for i, j, k in ti.ndrange(N, N, N):
        im1 = (i - 1) % N
        ip1 = (i + 1) % N
        jm1 = (j - 1) % N
        jp1 = (j + 1) % N
        km1 = (k - 1) % N
        kp1 = (k + 1) % N
        laplacian(out, x, i, im1, ip1, j, jm1, jp1, k, km1, kp1, six, invh2)


@ti.kernel
def ndrange_domain(out: ti.template(), x: ti.template(), N: ti.i32):
    invh2 = ti.f32(N * N)
    six = ti.f32(6.0)
    # Interior
    for i, j, k in ti.ndrange((1, N - 1), (1, N - 1), (1, N - 1)):
        laplacian(out, x, i, i-1, i+1, j, j-1, j+1, k, k-1, k+1, six, invh2)

    # Faces
    for i, j in ti.ndrange((1, N - 1), (1, N - 1)):
        laplacian(out, x, i, i-1, i+1, j, j-1, j+1, N-1, N-2, 0, six, invh2)
        laplacian(out, x, i, i-1, i+1, j, j-1, j+1, 0  , N-1, 1, six, invh2)

    for i, k in ti.ndrange((1, N - 1), (1, N - 1)):
        laplacian(out, x, i, i-1, i+1, N-1, N-2, 0, k, k-1, k+1, six, invh2)
        laplacian(out, x, i, i-1, i+1, 0,   N-1, 1, k, k-1, k+1, six, invh2)

    for j, k in ti.ndrange((1, N - 1), (1, N - 1)):
        laplacian(out, x, N-1, N-2, 0, j, j-1, j+1, k, k-1, k+1, six, invh2)
        laplacian(out, x, 0  , N-1, 1, j, j-1, j+1, k, k-1, k+1, six, invh2)

    # Edges
    for k in range(1, N - 1):
        laplacian(out, x, N-1, N-2, 0  , N-1, N-2, 0  , k, k-1, k+1, six, invh2)
        laplacian(out, x, N-1, N-2, 0  , 0  , N-1, 1  , k, k-1, k+1, six, invh2)
        laplacian(out, x, 0  , N-1, 1  , N-1, N-2, 0  , k, k-1, k+1, six, invh2)
        laplacian(out, x, 0  , N-1, 1  , 0  , N-1, 1  , k, k-1, k+1, six, invh2)        

    for j in range(1, N - 1):
        laplacian(out, x, N-1, N-2, 0  , j, j-1, j+1, N-1, N-2, 0  , six, invh2)
        laplacian(out, x, N-1, N-2, 0  , j, j-1, j+1, 0  , N-1, 1  , six, invh2)
        laplacian(out, x, 0  , N-1, 1  , j, j-1, j+1, N-1, N-2, 0  , six, invh2)
        laplacian(out, x, 0  , N-1, 1  , j, j-1, j+1, 0  , N-1, 1  , six, invh2)  

    for i in range(1, N - 1):
        laplacian(out, x, i, i-1, i+1, N-1, N-2, 0  , N-1, N-2, 0  , six, invh2)
        laplacian(out, x, i, i-1, i+1, N-1, N-2, 0  , 0  , N-1, 1  , six, invh2)
        laplacian(out, x, i, i-1, i+1, 0  , N-1, 1  , N-1, N-2, 0  , six, invh2)
        laplacian(out, x, i, i-1, i+1, 0  , N-1, 1  , 0  , N-1, 1  , six, invh2)  

    # Corners
    laplacian(out, x, N-1, N-2, 0, N-1, N-2, 0  , N-1, N-2, 0  , six, invh2)
    laplacian(out, x, N-1, N-2, 0, N-1, N-2, 0  , 0  , N-1, 1  , six, invh2)
    laplacian(out, x, N-1, N-2, 0, 0  , N-1, 1  , N-1, N-2, 0  , six, invh2)
    laplacian(out, x, N-1, N-2, 0, 0  , N-1, 1  , 0  , N-1, 1  , six, invh2) 
    laplacian(out, x, 0  , N-1, 1, N-1, N-2, 0  , N-1, N-2, 0  , six, invh2)
    laplacian(out, x, 0  , N-1, 1, N-1, N-2, 0  , 0  , N-1, 1  , six, invh2)
    laplacian(out, x, 0  , N-1, 1, 0  , N-1, 1  , N-1, N-2, 0  , six, invh2)
    laplacian(out, x, 0  , N-1, 1, 0  , N-1, 1  , 0  , N-1, 1  , six, invh2) 
   
@ti.func
def wrap(i, N):
    return i % N

@ti.kernel
def grouped(out: ti.template(), x: ti.template(), N: ti.i32):
    invh2 = ti.f32(N * N)
    six = ti.f32(6.0)
    zero = ti.f32(0.0)
    neighbours = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for I in ti.grouped(x):
        if 1 <= I[0] < N - 1 and 1 <= I[1] < N - 1 and 1 <= I[2] < N - 1:
            res = zero
            for stencil in ti.static(neighbours):
                res += x[I + stencil]
            out[I] = invh2 * (res - six * x[I])

@ti.kernel
def grouped_modulo(out: ti.template(), x: ti.template(), N: ti.i32):
    invh2 = ti.f32(N * N)
    six = ti.f32(6.0)
    zero = ti.f32(0.0)
    neighbours = [ti.Vector([1, 0, 0]), ti.Vector([-1, 0, 0]),
                  ti.Vector([0, 1, 0]), ti.Vector([0, -1, 0]),
                  ti.Vector([0, 0, 1]), ti.Vector([0, 0, -1])]
    
    for I in ti.grouped(x):
        res = zero
        for offset in ti.static(neighbours):
            J = ti.Vector([
                wrap(I[0] + offset[0], N),
                wrap(I[1] + offset[1], N),
                wrap(I[2] + offset[2], N),
            ])
            res += x[J]
        out[I] = invh2 * (res - six * x[I])

@ti.func
def laplacian_grouped_domain(out: ti.template(), x: ti.template(), I, neighbours, zero, six, invh2):
    res = zero
    for stencil in ti.static(neighbours):
        res += x[I + stencil]
    out[I] = invh2 * (res - six * x[I])


@ti.kernel
def grouped_domain(out: ti.template(), x: ti.template(), N: ti.i32):
    invh2 = ti.f32(N * N)
    six = ti.f32(6.0)
    zero = ti.f32(0.0)
    # Interior
    neighbours = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    # Faces
    neighbours_i0 = [(N-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    neighbours_iN = [(-1, 0, 0), (-N+1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    neighbours_j0 = [(-1, 0, 0), (1, 0, 0), (0, N-1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    neighbours_jN = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, -N+1, 0), (0, 0, -1), (0, 0, 1)]
    neighbours_k0 = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, N-1), (0, 0, 1)]
    neighbours_kN = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, -N+1)]
    # Edges
    neighbours_i0j0 = [(N-1, 0, 0), (1, 0, 0), (0, N-1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    neighbours_i0k0 = [(N-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, N-1), (0, 0, 1)]
    neighbours_i0jN = [(N-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, -N+1, 0), (0, 0, -1), (0, 0, +1)]
    neighbours_i0kN = [(N-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, -N+1)]
    neighbours_iNj0 = [(-1, 0, 0), (-N+1, 0, 0), (0, N-1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    neighbours_iNk0 = [(-1, 0, 0), (-N+1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, N-1), (0, 0, 1)]
    neighbours_iNjN = [(-1, 0, 0), (-N+1, 0, 0), (0, -1, 0), (0, -N+1, 0), (0, 0, -1), (0, 0, 1)]
    neighbours_iNkN = [(-1, 0, 0), (-N+1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, -N+1)]
    neighbours_j0k0 = [(-1, 0, 0), (1, 0, 0), (0, N-1, 0), (0, 1, 0), (0, 0, N-1), (0, 0, 1)]
    neighbours_j0kN = [(-1, 0, 0), (1, 0, 0), (0, N-1, 0), (0, 1, 0), (0, 0, -1), (0, 0, -N+1)]
    neighbours_jNk0 = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, -N+1, 0), (0, 0, N-1), (0, 0, 1)]
    neighbours_jNkN = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, -N+1, 0), (0, 0, -1), (0, 0, -N+1)]
    # Corners
    neighbours_i0j0k0 = [(N-1, 0, 0), (1, 0, 0), (0, N-1, 0), (0, 1, 0), (0, 0, N-1), (0, 0, 1)]
    neighbours_i0j0kN = [(N-1, 0, 0), (1, 0, 0), (0, N-1, 0), (0, 1, 0), (0, 0, -1), (0, 0, -N+1)]
    neighbours_i0jNk0 = [(N-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, -N+1, 0), (0, 0, N-1), (0, 0, 1)]
    neighbours_i0jNkN = [(N-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, -N+1, 0), (0, 0, -1), (0, 0, -N+1)]
    neighbours_iNj0k0 = [(-1, 0, 0), (-N+1, 0, 0), (0, N-1, 0), (0, 1, 0), (0, 0, N-1), (0, 0, 1)]
    neighbours_iNj0kN = [(-1, 0, 0), (-N+1, 0, 0), (0, N-1, 0), (0, 1, 0), (0, 0, -1), (0, 0, -N+1)]
    neighbours_iNjNk0 = [(-1, 0, 0), (-N+1, 0, 0), (0, -1, 0), (0, -N+1, 0), (0, 0, N-1), (0, 0, 1)]
    neighbours_iNjNkN = [(-1, 0, 0), (-N+1, 0, 0), (0, -1, 0), (0, -N+1, 0), (0, 0, -1), (0, 0, -N+1)]

    for I in ti.grouped(x):
        # Interior
        if 1 <= I[0] < N - 1 and 1 <= I[1] < N - 1 and 1 <= I[2] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours, zero, six, invh2)
        # Faces
        if I[0] == 0 and 1 <= I[1] < N - 1 and 1 <= I[2] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_i0, zero, six, invh2)
        if I[0] == N-1 and 1 <= I[1] < N - 1 and 1 <= I[2] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_iN, zero, six, invh2)
        if I[1] == 0 and 1 <= I[0] < N - 1 and 1 <= I[2] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_j0, zero, six, invh2)
        if I[1] == N-1 and 1 <= I[0] < N - 1 and 1 <= I[2] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_jN, zero, six, invh2)
        if I[2] == 0 and 1 <= I[0] < N - 1 and 1 <= I[1] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_k0, zero, six, invh2)
        if I[2] == N-1 and 1 <= I[0] < N - 1 and 1 <= I[1] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_kN, zero, six, invh2)
        # Edges
        if I[0] == 0 and I[1] == 0 and 1 <= I[2] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_i0j0, zero, six, invh2)
        if I[0] == 0 and I[2] == 0 and 1 <= I[1] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_i0k0, zero, six, invh2)
        if I[1] == 0 and I[2] == 0 and 1 <= I[0] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_j0k0, zero, six, invh2)
        if I[0] == 0 and I[1] == N - 1 and 1 <= I[2] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_i0jN, zero, six, invh2)
        if I[0] == 0 and I[2] == N - 1 and 1 <= I[1] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_i0kN, zero, six, invh2)
        if I[1] == 0 and I[2] == N - 1 and 1 <= I[0] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_j0kN, zero, six, invh2)
        if I[0] == N - 1 and I[1] == 0 and 1 <= I[2] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_iNj0, zero, six, invh2)
        if I[0] == N - 1 and I[2] == 0 and 1 <= I[1] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_iNk0, zero, six, invh2)
        if I[1] == N - 1 and I[2] == 0 and 1 <= I[0] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_jNk0, zero, six, invh2)
        if I[0] == N - 1 and I[1] == N - 1 and 1 <= I[2] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_iNjN, zero, six, invh2)
        if I[0] == N - 1 and I[2] == N - 1 and 1 <= I[1] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_iNkN, zero, six, invh2)
        if I[1] == N - 1 and I[2] == N - 1 and 1 <= I[0] < N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_jNkN, zero, six, invh2)
        # Corners
        if I[0] == 0 and I[1] == 0 and I[2] == 0:
            laplacian_grouped_domain(out, x, I, neighbours_i0j0k0, zero, six, invh2)
        if I[0] == 0 and I[1] == 0 and I[2] == N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_i0j0kN, zero, six, invh2)
        if I[0] == 0 and I[1] == N - 1 and I[2] == 0:
            laplacian_grouped_domain(out, x, I, neighbours_i0jNk0, zero, six, invh2)
        if I[0] == 0 and I[1] == N - 1 and I[2] == N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_i0jNkN, zero, six, invh2)
        if I[0] == N - 1 and I[1] == 0 and I[2] == 0:
            laplacian_grouped_domain(out, x, I, neighbours_iNj0k0, zero, six, invh2)
        if I[0] == N - 1 and I[1] == 0 and I[2] == N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_iNj0kN, zero, six, invh2)
        if I[0] == N - 1 and I[1] == N - 1 and I[2] == 0:
            laplacian_grouped_domain(out, x, I, neighbours_iNjNk0, zero, six, invh2)
        if I[0] == N - 1 and I[1] == N - 1 and I[2] == N - 1:
            laplacian_grouped_domain(out, x, I, neighbours_iNjNkN, zero, six, invh2)
        

        




