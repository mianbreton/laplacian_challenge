#pragma once

#include <Kokkos_Core.hpp>

class Laplacian {
    public:

    static void modulo_1d_flat (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void modulo_1d_mdrange (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void conditional_add_1d_flat (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void conditional_add_1d_mdrange (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void ternary_1d_flat (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void ternary_1d_mdrange (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void interior_1d_flat (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void interior_1d_mdrange (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void interior_1d_mdrange_i32_idx32 (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void interior_1d_mdrange_i32_idx64 (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void interior_1d_mdrange_i32_idx64promotion (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void interior_1d_mdrange_i64_idx32 (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N);
    static void interior_3d_flat (Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N);
    static void interior_3d_mdrange (Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N);
    static void modulo_3d_flat (Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N);
    static void modulo_3d_mdrange (Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N);
    static void conditional_add_3d_flat (Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N);
    static void conditional_add_3d_mdrange (Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N);
    static void ternary_3d_flat (Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N);
    static void ternary_3d_mdrange (Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N);
};

void Laplacian::modulo_1d_flat (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    const float invh2 = static_cast<float>(N2);
    Kokkos::parallel_for("Laplacian", N3, 
        KOKKOS_LAMBDA(const size_t idx)
    {
        const size_t i = idx / N2;
        const size_t iN2 = i * N2;
        const size_t j = (idx - iN2) / N;
        const size_t jN = j * N;
        const size_t k = idx - iN2 - jN;
        out(idx) =  ( x((iN2 + jN + (k-1)%N)) 
                    + x((iN2 + jN + (k+1)%N))
                    + x((iN2 + ((j-1)%N)*N + k)) 
                    + x((iN2 + ((j+1)%N)*N + k))
                    + x((((i-1)%N)*N2 + jN + k)) 
                    + x((((i+1)%N)*N2 + jN + k))
                    - 6.f * x(idx)
                    ) * invh2;
    });
}
void Laplacian::modulo_1d_mdrange(Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const size_t N2 = N * N;
    const float invh2 = static_cast<float>(N2);

    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        const size_t jN = j * N;
        const size_t iN2 = i * N2;
        const size_t idx = iN2 + jN + k;
        out(idx) =  ( x((iN2 + jN + (k-1)%N)) 
                    + x((iN2 + jN + (k+1)%N))
                    + x((iN2 + ((j-1)%N)*N + k)) 
                    + x((iN2 + ((j+1)%N)*N + k))
                    + x((((i-1)%N)*N2 + jN + k)) 
                    + x((((i+1)%N)*N2 + jN + k))
                    - 6.f * x(idx)
                    ) * invh2;
    });
}
void Laplacian::conditional_add_1d_flat (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const size_t Nm1 = N - 1;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    const float invh2 = static_cast<float>(N2);
    Kokkos::parallel_for("Laplacian", N3, 
        KOKKOS_LAMBDA(const size_t idx)
    {
        const size_t i = idx / N2;
        const size_t iN2 = i * N2;
        const size_t j = (idx - iN2) / N;
        const size_t jN = j * N;
        const size_t k = idx - iN2 - jN;

        const size_t im1 = (i - 1ul) + N*(i == 0ul);
        const size_t ip1 = (i + 1ul) - N*(i == Nm1);
        const size_t jm1 = (j - 1ul) + N*(j == 0ul);
        const size_t jp1 = (j + 1ul) - N*(j == Nm1);
        const size_t km1 = (k - 1ul) + N*(k == 0ul);
        const size_t kp1 = (k + 1ul) - N*(k == Nm1);

        out(idx) =  ( x((iN2 + jN + km1)) 
                    + x((iN2 + jN + kp1))
                    + x((iN2 + jm1*N + k)) 
                    + x((iN2 + jp1*N + k))
                    + x((im1*N2 + jN + k)) 
                    + x((ip1*N2 + jN + k))
                    - 6.f * x(idx)
                    ) * invh2;
    });
}
void Laplacian::conditional_add_1d_mdrange (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const size_t Nm1 = N - 1;
    const size_t N2 = N*N;
    const float invh2 = static_cast<float>(N2);
    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        const size_t iN2 = i * N2;
        const size_t jN = j * N;
        const size_t idx = iN2 + jN + k;
        const size_t im1 = (i - 1ul) + N*(i == 0ul);
        const size_t ip1 = (i + 1ul) - N*(i == Nm1);
        const size_t jm1 = (j - 1ul) + N*(j == 0ul);
        const size_t jp1 = (j + 1ul) - N*(j == Nm1);
        const size_t km1 = (k - 1ul) + N*(k == 0ul);
        const size_t kp1 = (k + 1ul) - N*(k == Nm1);

        out(idx) =  ( x((iN2 + jN + km1)) 
                    + x((iN2 + jN + kp1))
                    + x((iN2 + jm1*N + k)) 
                    + x((iN2 + jp1*N + k))
                    + x((im1*N2 + jN + k)) 
                    + x((ip1*N2 + jN + k))
                    - 6.f * x(idx)
                    ) * invh2;
    });
}
void Laplacian::ternary_1d_flat (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const size_t Nm1 = N - 1;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    const float invh2 = static_cast<float>(N2);
    Kokkos::parallel_for("Laplacian", N3, 
        KOKKOS_LAMBDA(const size_t idx)
    {
        const size_t i = idx / N2;
        const size_t iN2 = i * N2;
        const size_t j = (idx - iN2) / N;
        const size_t jN = j * N;
        const size_t k = idx - iN2 - jN;

        const size_t im1 = (i == 0ul) ? Nm1 : i - 1ul;
        const size_t ip1 = (i == Nm1) ? 0ul : i + 1ul;
        const size_t jm1 = (j  == 0ul) ? Nm1: j - 1ul;
        const size_t jp1 = (j  == Nm1) ? 0ul : j + 1ul;
        const size_t km1 = (k == 0ul) ? Nm1 : k - 1ul;
        const size_t kp1 = (k == Nm1) ? 0ul : k + 1ul;

        out(idx) =  ( x((iN2 + jN + km1)) 
                    + x((iN2 + jN + kp1))
                    + x((iN2 + jm1*N + k)) 
                    + x((iN2 + jp1*N + k))
                    + x((im1*N2 + jN + k)) 
                    + x((ip1*N2 + jN + k))
                    - 6.f * x(idx)
                    ) * invh2;
    });
}
void Laplacian::ternary_1d_mdrange (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const size_t Nm1 = N - 1;
    const size_t N2 = N*N;
    const float invh2 = static_cast<float>(N2);
    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        const size_t iN2 = i * N2;
        const size_t jN = j * N;
        const size_t idx = iN2 + jN + k;
        const size_t im1 = (i == 0ul) ? Nm1 : i - 1ul;
        const size_t ip1 = (i == Nm1) ? 0ul : i + 1ul;
        const size_t jm1 = (j  == 0ul) ? Nm1: j - 1ul;
        const size_t jp1 = (j  == Nm1) ? 0ul : j + 1ul;
        const size_t km1 = (k == 0ul) ? Nm1 : k - 1ul;
        const size_t kp1 = (k == Nm1) ? 0ul : k + 1ul;

        out(idx) =  ( x((iN2 + jN + km1)) 
                    + x((iN2 + jN + kp1))
                    + x((iN2 + jm1*N + k)) 
                    + x((iN2 + jp1*N + k))
                    + x((im1*N2 + jN + k)) 
                    + x((ip1*N2 + jN + k))
                    - 6.f * x(idx)
                    ) * invh2;
    });
}

void Laplacian::interior_1d_flat (Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const size_t Ni = N-2;
    const size_t N2i = Ni*Ni;
    const size_t N3i = N2i*Ni;
    const size_t N2 = N*N;
    const float invh2 = static_cast<float>(N2);
    Kokkos::parallel_for("Laplacian", N3i, 
        KOKKOS_LAMBDA(const size_t idx)
    {
        const size_t i = idx / N2i;
        const size_t j = (idx - i * N2i) / Ni;
        const size_t k = idx - i * N2i - j * Ni;
        const size_t idx2 = (i+1) * N2 + (j+1) * N + (k + 1);
        out(idx2) =  ( x(idx2-1) + x(idx2+1) + x(idx2-N)
                    + x(idx2+N) + x(idx2-N2)+ x(idx2+N2)
                    - 6.f * x(idx2)
                    ) * invh2;
    });
}

void Laplacian::interior_1d_mdrange(Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const size_t N2 = N * N;
    const float invh2 = static_cast<float>(N2);

    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {N-1, N-1, N-1}),
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        const size_t idx = i * N2 + j * N + k;
        out(idx) =  (
                          x(idx-1)
                        + x(idx+1)                    
                        + x(idx-N)
                        + x(idx+N)
                        + x(idx-N2)
                        + x(idx+N2)
                        - 6.f * x(idx)
                    ) * invh2;
    });
}
void Laplacian::interior_1d_mdrange_i32_idx32(Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const uint32_t N1 = N;
    const uint32_t N2 = N * N;
    const float invh2 = static_cast<float>(N2);

    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<uint32_t>>({1, 1, 1}, {N1-1, N1-1, N1-1}),
        KOKKOS_LAMBDA(const uint32_t i, const uint32_t j, const uint32_t k)
    {
        const uint32_t idx = i * N2 + j * N + k;
        out(idx) =  (
                          x(idx-1)
                        + x(idx+1)                    
                        + x(idx-N)
                        + x(idx+N)
                        + x(idx-N2)
                        + x(idx+N2)
                        - 6.f * x(idx)
                    ) * invh2;
    });
}
void Laplacian::interior_1d_mdrange_i32_idx64(Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const uint32_t N1 = N;
    const uint32_t N2 = N * N;
    const float invh2 = static_cast<float>(N2);

    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {N1-1, N1-1, N1-1}),
        KOKKOS_LAMBDA(const uint32_t i, const uint32_t j, const uint32_t k)
    {
        const size_t idx = i * N2 + j * N1 + k;
        out(idx) =  (
                          x(idx-1)
                        + x(idx+1)                    
                        + x(idx-N)
                        + x(idx+N)
                        + x(idx-N2)
                        + x(idx+N2)
                        - 6.f * x(idx)
                    ) * invh2;
    });
}
void Laplacian::interior_1d_mdrange_i32_idx64promotion(Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const uint32_t N1 = N;
    const uint32_t N2 = N * N;
    const float invh2 = static_cast<float>(N2);

    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {N1-1, N1-1, N1-1}),
        KOKKOS_LAMBDA(const uint32_t i, const uint32_t j, const uint32_t k)
    {
        const size_t idx = i * N2 + j * N + k;
        out(idx) =  (
                          x(idx-1)
                        + x(idx+1)                    
                        + x(idx-N)
                        + x(idx+N)
                        + x(idx-N2)
                        + x(idx+N2)
                        - 6.f * x(idx)
                    ) * invh2;
    });
}
void Laplacian::interior_1d_mdrange_i64_idx32(Kokkos::View<float*> out, const Kokkos::View<float*> x, const size_t N)
{
    const size_t N2 = N * N;
    const float invh2 = static_cast<float>(N2);

    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1ul, 1ul, 1ul}, {N-1ul, N-1ul, N-1ul}),
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        const uint32_t idx = i * N2 + j * N + k;
        out(idx) =  (
                          x(idx-1)
                        + x(idx+1)                    
                        + x(idx-N)
                        + x(idx+N)
                        + x(idx-N2)
                        + x(idx+N2)
                        - 6.f * x(idx)
                    ) * invh2;
    });
}

void Laplacian::interior_3d_flat(Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N)
{
    const size_t Ni = N-2;
    const size_t N2i = Ni*Ni;
    const size_t N3i = N2i*Ni;
    const size_t N2 = N*N;
    const float invh2 = static_cast<float>(N2);
    Kokkos::parallel_for("Laplacian", N3i, 
        KOKKOS_LAMBDA(const size_t idx)
    {
        size_t i = idx / N2i;
        size_t j = (idx - i * N2i) / Ni;
        size_t k = idx - i * N2i - j * Ni;
        i++; j++; k++;
        out(i, j, k) = (
                      x(i, j, k-1)
                    + x(i, j, k+1)                    
                    + x(i, j-1, k)
                    + x(i, j+1, k)
                    + x(i-1, j, k)
                    + x(i+1, j, k)
                    - 6.f * x(i, j, k)
                ) * invh2;
    });
}
void Laplacian::interior_3d_mdrange(Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N)
{
    const float invh2 = static_cast<float>(N * N);

    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {N-1, N-1, N-1}),
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        out(i, j, k) = (
                      x(i, j, k-1)
                    + x(i, j, k+1)                    
                    + x(i, j-1, k)
                    + x(i, j+1, k)
                    + x(i-1, j, k)
                    + x(i+1, j, k)
                    - 6.f * x(i, j, k)
                ) * invh2;
    });
}
void Laplacian::modulo_3d_flat(Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N)
{
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    const float invh2 = static_cast<float>(N * N);
    Kokkos::parallel_for("Laplacian", N3, 
        KOKKOS_LAMBDA(const size_t idx)
    {
        const size_t i = idx / N2;
        const size_t j = (idx - i * N2) / N;
        const size_t k = idx - i * N2 - j * N;
        out(i, j, k) = (
                      x(i, j, (k-1)%N)
                    + x(i, j, (k+1)%N)                    
                    + x(i, (j-1)%N, k)
                    + x(i, (j+1)%N, k)
                    + x((i-1)%N, j, k)
                    + x((i+1)%N, j, k)
                    - 6.f * x(i, j, k)
                ) * invh2;
    });
}
void Laplacian::modulo_3d_mdrange(Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N)
{
    const float invh2 = static_cast<float>(N * N);

    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        out(i, j, k) = (
                      x(i, j, (k-1)%N)
                    + x(i, j, (k+1)%N)                    
                    + x(i, (j-1)%N, k)
                    + x(i, (j+1)%N, k)
                    + x((i-1)%N, j, k)
                    + x((i+1)%N, j, k)
                    - 6.f * x(i, j, k)
                ) * invh2;
    });
}
void Laplacian::conditional_add_3d_flat(Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N)
{
    const size_t Nm1 = N - 1;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    const float invh2 = static_cast<float>(N * N);
    Kokkos::parallel_for("Laplacian", N3, 
        KOKKOS_LAMBDA(const size_t idx)
    {
        const size_t i = idx / N2;
        const size_t j = (idx - i * N2) / N;
        const size_t k = idx - i * N2 - j * N;
        const size_t im1 = (i - 1ul) + N*(i == 0ul);
        const size_t ip1 = (i + 1ul) - N*(i == Nm1);
        const size_t jm1 = (j - 1ul) + N*(j == 0ul);
        const size_t jp1 = (j + 1ul) - N*(j == Nm1);
        const size_t km1 = (k - 1ul) + N*(k == 0ul);
        const size_t kp1 = (k + 1ul) - N*(k == Nm1);
        out(i, j, k) = (
                      x(i, j, km1)
                    + x(i, j, kp1)                    
                    + x(i, jm1, k)
                    + x(i, jp1, k)
                    + x(im1, j, k)
                    + x(ip1, j, k)
                    - 6.f * x(i, j, k)
                ) * invh2;
    });
}
void Laplacian::conditional_add_3d_mdrange(Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N)
{
    const float invh2 = static_cast<float>(N * N);
    const size_t Nm1 = N - 1;
    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        const size_t im1 = (i - 1ul) + N*(i == 0ul);
        const size_t ip1 = (i + 1ul) - N*(i == Nm1);
        const size_t jm1 = (j - 1ul) + N*(j == 0ul);
        const size_t jp1 = (j + 1ul) - N*(j == Nm1);
        const size_t km1 = (k - 1ul) + N*(k == 0ul);
        const size_t kp1 = (k + 1ul) - N*(k == Nm1);
        out(i, j, k) = (
                      x(i, j, km1)
                    + x(i, j, kp1)                    
                    + x(i, jm1, k)
                    + x(i, jp1, k)
                    + x(im1, j, k)
                    + x(ip1, j, k)
                    - 6.f * x(i, j, k)
                ) * invh2;
    });
}
void Laplacian::ternary_3d_flat(Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N)
{
    const size_t Nm1 = N - 1;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    const float invh2 = static_cast<float>(N * N);
    Kokkos::parallel_for("Laplacian", N3, 
        KOKKOS_LAMBDA(const size_t idx)
    {
        const size_t i = idx / N2;
        const size_t j = (idx - i * N2) / N;
        const size_t k = idx - i * N2 - j * N;
        const size_t im1 = (i == 0ul) ? Nm1 : i - 1ul;
        const size_t ip1 = (i == Nm1) ? 0ul : i + 1ul;
        const size_t jm1 = (j  == 0ul) ? Nm1: j - 1ul;
        const size_t jp1 = (j  == Nm1) ? 0ul : j + 1ul;
        const size_t km1 = (k == 0ul) ? Nm1 : k - 1ul;
        const size_t kp1 = (k == Nm1) ? 0ul : k + 1ul;
        out(i, j, k) = (
                      x(i, j, km1)
                    + x(i, j, kp1)                    
                    + x(i, jm1, k)
                    + x(i, jp1, k)
                    + x(im1, j, k)
                    + x(ip1, j, k)
                    - 6.f * x(i, j, k)
                ) * invh2;
    });
}
void Laplacian::ternary_3d_mdrange(Kokkos::View<float***> out, const Kokkos::View<float***> x, const size_t N)
{
    const float invh2 = static_cast<float>(N * N);
    const size_t Nm1 = N - 1;
    Kokkos::parallel_for("Laplacian3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        const size_t im1 = (i == 0ul) ? Nm1 : i - 1ul;
        const size_t ip1 = (i == Nm1) ? 0ul : i + 1ul;
        const size_t jm1 = (j  == 0ul) ? Nm1: j - 1ul;
        const size_t jp1 = (j  == Nm1) ? 0ul : j + 1ul;
        const size_t km1 = (k == 0ul) ? Nm1 : k - 1ul;
        const size_t kp1 = (k == Nm1) ? 0ul : k + 1ul;
        out(i, j, k) = (
                      x(i, j, km1)
                    + x(i, j, kp1)                    
                    + x(i, jm1, k)
                    + x(i, jp1, k)
                    + x(im1, j, k)
                    + x(ip1, j, k)
                    - 6.f * x(i, j, k)
                ) * invh2;
    });
}