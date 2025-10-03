#pragma once

#include <Kokkos_Core.hpp>

class Run {
    public:

    static void initialise_1d(Kokkos::View<float*> x, const size_t N);
    static void initialise_1d_zero(Kokkos::View<float*> x, const size_t N);
    static void initialise_3d(Kokkos::View<float***> x, const size_t N);
    static void initialise_3d_zero(Kokkos::View<float***> x, const size_t N);
    static void check_interior(const Kokkos::View<float*> x, const size_t N);
    static void check_interior(const Kokkos::View<float***> x, const size_t N);
    static void check_exterior(const Kokkos::View<float*> x, const size_t N);
    static void check_exterior(const Kokkos::View<float***> x, const size_t N);
};

void Run::initialise_1d(Kokkos::View<float*> x, const size_t N)
{
    const size_t size = N*N*N; 
    Kokkos::parallel_for("Initialise", size, 
        KOKKOS_LAMBDA(const size_t i)
    {
        x(i) = i;
    });
}
void Run::initialise_1d_zero(Kokkos::View<float*> x, const size_t N)
{
    const size_t size = N*N*N; 
    Kokkos::parallel_for("Initialise", size, 
        KOKKOS_LAMBDA(const size_t i)
    {
        x(i) = 0;
    });
}
void Run::initialise_3d(Kokkos::View<float***> x, const size_t N)
{
    const size_t N2 = N*N;
    Kokkos::parallel_for("Initialise", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}), 
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        const size_t idx = i*N2 + j*N + k;
        x(i,j,k) = idx;
    });
}
void Run::initialise_3d_zero(Kokkos::View<float***> x, const size_t N)
{
    Kokkos::parallel_for("Initialise", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}), 
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        x(i,j,k) = 0;
    });
}

KOKKOS_INLINE_FUNCTION
void assertion_device(const float value, const float expected) 
{
    if (Kokkos::fabsf(value - expected) > 1e-7f) {
        Kokkos::abort("Device-side assertion failed!\n");
    }
}

KOKKOS_INLINE_FUNCTION
void assertion_host(const float value, const float expected)
{
    const bool isGood = (Kokkos::fabsf(value - expected) < 1e-7f);
    KOKKOS_ASSERT(isGood);
}

KOKKOS_INLINE_FUNCTION
void assertion(const float value, const float expected)
{
    #ifdef KOKKOS_HAS_GPU
        assertion_device(value, expected);
    #else
        assertion_host(value, expected);
    #endif
}

void Run::check_interior(const Kokkos::View<float*> x, const size_t N)
{
    printf("Check interior\n");
    const size_t N2 = N*N;
    Kokkos::parallel_for("Check", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {N-1, N-1, N-1}), 
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        assertion(x(i*N2 + j*N + k), 0.f);
    });
}
void Run::check_interior(const Kokkos::View<float***> x, const size_t N)
{   
    printf("Check interior\n");
    Kokkos::parallel_for("Check", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {N-1, N-1, N-1}), 
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        assertion(x(i,j,k), 0.f);
    });
}
void Run::check_exterior(const Kokkos::View<float*> x, const size_t N)
{
    printf("Check exterior\n");
    const size_t Nm1 = N - 1;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    // Faces
    Kokkos::parallel_for("Check", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N-1, N-1}), 
        KOKKOS_LAMBDA(const size_t i, const size_t k)
    {
        assertion(x(i*N2 + Nm1*N + k), -static_cast<float>(N2*N2));
        assertion(x(i*N2 + k), static_cast<float>(N2*N2));
    });
    Kokkos::parallel_for("Check", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N-1, N-1}), 
        KOKKOS_LAMBDA(const size_t j, const size_t k)
    {
        assertion(x(Nm1*N2 + j*N + k), -static_cast<float>(N2*N3));
        assertion(x(j*N + k), static_cast<float>(N2*N3));
    });
    Kokkos::parallel_for("Check", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N-1, N-1}), 
        KOKKOS_LAMBDA(const size_t i, const size_t j)
    {
        assertion(x(i*N2 + j*N + Nm1), -static_cast<float>(N2*N));
        assertion(x(i*N2 + j*N), static_cast<float>(N2*N));
    });
    // Edges 
    Kokkos::parallel_for("Check", Kokkos::RangePolicy<>(1, Nm1), 
        KOKKOS_LAMBDA(const size_t i)
    {
        assertion(x(i*N2 + Nm1*N + Nm1), -static_cast<float>(N2*(N2+N)));
        assertion(x(i*N2 + Nm1*N), -static_cast<float>(N2*(N2-N)));
        assertion(x(i*N2 + Nm1), static_cast<float>(N2*(N2-N)));
        assertion(x(i*N2), static_cast<float>(N2*(N2+N)));
    });
    Kokkos::parallel_for("Check", Kokkos::RangePolicy<>(1, Nm1), 
        KOKKOS_LAMBDA(const size_t j)
    {
        assertion(x(Nm1*N2 + j*N + Nm1), -static_cast<float>(N2*(N3+N)));
        assertion(x(Nm1*N2 + j*N), -static_cast<float>(N2*(N3-N)));
        assertion(x(j*N + Nm1), static_cast<float>(N2*(N3-N)));
        assertion(x(j*N), static_cast<float>(N2*(N3+N)));
    });
    Kokkos::parallel_for("Check", Kokkos::RangePolicy<>(1, Nm1),
        KOKKOS_LAMBDA(const size_t k)
    {
        assertion(x(Nm1*N2 + Nm1*N + k), -static_cast<float>(N2*(N3+N2)));
        assertion(x(Nm1*N2 + k), -static_cast<float>(N2*(N3-N2)));
        assertion(x(Nm1*N + k), static_cast<float>(N2*(N3-N2)));
        assertion(x(k), static_cast<float>(N2*(N3+N2)));
    });
    // Corners
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(Nm1*N2 + Nm1*N + Nm1), -static_cast<float>(N2*(N3+N2+N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(Nm1*N2 + Nm1*N), -static_cast<float>(N2*(N3+N2-N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(Nm1*N2 + Nm1), -static_cast<float>(N2*(N3-N2+N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(Nm1*N2), -static_cast<float>(N2*(N3-N2-N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(Nm1*N + Nm1), static_cast<float>(N2*(N3-N2-N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(Nm1*N), static_cast<float>(N2*(N3-N2+N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(Nm1), static_cast<float>(N2*(N3+N2-N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(0), static_cast<float>(N2*(N3+N2+N)));});
}
void Run::check_exterior(const Kokkos::View<float***> x, const size_t N)
{
    printf("Check exterior\n");
    const size_t Nm1 = N - 1;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    // Faces
    Kokkos::parallel_for("Check", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N-1, N-1}), 
        KOKKOS_LAMBDA(const size_t i, const size_t k)
    {
        assertion(x(i, Nm1, k), -static_cast<float>(N2*N2));
        assertion(x(i, 0, k), static_cast<float>(N2*N2));
    });
    Kokkos::parallel_for("Check", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N-1, N-1}), 
        KOKKOS_LAMBDA(const size_t j, const size_t k)
    {
        assertion(x(Nm1, j, k), -static_cast<float>(N2*N3));
        assertion(x(0, j, k), static_cast<float>(N2*N3));
    });
    Kokkos::parallel_for("Check", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N-1, N-1}), 
        KOKKOS_LAMBDA(const size_t i, const size_t j)
    {
        assertion(x(i, j, Nm1), -static_cast<float>(N2*N));
        assertion(x(i, j, 0), static_cast<float>(N2*N));
    });
    // Edges 
    Kokkos::parallel_for("Check", Kokkos::RangePolicy<>(1, Nm1), 
        KOKKOS_LAMBDA(const size_t i)
    {
        assertion(x(i, Nm1, Nm1), -static_cast<float>(N2*(N2+N)));
        assertion(x(i, Nm1, 0), -static_cast<float>(N2*(N2-N)));
        assertion(x(i, 0, Nm1), static_cast<float>(N2*(N2-N)));
        assertion(x(i, 0, 0), static_cast<float>(N2*(N2+N)));
    });
    Kokkos::parallel_for("Check", Kokkos::RangePolicy<>(1, Nm1), 
        KOKKOS_LAMBDA(const size_t j)
    {
        assertion(x(Nm1, j, Nm1), -static_cast<float>(N2*(N3+N)));
        assertion(x(Nm1, j, 0), -static_cast<float>(N2*(N3-N)));
        assertion(x(0, j, Nm1), static_cast<float>(N2*(N3-N)));
        assertion(x(0, j, 0), static_cast<float>(N2*(N3+N)));
    });
    Kokkos::parallel_for("Check", Kokkos::RangePolicy<>(1, Nm1),
        KOKKOS_LAMBDA(const size_t k)
    {
        assertion(x(Nm1, Nm1, k), -static_cast<float>(N2*(N3+N2)));
        assertion(x(Nm1, 0, k), -static_cast<float>(N2*(N3-N2)));
        assertion(x(0, Nm1, k), static_cast<float>(N2*(N3-N2)));
        assertion(x(0, 0, k), static_cast<float>(N2*(N3+N2)));
    });
    // Corners
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(Nm1, Nm1, Nm1), -static_cast<float>(N2*(N3+N2+N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(Nm1, Nm1, 0), -static_cast<float>(N2*(N3+N2-N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(Nm1, 0, Nm1), -static_cast<float>(N2*(N3-N2+N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(Nm1, 0, 0), -static_cast<float>(N2*(N3-N2-N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(0, Nm1, Nm1), static_cast<float>(N2*(N3-N2-N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(0, Nm1, 0), static_cast<float>(N2*(N3-N2+N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(0, 0, Nm1), static_cast<float>(N2*(N3+N2-N)));});
    Kokkos::parallel_for("Check", 1, KOKKOS_LAMBDA(const size_t)
    {assertion(x(0, 0, 0), static_cast<float>(N2*(N3+N2+N)));});
}