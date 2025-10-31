#pragma once

#include <Kokkos_Core.hpp>

class Run {
    public:

    static void initialise(Kokkos::View<float*> x, const size_t N);
    static void initialise_zero(Kokkos::View<float*> x, const size_t N);
    static void initialise(Kokkos::View<float***> x, const size_t N);
    static void initialise_zero(Kokkos::View<float***> x, const size_t N);
};

void Run::initialise(Kokkos::View<float*> x, const size_t N)
{
    const size_t size = N*N*N; 
    Kokkos::parallel_for("Initialise", size, 
        KOKKOS_LAMBDA(const size_t i)
    {
        x(i) = i;
    });
}
void Run::initialise_zero(Kokkos::View<float*> x, const size_t N)
{
    const size_t size = N*N*N; 
    Kokkos::parallel_for("Initialise", size, 
        KOKKOS_LAMBDA(const size_t i)
    {
        x(i) = 0.f;
    });
}
void Run::initialise(Kokkos::View<float***> x, const size_t N)
{
    const size_t N2 = N*N;
    Kokkos::parallel_for("Initialise", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}), 
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        const size_t idx = i*N2 + j*N + k;
        x(i,j,k) = idx;
    });
}
void Run::initialise_zero(Kokkos::View<float***> x, const size_t N)
{
    Kokkos::parallel_for("Initialise", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}), 
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k)
    {
        x(i,j,k) = 0.f;
    });
}