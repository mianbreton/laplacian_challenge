#pragma once

#include <iostream>
#include <array>
#include <vector>
#include <cstdint>
#include <cassert>

class Run
{
    public:
    static void initialise_malloc(float* x, const size_t N);
    static void initialise_vector1d(std::vector<float>& x, const size_t N);
    static void initialise_vector3d(std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void check_interior(const std::vector<float>& x, const size_t N);
    static void check_interior(const float* x, const size_t N);
    static void check_interior(const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void check_exterior(const std::vector<float>& x, const size_t N);
    static void check_exterior(const float* x, const size_t N);
    static void check_exterior(const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
};

void Run::initialise_malloc(float* x, const size_t N)
{
    for(uint32_t i(0u); i < N*N*N; i++)
        x[i] = i;
}
void Run::initialise_vector1d(std::vector<float>& x, const size_t N)
{
    x.resize(N*N*N);
    for(uint32_t i(0u); i < N*N*N; i++)
        x[i] = i;
}
void Run::initialise_vector3d(std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    x.resize(N);
    const size_t N2 = N*N;
    for (uint32_t i(0u); i < N; ++i) 
    {
        x[i].resize(N);
        for (uint32_t j(0u); j < N; ++j) 
            x[i][j].resize(N);
    }

    for(uint32_t i(0u); i < N; i++)
    for(uint32_t j(0u); j < N; j++)
    for(uint32_t k(0u); k < N; k++)
    {
        const int idx = i*N2 + j*N + k;
        x[i][j][k] = idx;
    }
}

void assertion(const float value, const float expected)
{
    const bool isGood = (fabsf(value - expected) < 1e-7f);
    assert(isGood);
}

void Run::check_interior(const std::vector<float>& x, const size_t N)
{
    const size_t N2 = N*N;
    const size_t Nm1 = N - 1ul;
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i*N2 + j*N + k], 0.f);
    }
}
void Run::check_interior(const float* x, const size_t N)
{
    const size_t N2 = N*N;
    const size_t Nm1 = N - 1ul;
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i*N2 + j*N + k], 0.f);
    }
}
void Run::check_interior(const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{   
    const size_t Nm1 = N - 1ul;
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i][j][k], 0.f);
    }
}
void Run::check_exterior(const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    const size_t Nm1 = N - 1ul;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    // Faces
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i][Nm1][k], -static_cast<float>(N2*N2));
        assertion(x[i][0][k], static_cast<float>(N2*N2));
    }
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1][j][k], -static_cast<float>(N2*N3));
        assertion(x[0][j][k], static_cast<float>(N2*N3));
    }
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[i][j][Nm1], -static_cast<float>(N2*N));
        assertion(x[i][j][0], static_cast<float>(N2*N));
    }
    // Edges 
    for(size_t i=1ul; i < Nm1; i++)
    {
        assertion(x[i][Nm1][Nm1], -static_cast<float>(N2*(N2+N)));
        assertion(x[i][Nm1][0], -static_cast<float>(N2*(N2-N)));
        assertion(x[i][0][Nm1], static_cast<float>(N2*(N2-N)));
        assertion(x[i][0][0], static_cast<float>(N2*(N2+N)));
    }
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[Nm1][j][Nm1], -static_cast<float>(N2*(N3+N)));
        assertion(x[Nm1][j][0], -static_cast<float>(N2*(N3-N)));
        assertion(x[0][j][Nm1], static_cast<float>(N2*(N3-N)));
        assertion(x[0][j][0], static_cast<float>(N2*(N3+N)));
    }
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1][Nm1][k], -static_cast<float>(N2*(N3+N2)));
        assertion(x[Nm1][0][k], -static_cast<float>(N2*(N3-N2)));
        assertion(x[0][Nm1][k], static_cast<float>(N2*(N3-N2)));
        assertion(x[0][0][k], static_cast<float>(N2*(N3+N2)));
    }
    // Corners
    assertion(x[Nm1][Nm1][Nm1] , -static_cast<float>(N2*(N3+N2+N)));
    assertion(x[Nm1][Nm1][0], -static_cast<float>(N2*(N3+N2-N)));
    assertion(x[Nm1][0][Nm1], -static_cast<float>(N2*(N3-N2+N)));
    assertion(x[Nm1][0][0], -static_cast<float>(N2*(N3-N2-N)));
    assertion(x[0][Nm1][Nm1], static_cast<float>(N2*(N3-N2-N)));
    assertion(x[0][Nm1][0], static_cast<float>(N2*(N3-N2+N)));
    assertion(x[0][0][Nm1], static_cast<float>(N2*(N3+N2-N)));
    assertion(x[0][0][0], static_cast<float>(N2*(N3+N2+N)));
}
void Run::check_exterior(const std::vector<float>& x, const size_t N)
{
    const size_t Nm1 = N - 1ul;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    // Faces
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i*N2 + Nm1*N + k], -static_cast<float>(N2*N2));
        assertion(x[i*N2 + k], static_cast<float>(N2*N2));
    }
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1*N2 + j*N + k], -static_cast<float>(N2*N3));
        assertion(x[j*N + k], static_cast<float>(N2*N3));
    }
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[i*N2 + j*N + Nm1], -static_cast<float>(N2*N));
        assertion(x[i*N2 + j*N], static_cast<float>(N2*N));
    }
    // Edges 
    for(size_t i=1ul; i < Nm1; i++)
    {
        assertion(x[i*N2 + Nm1*N + Nm1], -static_cast<float>(N2*(N2+N)));
        assertion(x[i*N2 + Nm1*N], -static_cast<float>(N2*(N2-N)));
        assertion(x[i*N2 + Nm1], static_cast<float>(N2*(N2-N)));
        assertion(x[i*N2], static_cast<float>(N2*(N2+N)));
    }
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[Nm1*N2 + j*N + Nm1], -static_cast<float>(N2*(N3+N)));
        assertion(x[Nm1*N2 + j*N], -static_cast<float>(N2*(N3-N)));
        assertion(x[j*N + Nm1], static_cast<float>(N2*(N3-N)));
        assertion(x[j*N], static_cast<float>(N2*(N3+N)));
    }
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1*N2 + Nm1*N + k], -static_cast<float>(N2*(N3+N2)));
        assertion(x[Nm1*N2 + k], -static_cast<float>(N2*(N3-N2)));
        assertion(x[Nm1*N + k], static_cast<float>(N2*(N3-N2)));
        assertion(x[k], static_cast<float>(N2*(N3+N2)));
    }
    // Corners
    assertion(x[Nm1*N2 + Nm1*N + Nm1], -static_cast<float>(N2*(N3+N2+N)));
    assertion(x[Nm1*N2 + Nm1*N], -static_cast<float>(N2*(N3+N2-N)));
    assertion(x[Nm1*N2 + Nm1], -static_cast<float>(N2*(N3-N2+N)));
    assertion(x[Nm1*N2], -static_cast<float>(N2*(N3-N2-N)));
    assertion(x[Nm1*N + Nm1], static_cast<float>(N2*(N3-N2-N)));
    assertion(x[Nm1*N], static_cast<float>(N2*(N3-N2+N)));
    assertion(x[Nm1], static_cast<float>(N2*(N3+N2-N)));
    assertion(x[0], static_cast<float>(N2*(N3+N2+N)));
}
void Run::check_exterior(const float* x, const size_t N)
{
    printf("Check exterior\n");
    const size_t Nm1 = N - 1ul;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    // Faces
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i*N2 + Nm1*N + k], -static_cast<float>(N2*N2));
        assertion(x[i*N2 + k], static_cast<float>(N2*N2));
    }
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1*N2 + j*N + k], -static_cast<float>(N2*N3));
        assertion(x[j*N + k], static_cast<float>(N2*N3));
    }
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[i*N2 + j*N + Nm1], -static_cast<float>(N2*N));
        assertion(x[i*N2 + j*N], static_cast<float>(N2*N));
    }
    // Edges 
    for(size_t i=1ul; i < Nm1; i++)
    {
        assertion(x[i*N2 + Nm1*N + Nm1], -static_cast<float>(N2*(N2+N)));
        assertion(x[i*N2 + Nm1*N], -static_cast<float>(N2*(N2-N)));
        assertion(x[i*N2 + Nm1], static_cast<float>(N2*(N2-N)));
        assertion(x[i*N2], static_cast<float>(N2*(N2+N)));
    }
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[Nm1*N2 + j*N + Nm1], -static_cast<float>(N2*(N3+N)));
        assertion(x[Nm1*N2 + j*N], -static_cast<float>(N2*(N3-N)));
        assertion(x[j*N + Nm1], static_cast<float>(N2*(N3-N)));
        assertion(x[j*N], static_cast<float>(N2*(N3+N)));
    }
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1*N2 + Nm1*N + k], -static_cast<float>(N2*(N3+N2)));
        assertion(x[Nm1*N2 + k], -static_cast<float>(N2*(N3-N2)));
        assertion(x[Nm1*N + k], static_cast<float>(N2*(N3-N2)));
        assertion(x[k], static_cast<float>(N2*(N3+N2)));
    }
    // Corners
    assertion(x[Nm1*N2 + Nm1*N + Nm1], -static_cast<float>(N2*(N3+N2+N)));
    assertion(x[Nm1*N2 + Nm1*N], -static_cast<float>(N2*(N3+N2-N)));
    assertion(x[Nm1*N2 + Nm1], -static_cast<float>(N2*(N3-N2+N)));
    assertion(x[Nm1*N2], -static_cast<float>(N2*(N3-N2-N)));
    assertion(x[Nm1*N + Nm1], static_cast<float>(N2*(N3-N2-N)));
    assertion(x[Nm1*N], static_cast<float>(N2*(N3-N2+N)));
    assertion(x[Nm1], static_cast<float>(N2*(N3+N2-N)));
    assertion(x[0], static_cast<float>(N2*(N3+N2+N)));
}