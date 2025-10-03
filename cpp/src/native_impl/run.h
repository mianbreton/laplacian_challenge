#pragma once

#include <iostream>
#include <array>
#include <vector>
#include <cstdint>
#include <cassert>

class Run
{
    public:
    static void initialise(float* x, const size_t N);
    static void initialise(std::vector<float>& x, const size_t N);
    static void initialise_zero(float* x, const size_t N);
    static void initialise_zero(std::vector<float>& x, const size_t N);
    static void initialise(std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void initialise_zero(std::vector<std::vector<std::vector<float>>>& x, const size_t N);
};

void Run::initialise(float* x, const size_t N)
{
    for(size_t i(0); i < N*N*N; i++)
        x[i] = i;
}
void Run::initialise(std::vector<float>& x, const size_t N)
{
    for(size_t i(0); i < N*N*N; i++)
        x[i] = i;
}
void Run::initialise_zero(float* x, const size_t N)
{
    for(size_t i(0); i < N*N*N; i++)
        x[i] = 0.f;
}
void Run::initialise_zero(std::vector<float>& x, const size_t N)
{
    for(size_t i(0); i < N*N*N; i++)
        x[i] = 0.f;
}
void Run::initialise(std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    const size_t N2 = N*N;
    for(size_t i(0); i < N; i++)
    for(size_t j(0); j < N; j++)
    for(size_t k(0); k < N; k++)
    {
        const int idx = i*N2 + j*N + k;
        x[i][j][k] = idx;
    }
}
void Run::initialise_zero(std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    for(size_t i(0); i < N; i++)
    for(size_t j(0); j < N; j++)
    for(size_t k(0); k < N; k++)
        x[i][j][k] = 0.f;
}