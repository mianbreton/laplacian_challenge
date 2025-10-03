#include <iostream>
#include <array>
#include <vector>
#include <cstdint>

class Run
{
    public:
    static void initialise_malloc(float* x, const size_t N);
    static void initialise_vector1d(std::vector<float>& x, const size_t N);
    static void initialise_vector3d(std::vector<std::vector<std::vector<float>>>& x, const size_t N);
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