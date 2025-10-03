#include <iostream>
#include <array>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <fstream>   
#include <numeric>   

#include "../src/native_impl/run.h"
#include "../src/native_impl/laplacian.h"
#include "test_native_utils.h"

#define BENCHMARK(func, out, x, N) \
    benchmark(strip_namespace(#func), out, N, [&]() { func(out, x, N); })

std::string strip_namespace(const std::string& full_name) 
{
    size_t pos = full_name.rfind("::");
    return (pos == std::string::npos) ? full_name : full_name.substr(pos + 2);
}

template <typename Func, class Array>
void benchmark (const std::string& name, Array& out, const int N, Func&& func)
{   
    Run::initialise_zero(out, N);
    func();
    Testing::check_interior(out, N);
    if (name.rfind("interior") == std::string::npos)
        Testing::check_exterior(out, N);
}

int main() 
{

    constexpr uint32_t N = 32; // Number of cells along each dimension
    
    // 3D
    {
        std::vector<std::vector<std::vector<float>>> x_3d, out_3d;
        x_3d.resize(N);
        out_3d.resize(N);
        for (size_t i(0); i < N; ++i) 
        {
            x_3d[i].resize(N);
            out_3d[i].resize(N);
            for (size_t j(0); j < N; ++j) 
            {
                x_3d[i][j].resize(N);
                out_3d[i][j].resize(N);
            }
        }
        Run::initialise(x_3d, N);
        BENCHMARK(Laplacian::modulo_3d_nested, out_3d, x_3d, N);
        BENCHMARK(Laplacian::modulo_3d_nested_simd, out_3d, x_3d, N);
        BENCHMARK(Laplacian::conditional_add_3d_nested, out_3d, x_3d, N);
        BENCHMARK(Laplacian::conditional_add_3d_nested_simd, out_3d, x_3d, N);
        BENCHMARK(Laplacian::ternary_3d_nested_simd, out_3d, x_3d, N);
        BENCHMARK(Laplacian::ternary_3d_nested, out_3d, x_3d, N);
        BENCHMARK(Laplacian::interior_3d_flat, out_3d, x_3d, N);
        BENCHMARK(Laplacian::interior_3d_flat_simd, out_3d, x_3d, N);
        BENCHMARK(Laplacian::interior_3d_nested, out_3d, x_3d, N);
        BENCHMARK(Laplacian::interior_3d_nested_simd, out_3d, x_3d, N);
        BENCHMARK(Laplacian::interior_3d_nested_constexpr, out_3d, x_3d, N);
        BENCHMARK(Laplacian::interior_3d_nested_constexpr_simd, out_3d, x_3d, N);
    }

    // 1D std::vector<float>
    {
        std::vector<float> x_1d(N*N*N), out_1d(N*N*N);
        Run::initialise(x_1d, N);
        BENCHMARK(Laplacian::interior_1d_flat, out_1d, x_1d, N);
        BENCHMARK(Laplacian::interior_1d_flat_simd, out_1d, x_1d, N);
        BENCHMARK(Laplacian::interior_1d_nested, out_1d, x_1d, N);
        BENCHMARK(Laplacian::interior_1d_nested_simd, out_1d, x_1d, N);
        BENCHMARK(Laplacian::modulo_1d_flat, out_1d, x_1d, N);
        BENCHMARK(Laplacian::modulo_1d_flat_simd, out_1d, x_1d, N);
    }

    // 1D float* via malloc
    {
        float* x_1d_malloc = static_cast<float*>(std::malloc(N * N * N * sizeof(float)));
        float* out_1d_malloc = static_cast<float*>(std::malloc(N * N * N * sizeof(float)));
        Run::initialise(x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_simd, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx32, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx32_simd, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64_simd, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64promotion, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64promotion_simd, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max64_idx64, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max64_idx64_simd, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max64_idx32, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max64_idx32_simd, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_constexpr, out_1d_malloc, x_1d_malloc, N);
        BENCHMARK(Laplacian::interior_1d_malloc_nested_constexpr_simd, out_1d_malloc, x_1d_malloc, N);
        std::free(x_1d_malloc);
        std::free(out_1d_malloc);
    }

    // 1D float* via std::aligned_alloc
    {
        float* x_1d_aligned = static_cast<float*>(std::aligned_alloc(64, N*N*N*sizeof(float)));
        float* out_1d_aligned = static_cast<float*>(std::aligned_alloc(64, N*N*N*sizeof(float)));
        Run::initialise(x_1d_aligned, N);
        BENCHMARK(Laplacian::interior_1d_aligned_nested, out_1d_aligned, x_1d_aligned, N);
        std::free(x_1d_aligned);
        std::free(out_1d_aligned);
    }

    return EXIT_SUCCESS;
}
