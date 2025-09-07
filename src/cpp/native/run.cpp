#include <iostream>
#include <array>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <fstream>   
#include <numeric>   

#include "run.h"
#include "laplacian.h"

    

std::string get_executable_name(const char* argv0) 
{
    std::string full_path(argv0);
    size_t pos = full_path.find_last_of("/\\"); // handle both '/' and '\' separators
    if (pos == std::string::npos)
        return full_path; // no path separators found, argv0 is already a filename
    else
        return full_path.substr(pos + 1);
}

template <typename Func>
void benchmark (const std::string& name, std::string& output_base, Func&& func)
{   
    constexpr uint32_t runs = 10;
    std::vector<float> timings(runs);
    for (uint32_t i(0); i < runs; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        std::chrono::duration<double, std::milli> elapsed = (std::chrono::high_resolution_clock::now() - start);
        timings[i] = elapsed.count();
    }
    const auto time = std::reduce(timings.begin(), timings.end()) / runs;
    std::cout << name<< " took " << time << " ms\n";
    const std::string output_file = output_base + "_" + name + ".dat";
    std::ofstream out_file(output_file);
    if (out_file.is_open())
    {
        for (auto t : timings)
            out_file << t << "\n";
    }
    else
        std::cerr << "Error: Could not write timings to file\n";
}


int main(int argc, char* argv[]) 
{

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " output_base.txt\n";
        return EXIT_FAILURE;
    }
    std::string output = get_executable_name(argv[0]);
    output.erase(0,10); 

    constexpr std::array<size_t, 4> N_array = {32, 64, 128, 256};
    for (size_t N : N_array) {
        std::cout << "\nN = " << N << "\n";

        std::string output_base = std::string() + argv[1] + "_ncells1d_" + std::to_string(N) + "_" + output;

        // 3D
        {
            std::vector<std::vector<std::vector<float>>> x_3d, out_3d;
            Run::initialise_vector3d(x_3d, N);
            Run::initialise_vector3d(out_3d, N);

            benchmark("modulo_3d_nested", output_base, [&]() { Native::modulo_3d_nested(out_3d, x_3d, N); });
            if (N == 32)
            {
                Native::check_interior(out_3d, N);
                Native::check_exterior(out_3d, N);
            }
            benchmark("modulo_3d_nested_simd", output_base, [&]() { Native::modulo_3d_nested_simd(out_3d, x_3d, N); });
            if (N == 32)
            {
                Native::check_interior(out_3d, N);
                Native::check_exterior(out_3d, N);
            }
            benchmark("conditional_add_3d_nested", output_base, [&]() { Native::conditional_add_3d_nested(out_3d, x_3d, N); });
            if (N == 32)
            {
                Native::check_interior(out_3d, N);
                Native::check_exterior(out_3d, N);
            }
            benchmark("conditional_add_3d_nested_simd", output_base, [&]() { Native::conditional_add_3d_nested_simd(out_3d, x_3d, N); });
            if (N == 32)
            {
                Native::check_interior(out_3d, N);
                Native::check_exterior(out_3d, N);
            }
            benchmark("ternary_3d_nested", output_base, [&]() { Native::ternary_3d_nested(out_3d, x_3d, N); });
            if (N == 32)
            {
                Native::check_interior(out_3d, N);
                Native::check_exterior(out_3d, N);
            }
            benchmark("ternary_3d_nested_simd", output_base, [&]() { Native::ternary_3d_nested_simd(out_3d, x_3d, N); });
            if (N == 32)
            {
                Native::check_interior(out_3d, N);
                Native::check_exterior(out_3d, N);
            }
            benchmark("interior_3d_flat", output_base, [&]() { Native::interior_3d_flat(out_3d, x_3d, N); });
            if (N == 32) Native::check_interior(out_3d, N);
            benchmark("interior_3d_flat_simd", output_base, [&]() { Native::interior_3d_flat_simd(out_3d, x_3d, N); });
            if (N == 32) Native::check_interior(out_3d, N);
            benchmark("interior_3d_nested", output_base, [&]() { Native::interior_3d_nested(out_3d, x_3d, N); });
            if (N == 32) Native::check_interior(out_3d, N);
            benchmark("interior_3d_nested_simd", output_base, [&]() { Native::interior_3d_nested_simd(out_3d, x_3d, N); });
            if (N == 32) Native::check_interior(out_3d, N);
            benchmark("interior_3d_nested_constexpr", output_base, [&]() { Native::run_interior_3d_nested_constexpr(out_3d, x_3d, N); });
            if (N == 32) Native::check_interior(out_3d, N);
            benchmark("interior_3d_nested_constexpr_simd", output_base, [&]() { Native::run_interior_3d_nested_constexpr_simd(out_3d, x_3d, N); });
            if (N == 32) Native::check_interior(out_3d, N);
        }

        // 1D std::vector<float>
        {
            std::vector<float> x_1d, out_1d;
            Run::initialise_vector1d(x_1d, N);
            Run::initialise_vector1d(out_1d, N);

            benchmark("interior_1d_flat", output_base, [&]() { Native::interior_1d_flat(out_1d, x_1d, N); });
            if (N == 32) Native::check_interior(out_1d, N);
            benchmark("interior_1d_flat_simd", output_base, [&]() { Native::interior_1d_flat_simd(out_1d, x_1d, N); });
            if (N == 32) Native::check_interior(out_1d, N);
            benchmark("interior_1d_nested", output_base, [&]() { Native::interior_1d_nested(out_1d, x_1d, N); });
            if (N == 32) Native::check_interior(out_1d, N);
            benchmark("interior_1d_nested_simd", output_base, [&]() { Native::interior_1d_nested_simd(out_1d, x_1d, N); });
            if (N == 32) Native::check_interior(out_1d, N);
            benchmark("modulo_1d_flat", output_base, [&]() { Native::modulo_1d_flat(out_1d, x_1d, N); });
            if (N == 32)
            {
                Native::check_interior(out_1d, N);
                Native::check_exterior(out_1d, N);
            }
            benchmark("modulo_1d_flat_simd", output_base, [&]() { Native::modulo_1d_flat_simd(out_1d, x_1d, N); });
            if (N == 32) 
            {
                Native::check_interior(out_1d, N);
                Native::check_exterior(out_1d, N);
            }
        }

        // 1D float* via malloc
        {
            float* x_1d_malloc = static_cast<float*>(std::malloc(N * N * N * sizeof(float)));
            float* out_1d_malloc = static_cast<float*>(std::malloc(N * N * N * sizeof(float)));

            Run::initialise_malloc(x_1d_malloc, N);
            Run::initialise_malloc(out_1d_malloc, N);

            benchmark("interior_1d_malloc_nested", output_base, [&]() { Native::interior_1d_malloc_nested(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_simd", output_base, [&]() { Native::interior_1d_malloc_nested_simd(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_i32_max32_idx32", output_base, [&]() { Native::interior_1d_malloc_nested_i32_max32_idx32(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_i32_max32_idx32_simd", output_base, [&]() { Native::interior_1d_malloc_nested_i32_max32_idx32_simd(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_i32_max32_idx64", output_base, [&]() { Native::interior_1d_malloc_nested_i32_max32_idx64(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_i32_max32_idx64_simd", output_base, [&]() { Native::interior_1d_malloc_nested_i32_max32_idx64_simd(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_i32_max32_idx64promotion", output_base, [&]() { Native::interior_1d_malloc_nested_i32_max32_idx64promotion(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_i32_max32_idx64promotion_simd", output_base, [&]() { Native::interior_1d_malloc_nested_i32_max32_idx64promotion_simd(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_i32_max64_idx64", output_base, [&]() { Native::interior_1d_malloc_nested_i32_max64_idx64(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_i32_max64_idx64_simd", output_base, [&]() { Native::interior_1d_malloc_nested_i32_max64_idx64_simd(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_i32_max64_idx32", output_base, [&]() { Native::interior_1d_malloc_nested_i32_max64_idx32(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_i32_max64_idx32_simd", output_base, [&]() { Native::interior_1d_malloc_nested_i32_max64_idx32_simd(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_constexpr", output_base, [&]() { Native::run_interior_1d_malloc_nested_constexpr(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);
            benchmark("interior_1d_malloc_nested_constexpr_simd", output_base, [&]() { Native::run_interior_1d_malloc_nested_constexpr_simd(out_1d_malloc, x_1d_malloc, N); });
            if (N == 32) Native::check_interior(out_1d_malloc, N);

            std::free(x_1d_malloc);
            std::free(out_1d_malloc);
        }

        // 1D float* via std::aligned_alloc
        {
            float* x_1d_aligned = static_cast<float*>(std::aligned_alloc(64, N*N*N*sizeof(float)));
            float* out_1d_aligned = static_cast<float*>(std::aligned_alloc(64, N*N*N*sizeof(float)));

            Run::initialise_malloc(x_1d_aligned, N);
            Run::initialise_malloc(out_1d_aligned, N);

            benchmark("interior_1d_aligned_nested", output_base, [&]() { Native::interior_1d_aligned_nested(out_1d_aligned, x_1d_aligned, N); });
            if (N == 32) Native::check_interior(out_1d_aligned, N);

            std::free(x_1d_aligned);
            std::free(out_1d_aligned);
        }
    }

    return EXIT_SUCCESS;
}
