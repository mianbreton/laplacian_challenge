#include <iostream>
#include <array>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <fstream>   
#include <numeric>   

#include "CLI/CLI.hpp"
#include "run.h"
#include "laplacian.h"

#define BENCHMARK(func, output_base, runs, ...) \
    benchmark(strip_namespace(#func), output_base, runs, [&]() { func(__VA_ARGS__); })

std::string get_executable_name(const char* argv0) 
{
    std::string full_path(argv0);
    size_t pos = full_path.find_last_of("/\\"); // handle both '/' and '\' separators
    if (pos == std::string::npos)
        return full_path; // no path separators found, argv0 is already a filename
    else
        return full_path.substr(pos + 1);
}

std::string strip_namespace(const std::string& full_name) {
    size_t pos = full_name.rfind("::");
    return (pos == std::string::npos) ? full_name : full_name.substr(pos + 2);
}

template <typename Func>
void benchmark (const std::string& name, std::string& output_base, const uint32_t runs, Func&& func)
{   
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
    
    CLI::App app{"Laplacian Benchmarking"};

    std::string basename;
    std::vector<size_t> ncells;
    int runs = 10;
    // Multiple values allowed
    app.add_option("-b,--basename", basename, "Output file prefix")->required();
    app.add_option("--ncells", ncells, "Number of cells along each dimension")->required()->check(CLI::PositiveNumber);
    app.add_option("--runs", runs, "Number of repetitions for timing")->check(CLI::PositiveNumber);

    CLI11_PARSE(app, argc, argv);  // Parse the arguments

    std::string output = get_executable_name(argv[0]);
    output.erase(0,10); 

    for (const size_t N: ncells)
    {
        std::cout << "\nN = " << N << "\n";

        std::string output_base = std::string() + basename + "_ncells1d_" + std::to_string(N) + "_" + output;

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
            BENCHMARK(Laplacian::modulo_3d_nested, output_base, runs, out_3d, x_3d, N);
            BENCHMARK(Laplacian::modulo_3d_nested_simd, output_base, runs, out_3d, x_3d, N);
            BENCHMARK(Laplacian::conditional_add_3d_nested, output_base, runs, out_3d, x_3d, N);
            BENCHMARK(Laplacian::conditional_add_3d_nested_simd, output_base, runs, out_3d, x_3d, N);
            BENCHMARK(Laplacian::ternary_3d_nested_simd, output_base, runs, out_3d, x_3d, N);
            BENCHMARK(Laplacian::ternary_3d_nested, output_base, runs, out_3d, x_3d, N);
            BENCHMARK(Laplacian::interior_3d_flat, output_base, runs, out_3d, x_3d, N);
            BENCHMARK(Laplacian::interior_3d_flat_simd, output_base, runs, out_3d, x_3d, N);
            BENCHMARK(Laplacian::interior_3d_nested, output_base, runs, out_3d, x_3d, N);
            BENCHMARK(Laplacian::interior_3d_nested_simd, output_base, runs, out_3d, x_3d, N);
            BENCHMARK(Laplacian::interior_3d_nested_constexpr, output_base, runs, out_3d, x_3d, N);
            BENCHMARK(Laplacian::interior_3d_nested_constexpr_simd, output_base, runs, out_3d, x_3d, N);
        }

        // 1D std::vector<float>
        {
            std::vector<float> x_1d(N*N*N), out_1d(N*N*N);
            Run::initialise(x_1d, N);
            BENCHMARK(Laplacian::interior_1d_flat, output_base, runs, out_1d, x_1d, N);
            BENCHMARK(Laplacian::interior_1d_flat_simd, output_base, runs, out_1d, x_1d, N);
            BENCHMARK(Laplacian::interior_1d_nested, output_base, runs, out_1d, x_1d, N);
            BENCHMARK(Laplacian::interior_1d_nested_simd, output_base, runs, out_1d, x_1d, N);
            BENCHMARK(Laplacian::modulo_1d_flat, output_base, runs, out_1d, x_1d, N);
            BENCHMARK(Laplacian::modulo_1d_flat_simd, output_base, runs, out_1d, x_1d, N);
        }

        // 1D float* via malloc
        {
            float* x_1d_malloc = static_cast<float*>(std::malloc(N * N * N * sizeof(float)));
            float* out_1d_malloc = static_cast<float*>(std::malloc(N * N * N * sizeof(float)));
            Run::initialise(x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_simd, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx32, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx32_simd, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64_simd, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64promotion, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64promotion_simd, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max64_idx64, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max64_idx64_simd, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max64_idx32, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_i32_max64_idx32_simd, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_constexpr, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            BENCHMARK(Laplacian::interior_1d_malloc_nested_constexpr_simd, output_base, runs, out_1d_malloc, x_1d_malloc, N);
            std::free(x_1d_malloc);
            std::free(out_1d_malloc);
        }

        // 1D float* via std::aligned_alloc
        {
            float* x_1d_aligned = static_cast<float*>(std::aligned_alloc(64, N*N*N*sizeof(float)));
            float* out_1d_aligned = static_cast<float*>(std::aligned_alloc(64, N*N*N*sizeof(float)));
            Run::initialise(x_1d_aligned, N);
            BENCHMARK(Laplacian::interior_1d_aligned_nested, output_base, runs, out_1d_aligned, x_1d_aligned, N);
            std::free(x_1d_aligned);
            std::free(out_1d_aligned);
        }
    }

    return EXIT_SUCCESS;
}
