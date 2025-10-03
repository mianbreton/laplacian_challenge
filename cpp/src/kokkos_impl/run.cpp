#include <Kokkos_Core.hpp>
#include <iostream>
#include <execution>
#include <fstream>   
#include <numeric> 

#include "CLI/CLI.hpp"
#include "laplacian.h"
#include "run.h"

#define BENCHMARK(func, output_base, runs, ...) \
    benchmark(strip_namespace(#func), output_base, runs, [&]() { func(__VA_ARGS__); })

std::string get_executable_name(const char* argv0) {
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
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = (std::chrono::high_resolution_clock::now() - start);
    for (uint32_t i(0); i < runs; i++)
    {
        start = std::chrono::high_resolution_clock::now();
        func();
        Kokkos::fence();
        elapsed = (std::chrono::high_resolution_clock::now() - start);
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

int main(int argc, char* argv[]) {
    
    CLI::App app{"Laplacian Benchmarking"};

    std::string basename;
    std::vector<int> ncells;
    int runs = 10;
    // Multiple values allowed
    app.add_option("-b,--basename", basename, "Output file prefix")->required();
    app.add_option("--ncells", ncells, "Number of cells along each dimension")->required()->check(CLI::PositiveNumber);
    app.add_option("--runs", runs, "Number of repetitions for timing")->check(CLI::PositiveNumber);

    CLI11_PARSE(app, argc, argv);  // Parse the arguments

    std::string output = get_executable_name(argv[0]);
    output.erase(0,10); 

    Kokkos::initialize(argc, argv);
    {
        for (const int N: ncells)
        {
            std::cout<<"\n N = "<<N<<"\n";
            std::string output_base = std::string() + basename + "_ncells1d_" + std::to_string(N) + "_" + output;
            // 1D View
            {
                Kokkos::View<float*> x_1d("x_1d", N*N*N), out_1d("out_1d", N*N*N);
                Run::initialise_1d(x_1d, N);
                Run::initialise_1d_zero(out_1d, N);

                BENCHMARK(Laplacian::interior_1d_flat, output_base, runs, out_1d, x_1d, N);
                if (N == 32) Run::check_interior(out_1d, N);
                BENCHMARK(Laplacian::interior_1d_mdrange, output_base, runs, out_1d, x_1d, N);
                if (N == 32) Run::check_interior(out_1d, N);
                BENCHMARK(Laplacian::interior_1d_mdrange_i32_idx32, output_base, runs, out_1d, x_1d, N);
                if (N == 32) Run::check_interior(out_1d, N);
                BENCHMARK(Laplacian::interior_1d_mdrange_i32_idx64, output_base, runs, out_1d, x_1d, N);
                if (N == 32) Run::check_interior(out_1d, N);
                BENCHMARK(Laplacian::interior_1d_mdrange_i32_idx64promotion, output_base, runs, out_1d, x_1d, N);
                if (N == 32) Run::check_interior(out_1d, N);
                BENCHMARK(Laplacian::interior_1d_mdrange_i64_idx32, output_base, runs, out_1d, x_1d, N);
                if (N == 32) Run::check_interior(out_1d, N);
                BENCHMARK(Laplacian::modulo_1d_flat, output_base, runs, out_1d, x_1d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_1d, N); 
                    Run::check_exterior(out_1d, N);
                }
                BENCHMARK(Laplacian::modulo_1d_mdrange, output_base, runs, out_1d, x_1d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_1d, N); 
                    Run::check_exterior(out_1d, N);
                }
                BENCHMARK(Laplacian::conditional_add_1d_flat, output_base, runs, out_1d, x_1d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_1d, N); 
                    Run::check_exterior(out_1d, N);
                }
                BENCHMARK(Laplacian::conditional_add_1d_mdrange, output_base, runs, out_1d, x_1d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_1d, N); 
                    Run::check_exterior(out_1d, N);
                }
                BENCHMARK(Laplacian::ternary_1d_flat, output_base, runs, out_1d, x_1d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_1d, N); 
                    Run::check_exterior(out_1d, N);
                }
                BENCHMARK(Laplacian::ternary_1d_mdrange, output_base, runs, out_1d, x_1d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_1d, N); 
                    Run::check_exterior(out_1d, N);
                }
            }
            // 3D View
            {
                Kokkos::View<float***> x_3d("x_3d", N,N,N), out_3d("out_3d", N,N,N);
                Run::initialise_3d(x_3d, N);
                Run::initialise_3d_zero(out_3d, N);

                BENCHMARK(Laplacian::interior_3d_flat, output_base, runs, out_3d, x_3d, N);
                if (N == 32) Run::check_interior(out_3d, N);
                BENCHMARK(Laplacian::interior_3d_mdrange, output_base, runs, out_3d, x_3d, N);
                if (N == 32) Run::check_interior(out_3d, N);
                BENCHMARK(Laplacian::modulo_3d_flat, output_base, runs, out_3d, x_3d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_3d, N); 
                    Run::check_exterior(out_3d, N);
                }
                BENCHMARK(Laplacian::modulo_3d_mdrange, output_base, runs, out_3d, x_3d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_3d, N); 
                    Run::check_exterior(out_3d, N);
                }
                BENCHMARK(Laplacian::conditional_add_3d_flat, output_base, runs, out_3d, x_3d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_3d, N); 
                    Run::check_exterior(out_3d, N);
                }
                BENCHMARK(Laplacian::conditional_add_3d_mdrange, output_base, runs, out_3d, x_3d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_3d, N); 
                    Run::check_exterior(out_3d, N);
                }
                BENCHMARK(Laplacian::ternary_3d_flat, output_base, runs, out_3d, x_3d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_3d, N); 
                    Run::check_exterior(out_3d, N);
                }
                BENCHMARK(Laplacian::ternary_3d_mdrange, output_base, runs, out_3d, x_3d, N);
                if (N == 32)
                {   
                    Run::check_interior(out_3d, N); 
                    Run::check_exterior(out_3d, N);
                }
            }
        }
    }
    Kokkos::finalize();
    return EXIT_SUCCESS;
}
