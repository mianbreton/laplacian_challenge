#include <Kokkos_Core.hpp>
#include <iostream>
#include <execution>
#include <fstream>   
#include <numeric> 

#include "laplacian.h"


std::string get_executable_name(const char* argv0) {
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
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " output_base.txt\n";
        return EXIT_FAILURE;
    }
    std::string output = get_executable_name(argv[0]);
    output.erase(0,10); 

    Kokkos::initialize(argc, argv);
    {
        constexpr std::array<size_t, 4> N_array = {32, 64, 128, 256};
        for (size_t N : N_array)
        {
            std::cout<<"\n N = "<<N<<"\n";
            std::string output_base = std::string() + argv[1] + "_ncells1d_" + std::to_string(N) + "_" + output;
            // 1D View
            {
                Kokkos::View<float*> x_1d("x_1d", N*N*N), out_1d("out_1d", N*N*N);
                Laplacians::initialise_1d(x_1d, N);
                Laplacians::initialise_1d_zero(out_1d, N);

                benchmark("interior_1d_flat", output_base, [&](){Laplacians::interior_1d_flat(out_1d, x_1d, N);});
                if (N == 32) Laplacians::check_interior(out_1d, N);
                benchmark("interior_1d_mdrange", output_base, [&](){Laplacians::interior_1d_mdrange(out_1d, x_1d, N);});
                if (N == 32) Laplacians::check_interior(out_1d, N);
                benchmark("interior_1d_mdrange_i32_idx32", output_base, [&](){Laplacians::interior_1d_mdrange_i32_idx32(out_1d, x_1d, N);});
                if (N == 32) Laplacians::check_interior(out_1d, N);
                benchmark("interior_1d_mdrange_i32_idx64", output_base, [&](){Laplacians::interior_1d_mdrange_i32_idx64(out_1d, x_1d, N);});
                if (N == 32) Laplacians::check_interior(out_1d, N);
                benchmark("interior_1d_mdrange_i32_idx64promotion", output_base, [&](){Laplacians::interior_1d_mdrange_i32_idx64promotion(out_1d, x_1d, N);});
                if (N == 32) Laplacians::check_interior(out_1d, N);
                benchmark("interior_1d_mdrange_i64_idx32", output_base, [&](){Laplacians::interior_1d_mdrange_i64_idx32(out_1d, x_1d, N);});
                if (N == 32) Laplacians::check_interior(out_1d, N);
                benchmark("modulo_1d_flat", output_base, [&](){Laplacians::modulo_1d_flat(out_1d, x_1d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_1d, N); 
                    Laplacians::check_exterior(out_1d, N);
                }
                benchmark("modulo_1d_mdrange", output_base, [&](){Laplacians::modulo_1d_mdrange(out_1d, x_1d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_1d, N); 
                    Laplacians::check_exterior(out_1d, N);
                }
                benchmark("conditional_add_1d_flat", output_base, [&](){Laplacians::conditional_add_1d_flat(out_1d, x_1d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_1d, N); 
                    Laplacians::check_exterior(out_1d, N);
                }
                benchmark("conditional_add_1d_mdrange", output_base, [&](){Laplacians::conditional_add_1d_mdrange(out_1d, x_1d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_1d, N); 
                    Laplacians::check_exterior(out_1d, N);
                }
                benchmark("ternary_1d_flat", output_base, [&](){Laplacians::ternary_1d_flat(out_1d, x_1d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_1d, N); 
                    Laplacians::check_exterior(out_1d, N);
                }
                benchmark("ternary_1d_mdrange", output_base, [&](){Laplacians::ternary_1d_mdrange(out_1d, x_1d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_1d, N); 
                    Laplacians::check_exterior(out_1d, N);
                }
            }
            // 3D View
            {
                Kokkos::View<float***> x_3d("x_3d", N,N,N), out_3d("out_3d", N,N,N);
                Laplacians::initialise_3d(x_3d, N);
                Laplacians::initialise_3d_zero(out_3d, N);

                benchmark("interior_3d_flat", output_base, [&](){Laplacians::interior_3d_flat(out_3d, x_3d, N);});
                if (N == 32) Laplacians::check_interior(out_3d, N);
                benchmark("interior_3d_mdrange", output_base, [&](){Laplacians::interior_3d_mdrange(out_3d, x_3d, N);});
                if (N == 32) Laplacians::check_interior(out_3d, N);
                benchmark("modulo_3d_flat", output_base, [&](){Laplacians::modulo_3d_flat(out_3d, x_3d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_3d, N); 
                    Laplacians::check_exterior(out_3d, N);
                }
                benchmark("modulo_3d_mdrange", output_base, [&](){Laplacians::modulo_3d_mdrange(out_3d, x_3d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_3d, N); 
                    Laplacians::check_exterior(out_3d, N);
                }
                benchmark("conditional_add_3d_flat", output_base, [&](){Laplacians::conditional_add_3d_flat(out_3d, x_3d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_3d, N); 
                    Laplacians::check_exterior(out_3d, N);
                }
                benchmark("conditional_add_3d_mdrange", output_base, [&](){Laplacians::conditional_add_3d_mdrange(out_3d, x_3d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_3d, N); 
                    Laplacians::check_exterior(out_3d, N);
                }
                benchmark("ternary_3d_flat", output_base, [&](){Laplacians::ternary_3d_flat(out_3d, x_3d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_3d, N); 
                    Laplacians::check_exterior(out_3d, N);
                }
                benchmark("ternary_3d_mdrange", output_base, [&](){Laplacians::ternary_3d_mdrange(out_3d, x_3d, N);});
                if (N == 32)
                {   
                    Laplacians::check_interior(out_3d, N); 
                    Laplacians::check_exterior(out_3d, N);
                }
            }
        }
    }
    Kokkos::finalize();
    return EXIT_SUCCESS;
}
