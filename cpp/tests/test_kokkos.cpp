#include <Kokkos_Core.hpp>
#include <iostream>
#include <execution>
#include <fstream>   
#include <numeric> 

#include "../src/kokkos_impl/laplacian.h"
#include "../src/kokkos_impl/run.h"
#include "test_kokkos_utils.h"

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

int main(int argc, char* argv[]) {
    
    Kokkos::initialize(argc, argv);
    {
        constexpr uint32_t N = 32; // Number of cells along each dimension
        // 1D View
        {
            Kokkos::View<float*> x_1d("x_1d", N*N*N), out_1d("out_1d", N*N*N);
            Run::initialise(x_1d, N);
            BENCHMARK(Laplacian::interior_1d_flat, out_1d, x_1d, N);
            BENCHMARK(Laplacian::interior_1d_mdrange, out_1d, x_1d, N);
            BENCHMARK(Laplacian::interior_1d_mdrange_i32_idx32, out_1d, x_1d, N);
            BENCHMARK(Laplacian::interior_1d_mdrange_i32_idx64, out_1d, x_1d, N);
            BENCHMARK(Laplacian::interior_1d_mdrange_i32_idx64promotion, out_1d, x_1d, N);
            BENCHMARK(Laplacian::interior_1d_mdrange_i64_idx32, out_1d, x_1d, N);
            BENCHMARK(Laplacian::modulo_1d_flat, out_1d, x_1d, N);
            BENCHMARK(Laplacian::modulo_1d_mdrange, out_1d, x_1d, N);
            BENCHMARK(Laplacian::conditional_add_1d_flat, out_1d, x_1d, N);
            BENCHMARK(Laplacian::conditional_add_1d_mdrange, out_1d, x_1d, N);
            BENCHMARK(Laplacian::ternary_1d_flat, out_1d, x_1d, N);
            BENCHMARK(Laplacian::ternary_1d_mdrange, out_1d, x_1d, N);
        }
        // 3D View
        {
            Kokkos::View<float***> x_3d("x_3d", N,N,N), out_3d("out_3d", N,N,N);
            Run::initialise(x_3d, N);
            BENCHMARK(Laplacian::interior_3d_flat, out_3d, x_3d, N);
            BENCHMARK(Laplacian::interior_3d_mdrange, out_3d, x_3d, N);
            BENCHMARK(Laplacian::modulo_3d_flat, out_3d, x_3d, N);
            BENCHMARK(Laplacian::modulo_3d_mdrange, out_3d, x_3d, N);
            BENCHMARK(Laplacian::conditional_add_3d_flat, out_3d, x_3d, N);
            BENCHMARK(Laplacian::conditional_add_3d_mdrange, out_3d, x_3d, N);
            BENCHMARK(Laplacian::ternary_3d_flat, out_3d, x_3d, N);
            BENCHMARK(Laplacian::ternary_3d_mdrange, out_3d, x_3d, N);
        }
    }
    Kokkos::finalize();
    return EXIT_SUCCESS;
}
