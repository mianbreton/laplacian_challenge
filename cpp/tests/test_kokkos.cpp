#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <string>
#include <numeric>

#include "../src/kokkos_impl/laplacian.h"
#include "../src/kokkos_impl/run.h"
#include "test_kokkos_utils.h"

#define CHECK(func, out, x, N) \
    check_kernel(strip_namespace(#func), out, N, [&]() { func(out, x, N); })

std::string strip_namespace(const std::string& full_name)
{
    size_t pos = full_name.rfind("::");
    return (pos == std::string::npos) ? full_name : full_name.substr(pos + 2);
}

template <typename Func, class Array>
void check_kernel(const std::string& name, Array& out, const int N, Func&& func)
{
    Run::initialise_zero(out, N);
    func();
    Testing::check_interior(out, N);
    if (name.rfind("interior") == std::string::npos)
        Testing::check_exterior(out, N);
}

class LaplacianTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        int argc = 0;
        Kokkos::initialize(argc, nullptr);
    }

    static void TearDownTestSuite() {
        Kokkos::finalize();
    }
};

//-------------------------------------------------------------
// Actual tests
//-------------------------------------------------------------
TEST_F(LaplacianTest, OneDimensionalKernels)
{
    constexpr uint32_t N = 32;
    Kokkos::View<float*> x_1d("x_1d", N*N*N), out_1d("out_1d", N*N*N);
    Run::initialise(x_1d, N);

    CHECK(Laplacian::interior_1d_flat, out_1d, x_1d, N);
    CHECK(Laplacian::interior_1d_mdrange, out_1d, x_1d, N);
    CHECK(Laplacian::interior_1d_mdrange_i32_idx32, out_1d, x_1d, N);
    CHECK(Laplacian::interior_1d_mdrange_i32_idx64, out_1d, x_1d, N);
    CHECK(Laplacian::interior_1d_mdrange_i32_idx64promotion, out_1d, x_1d, N);
    CHECK(Laplacian::interior_1d_mdrange_i64_idx32, out_1d, x_1d, N);
    CHECK(Laplacian::modulo_1d_flat, out_1d, x_1d, N);
    CHECK(Laplacian::modulo_1d_mdrange, out_1d, x_1d, N);
    CHECK(Laplacian::conditional_add_1d_flat, out_1d, x_1d, N);
    CHECK(Laplacian::conditional_add_1d_mdrange, out_1d, x_1d, N);
    CHECK(Laplacian::ternary_1d_flat, out_1d, x_1d, N);
    CHECK(Laplacian::ternary_1d_mdrange, out_1d, x_1d, N);
}

TEST_F(LaplacianTest, ThreeDimensionalKernels)
{
    constexpr uint32_t N = 32;
    Kokkos::View<float***> x_3d("x_3d", N, N, N), out_3d("out_3d", N, N, N);
    Run::initialise(x_3d, N);

    CHECK(Laplacian::interior_3d_flat, out_3d, x_3d, N);
    CHECK(Laplacian::interior_3d_mdrange, out_3d, x_3d, N);
    CHECK(Laplacian::modulo_3d_flat, out_3d, x_3d, N);
    CHECK(Laplacian::modulo_3d_mdrange, out_3d, x_3d, N);
    CHECK(Laplacian::conditional_add_3d_flat, out_3d, x_3d, N);
    CHECK(Laplacian::conditional_add_3d_mdrange, out_3d, x_3d, N);
    CHECK(Laplacian::ternary_3d_flat, out_3d, x_3d, N);
    CHECK(Laplacian::ternary_3d_mdrange, out_3d, x_3d, N);
}
