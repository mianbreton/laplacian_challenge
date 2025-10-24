#include <gtest/gtest.h>
#include <vector>
#include <cstdlib>
#include <string>

#include "../src/native_impl/run.h"
#include "../src/native_impl/laplacian.h"
#include "test_native_utils.h"

constexpr uint32_t N = 32;

//--------------------------------------
// CHECK
//--------------------------------------
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
    if (name.find("interior") == std::string::npos)
        Testing::check_exterior(out, N);
}

//--------------------------------------
// 3D kernels
//--------------------------------------
TEST(NativeLaplacian, ThreeDimensionalKernels)
{
    std::vector<std::vector<std::vector<float>>> x(N,
        std::vector<std::vector<float>>(N, std::vector<float>(N, 0.f)));
    std::vector<std::vector<std::vector<float>>> out = x;

    Run::initialise(x, N);

    CHECK(Laplacian::modulo_3d_nested, out, x, N);
    CHECK(Laplacian::modulo_3d_nested_simd, out, x, N);
    CHECK(Laplacian::conditional_add_3d_nested, out, x, N);
    CHECK(Laplacian::conditional_add_3d_nested_simd, out, x, N);
    CHECK(Laplacian::ternary_3d_nested_simd, out, x, N);
    CHECK(Laplacian::ternary_3d_nested, out, x, N);
    CHECK(Laplacian::interior_3d_flat, out, x, N);
    CHECK(Laplacian::interior_3d_flat_simd, out, x, N);
    CHECK(Laplacian::interior_3d_nested, out, x, N);
    CHECK(Laplacian::interior_3d_nested_simd, out, x, N);
    CHECK(Laplacian::interior_3d_nested_constexpr, out, x, N);
    CHECK(Laplacian::interior_3d_nested_constexpr_simd, out, x, N);
}

//--------------------------------------
// 1D std::vector kernels
//--------------------------------------
TEST(NativeLaplacian, OneDimensionalVectorKernels)
{
    std::vector<float> x(N * N * N), out(N * N * N);
    Run::initialise(x, N);

    CHECK(Laplacian::interior_1d_flat, out, x, N);
    CHECK(Laplacian::interior_1d_flat_simd, out, x, N);
    CHECK(Laplacian::interior_1d_nested, out, x, N);
    CHECK(Laplacian::interior_1d_nested_simd, out, x, N);
    CHECK(Laplacian::modulo_1d_flat, out, x, N);
    CHECK(Laplacian::modulo_1d_flat_simd, out, x, N);
}

//--------------------------------------
// malloc-based float* kernels
//--------------------------------------
TEST(NativeLaplacian, OneDimensionalMallocKernels)
{
    float* x = static_cast<float*>(std::malloc(N * N * N * sizeof(float)));
    float* out = static_cast<float*>(std::malloc(N * N * N * sizeof(float)));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(out, nullptr);

    Run::initialise(x, N);

    CHECK(Laplacian::interior_1d_malloc_nested, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_simd, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_i32_max32_idx32, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_i32_max32_idx32_simd, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64_simd, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64promotion, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_i32_max32_idx64promotion_simd, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_i32_max64_idx64, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_i32_max64_idx64_simd, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_i32_max64_idx32, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_i32_max64_idx32_simd, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_constexpr, out, x, N);
    CHECK(Laplacian::interior_1d_malloc_nested_constexpr_simd, out, x, N);

    std::free(x);
    std::free(out);
}

TEST(NativeLaplacian, OneDimensionalAlignedKernels)
{
    float* x = static_cast<float*>(std::aligned_alloc(64, N*N*N*sizeof(float)));
    float* out = static_cast<float*>(std::aligned_alloc(64, N*N*N*sizeof(float)));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(out, nullptr);

    Run::initialise(x, N);
    CHECK(Laplacian::interior_1d_aligned_nested, out, x, N);

    std::free(x);
    std::free(out);
}
