#include <iostream>
#include <vector>
#include <cstdint>
#include <omp.h>
#include <cmath>
#include <cassert>



class Native {
    public:

    static void check_interior(const std::vector<float>& x, const size_t N);
    static void check_interior(const float* x, const size_t N);
    static void check_interior(const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void check_exterior(const std::vector<float>& x, const size_t N);
    static void check_exterior(const float* x, const size_t N);
    static void check_exterior(const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void modulo_1d_flat (std::vector<float>& out, const std::vector<float>& x, const size_t N);
    static void modulo_1d_flat_simd (std::vector<float>& out, const std::vector<float>& x, const size_t N);
    static void modulo_3d_nested (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void modulo_3d_nested_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void conditional_add_3d_nested (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void conditional_add_3d_nested_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void ternary_3d_nested (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void ternary_3d_nested_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void interior_3d_flat (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void interior_3d_flat_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void interior_3d_nested (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void interior_3d_nested_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void run_interior_3d_nested_constexpr (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    static void run_interior_3d_nested_constexpr_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N);
    template <size_t N> static void interior_3d_nested_constexpr (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x);
    template <size_t N> static void interior_3d_nested_constexpr_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x);
    static void interior_1d_flat(std::vector<float>& out, const std::vector<float>& x, size_t N);
    static void interior_1d_flat_simd(std::vector<float>& out, const std::vector<float>& x, size_t N);
    static void interior_1d_nested(std::vector<float>& out, const std::vector<float>& x, size_t N);
    static void interior_1d_nested_simd(std::vector<float>& out, const std::vector<float>& x, size_t N);
    static void interior_1d_malloc_nested(float* out, const float* x, size_t N);
    static void interior_1d_malloc_nested_simd(float* out, const float* x, size_t N);
    static void interior_1d_malloc_nested_i32_max32_idx32(float* out, const float* x, size_t N);
    static void interior_1d_malloc_nested_i32_max32_idx32_simd(float* out, const float* x, size_t N);
    static void interior_1d_malloc_nested_i32_max32_idx64(float* out, const float* x, size_t N);
    static void interior_1d_malloc_nested_i32_max32_idx64_simd(float* out, const float* x, size_t N);
    static void interior_1d_malloc_nested_i32_max32_idx64promotion(float* out, const float* x, size_t N);
    static void interior_1d_malloc_nested_i32_max32_idx64promotion_simd(float* out, const float* x, size_t N);
    static void interior_1d_malloc_nested_i32_max64_idx64(float* out, const float* x, size_t N);
    static void interior_1d_malloc_nested_i32_max64_idx64_simd(float* out, const float* x, size_t N);
    static void interior_1d_malloc_nested_i32_max64_idx32(float* out, const float* x, size_t N);
    static void interior_1d_malloc_nested_i32_max64_idx32_simd(float* out, const float* x, size_t N);
    static void run_interior_1d_malloc_nested_constexpr(float* out, const float* x, size_t N);
    static void run_interior_1d_malloc_nested_constexpr_simd(float* out, const float* x, size_t N);
    template <size_t N> static void interior_1d_malloc_nested_constexpr(float* out, const float* x);
    template <size_t N> static void interior_1d_malloc_nested_constexpr_simd(float* out, const float* x);
    static void interior_1d_aligned_nested(float* out, const float* x, size_t N);
};

void Native::run_interior_1d_malloc_nested_constexpr(float* out, const float* x, const size_t N) 
{
    switch (N)
    {
        case 32:
            interior_1d_malloc_nested_constexpr<32>(out, x);
            break;
        case 64:
            interior_1d_malloc_nested_constexpr<64>(out, x);
            break;
        case 128:
            interior_1d_malloc_nested_constexpr<128>(out, x);
            break;
        case 256:
            interior_1d_malloc_nested_constexpr<256>(out, x);
            break;
        case 512:
            interior_1d_malloc_nested_constexpr<512>(out, x);
            break;
        case 1024:
            interior_1d_malloc_nested_constexpr<1024>(out, x);
            break;
        case 2048:
            interior_1d_malloc_nested_constexpr<2048>(out, x);
            break;
        default:
            break;
    }
}
void Native::run_interior_1d_malloc_nested_constexpr_simd(float* out, const float* x, const size_t N) 
{
    switch (N)
    {
        case 32:
            interior_1d_malloc_nested_constexpr_simd<32>(out, x);
            break;
        case 64:
            interior_1d_malloc_nested_constexpr_simd<64>(out, x);
            break;
        case 128:
            interior_1d_malloc_nested_constexpr_simd<128>(out, x);
            break;
        case 256:
            interior_1d_malloc_nested_constexpr_simd<256>(out, x);
            break;
        case 512:
            interior_1d_malloc_nested_constexpr_simd<512>(out, x);
            break;
        case 1024:
            interior_1d_malloc_nested_constexpr_simd<1024>(out, x);
            break;
        case 2048:
            interior_1d_malloc_nested_constexpr_simd<2048>(out, x);
            break;
        default:
            break;
    }
}

void Native::run_interior_3d_nested_constexpr(std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N) 
{
    switch (N)
    {
        case 32:
            interior_3d_nested_constexpr<32>(out, x);
            break;
        case 64:
            interior_3d_nested_constexpr<64>(out, x);
            break;
        case 128:
            interior_3d_nested_constexpr<128>(out, x);
            break;
        case 256:
            interior_3d_nested_constexpr<256>(out, x);
            break;
        case 512:
            interior_3d_nested_constexpr<512>(out, x);
            break;
        case 1024:
            interior_3d_nested_constexpr<1024>(out, x);
            break;
        default:
            break;
    }
}
void Native::run_interior_3d_nested_constexpr_simd(std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N) 
{
    switch (N)
    {
        case 32:
            interior_3d_nested_constexpr_simd<32>(out, x);
            break;
        case 64:
            interior_3d_nested_constexpr_simd<64>(out, x);
            break;
        case 128:
            interior_3d_nested_constexpr_simd<128>(out, x);
            break;
        case 256:
            interior_3d_nested_constexpr_simd<256>(out, x);
            break;
        case 512:
            interior_3d_nested_constexpr_simd<512>(out, x);
            break;
        case 1024:
            interior_3d_nested_constexpr_simd<1024>(out, x);
            break;
        default:
            break;
    }
}

void assertion(const float value, const float expected)
{
    const bool isGood = (fabsf(value - expected) < 1e-7f);
    assert(isGood);
}

void Native::check_interior(const std::vector<float>& x, const size_t N)
{
    printf("Check interior\n");
    const size_t N2 = N*N;
    const size_t Nm1 = N - 1ul;
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i*N2 + j*N + k], 0.f);
    }
}
void Native::check_interior(const float* x, const size_t N)
{
    printf("Check interior\n");
    const size_t N2 = N*N;
    const size_t Nm1 = N - 1ul;
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i*N2 + j*N + k], 0.f);
    }
}
void Native::check_interior(const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{   
    printf("Check interior\n");
    const size_t Nm1 = N - 1ul;
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i][j][k], 0.f);
    }
}
void Native::check_exterior(const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    printf("Check exterior\n");
    const size_t Nm1 = N - 1ul;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    // Faces
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i][Nm1][k], -static_cast<float>(N2*N2));
        assertion(x[i][0][k], static_cast<float>(N2*N2));
    }
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1][j][k], -static_cast<float>(N2*N3));
        assertion(x[0][j][k], static_cast<float>(N2*N3));
    }
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[i][j][Nm1], -static_cast<float>(N2*N));
        assertion(x[i][j][0], static_cast<float>(N2*N));
    }
    // Edges 
    for(size_t i=1ul; i < Nm1; i++)
    {
        assertion(x[i][Nm1][Nm1], -static_cast<float>(N2*(N2+N)));
        assertion(x[i][Nm1][0], -static_cast<float>(N2*(N2-N)));
        assertion(x[i][0][Nm1], static_cast<float>(N2*(N2-N)));
        assertion(x[i][0][0], static_cast<float>(N2*(N2+N)));
    }
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[Nm1][j][Nm1], -static_cast<float>(N2*(N3+N)));
        assertion(x[Nm1][j][0], -static_cast<float>(N2*(N3-N)));
        assertion(x[0][j][Nm1], static_cast<float>(N2*(N3-N)));
        assertion(x[0][j][0], static_cast<float>(N2*(N3+N)));
    }
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1][Nm1][k], -static_cast<float>(N2*(N3+N2)));
        assertion(x[Nm1][0][k], -static_cast<float>(N2*(N3-N2)));
        assertion(x[0][Nm1][k], static_cast<float>(N2*(N3-N2)));
        assertion(x[0][0][k], static_cast<float>(N2*(N3+N2)));
    }
    // Corners
    assertion(x[Nm1][Nm1][Nm1] , -static_cast<float>(N2*(N3+N2+N)));
    assertion(x[Nm1][Nm1][0], -static_cast<float>(N2*(N3+N2-N)));
    assertion(x[Nm1][0][Nm1], -static_cast<float>(N2*(N3-N2+N)));
    assertion(x[Nm1][0][0], -static_cast<float>(N2*(N3-N2-N)));
    assertion(x[0][Nm1][Nm1], static_cast<float>(N2*(N3-N2-N)));
    assertion(x[0][Nm1][0], static_cast<float>(N2*(N3-N2+N)));
    assertion(x[0][0][Nm1], static_cast<float>(N2*(N3+N2-N)));
    assertion(x[0][0][0], static_cast<float>(N2*(N3+N2+N)));
}
void Native::check_exterior(const std::vector<float>& x, const size_t N)
{
    printf("Check exterior\n");
    const size_t Nm1 = N - 1ul;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    // Faces
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i*N2 + Nm1*N + k], -static_cast<float>(N2*N2));
        assertion(x[i*N2 + k], static_cast<float>(N2*N2));
    }
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1*N2 + j*N + k], -static_cast<float>(N2*N3));
        assertion(x[j*N + k], static_cast<float>(N2*N3));
    }
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[i*N2 + j*N + Nm1], -static_cast<float>(N2*N));
        assertion(x[i*N2 + j*N], static_cast<float>(N2*N));
    }
    // Edges 
    for(size_t i=1ul; i < Nm1; i++)
    {
        assertion(x[i*N2 + Nm1*N + Nm1], -static_cast<float>(N2*(N2+N)));
        assertion(x[i*N2 + Nm1*N], -static_cast<float>(N2*(N2-N)));
        assertion(x[i*N2 + Nm1], static_cast<float>(N2*(N2-N)));
        assertion(x[i*N2], static_cast<float>(N2*(N2+N)));
    }
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[Nm1*N2 + j*N + Nm1], -static_cast<float>(N2*(N3+N)));
        assertion(x[Nm1*N2 + j*N], -static_cast<float>(N2*(N3-N)));
        assertion(x[j*N + Nm1], static_cast<float>(N2*(N3-N)));
        assertion(x[j*N], static_cast<float>(N2*(N3+N)));
    }
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1*N2 + Nm1*N + k], -static_cast<float>(N2*(N3+N2)));
        assertion(x[Nm1*N2 + k], -static_cast<float>(N2*(N3-N2)));
        assertion(x[Nm1*N + k], static_cast<float>(N2*(N3-N2)));
        assertion(x[k], static_cast<float>(N2*(N3+N2)));
    }
    // Corners
    assertion(x[Nm1*N2 + Nm1*N + Nm1], -static_cast<float>(N2*(N3+N2+N)));
    assertion(x[Nm1*N2 + Nm1*N], -static_cast<float>(N2*(N3+N2-N)));
    assertion(x[Nm1*N2 + Nm1], -static_cast<float>(N2*(N3-N2+N)));
    assertion(x[Nm1*N2], -static_cast<float>(N2*(N3-N2-N)));
    assertion(x[Nm1*N + Nm1], static_cast<float>(N2*(N3-N2-N)));
    assertion(x[Nm1*N], static_cast<float>(N2*(N3-N2+N)));
    assertion(x[Nm1], static_cast<float>(N2*(N3+N2-N)));
    assertion(x[0], static_cast<float>(N2*(N3+N2+N)));
}
void Native::check_exterior(const float* x, const size_t N)
{
    printf("Check exterior\n");
    const size_t Nm1 = N - 1ul;
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    // Faces
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[i*N2 + Nm1*N + k], -static_cast<float>(N2*N2));
        assertion(x[i*N2 + k], static_cast<float>(N2*N2));
    }
    for(size_t j=1ul; j < Nm1; j++)
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1*N2 + j*N + k], -static_cast<float>(N2*N3));
        assertion(x[j*N + k], static_cast<float>(N2*N3));
    }
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[i*N2 + j*N + Nm1], -static_cast<float>(N2*N));
        assertion(x[i*N2 + j*N], static_cast<float>(N2*N));
    }
    // Edges 
    for(size_t i=1ul; i < Nm1; i++)
    {
        assertion(x[i*N2 + Nm1*N + Nm1], -static_cast<float>(N2*(N2+N)));
        assertion(x[i*N2 + Nm1*N], -static_cast<float>(N2*(N2-N)));
        assertion(x[i*N2 + Nm1], static_cast<float>(N2*(N2-N)));
        assertion(x[i*N2], static_cast<float>(N2*(N2+N)));
    }
    for(size_t j=1ul; j < Nm1; j++)
    {
        assertion(x[Nm1*N2 + j*N + Nm1], -static_cast<float>(N2*(N3+N)));
        assertion(x[Nm1*N2 + j*N], -static_cast<float>(N2*(N3-N)));
        assertion(x[j*N + Nm1], static_cast<float>(N2*(N3-N)));
        assertion(x[j*N], static_cast<float>(N2*(N3+N)));
    }
    for(size_t k=1ul; k < Nm1; k++)
    {
        assertion(x[Nm1*N2 + Nm1*N + k], -static_cast<float>(N2*(N3+N2)));
        assertion(x[Nm1*N2 + k], -static_cast<float>(N2*(N3-N2)));
        assertion(x[Nm1*N + k], static_cast<float>(N2*(N3-N2)));
        assertion(x[k], static_cast<float>(N2*(N3+N2)));
    }
    // Corners
    assertion(x[Nm1*N2 + Nm1*N + Nm1], -static_cast<float>(N2*(N3+N2+N)));
    assertion(x[Nm1*N2 + Nm1*N], -static_cast<float>(N2*(N3+N2-N)));
    assertion(x[Nm1*N2 + Nm1], -static_cast<float>(N2*(N3-N2+N)));
    assertion(x[Nm1*N2], -static_cast<float>(N2*(N3-N2-N)));
    assertion(x[Nm1*N + Nm1], static_cast<float>(N2*(N3-N2-N)));
    assertion(x[Nm1*N], static_cast<float>(N2*(N3-N2+N)));
    assertion(x[Nm1], static_cast<float>(N2*(N3+N2-N)));
    assertion(x[0], static_cast<float>(N2*(N3+N2+N)));
}
void Native::modulo_1d_flat(std::vector<float>& out, const std::vector<float>& x, size_t N) 
{
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    const float invh2 = static_cast<float>(N2);
    #pragma omp parallel for
    for(size_t idx=0ul; idx < N3; idx++)
    {   
        const size_t i = idx / N2;
        const size_t iN2 = i * N2;
        const size_t j = (idx - iN2) / N;
        const size_t jN = j * N;
        const size_t k = idx - iN2 - jN;
        out[idx] =  ( x[(iN2 + jN + (k-1ul)%N)] 
                    + x[(iN2 + jN + (k+1ul)%N)]
                    + x[(iN2 + (j-1ul)%N*N + k)]
                    + x[(iN2 + (j+1ul)%N*N + k)]
                    + x[((i-1ul)%N*N2 + jN + k)]
                    + x[((i+1ul)%N*N2 + jN + k)]
                        - 6.f * x[idx]) * invh2;
    }
}
void Native::modulo_1d_flat_simd(std::vector<float>& out, const std::vector<float>& x, size_t N) 
{
    const size_t N2 = N*N;
    const size_t N3 = N2*N;
    const float invh2 = static_cast<float>(N2);
    #pragma omp parallel for simd
    for(size_t idx=0ul; idx < N3; idx++)
    {   
        const size_t i = idx / N2;
        const size_t iN2 = i * N2;
        const size_t j = (idx - iN2) / N;
        const size_t jN = j * N;
        const size_t k = idx - iN2 - jN;
        out[idx] =  ( x[(iN2 + jN + (k-1ul)%N)] 
                    + x[(iN2 + jN + (k+1ul)%N)]
                    + x[(iN2 + (j-1ul)%N*N + k)]
                    + x[(iN2 + (j+1ul)%N*N + k)]
                    + x[((i-1ul)%N*N2 + jN + k)]
                    + x[((i+1ul)%N*N2 + jN + k)]
                        - 6.f * x[idx]) * invh2;
    }
}

void Native::modulo_3d_nested (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    const float invh2 = N*N;
    #pragma omp parallel for
    for(size_t i=0ul; i < N; i++)
    {
        const size_t im1 = (i - 1ul) % N;
        const size_t ip1 = (i + 1ul) % N;
        for(size_t j(0ul); j < N; j++)
        {
            const size_t jm1 = (j - 1ul) % N;
            const size_t jp1 = (j + 1ul) % N;
            for(size_t k(0ul); k < N; k++)
            {
                const size_t km1 = (k - 1ul) % N;
                const size_t kp1 = (k + 1ul) % N;
                out[i][j][k] =  (x[i][j][km1] + x[i][j][kp1]+ x[i][jm1][k]
                                + x[i][jp1][k]+ x[im1][j][k] + x[ip1][j][k]
                                - 6.f * x[i][j][k]) * invh2;
            }
        }
    }
}
void Native::modulo_3d_nested_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    const float invh2 = N*N;
    #pragma omp parallel for
    for(size_t i=0ul; i < N; i++)
    {
        const size_t im1 = (i - 1ul) % N;
        const size_t ip1 = (i + 1ul) % N;
        for(size_t j(0ul); j < N; j++)
        {
            const size_t jm1 = (j - 1ul) % N;
            const size_t jp1 = (j + 1ul) % N;
            #pragma omp simd
            for(size_t k = 0ul; k < N; k++)
            {
                const size_t km1 = (k - 1ul) % N;
                const size_t kp1 = (k + 1ul) % N;
                out[i][j][k] =  (x[i][j][km1] + x[i][j][kp1] + x[i][jm1][k]
                                + x[i][jp1][k]+ x[im1][j][k] + x[ip1][j][k]
                                - 6.f * x[i][j][k]) * invh2;
            }
        }
    }
}
void Native::conditional_add_3d_nested (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    const float invh2 = N*N;
    const size_t Nm1 = N - 1ul;
    #pragma omp parallel for
    for(size_t i=0ul; i < N; i++)
    {
        const size_t im1 = (i - 1ul) + N*(i == 0ul);
        const size_t ip1 = (i + 1ul) - N*(i == Nm1);
        for(size_t j(0ul); j < N; j++)
        {
            const size_t jm1 = (j - 1ul) + N*(j == 0ul);
            const size_t jp1 = (j + 1ul) - N*(j == Nm1);
            for(size_t k(0ul); k < N; k++)
            {
                const size_t km1 = (k - 1ul) + N*(k == 0ul);
                const size_t kp1 = (k + 1ul) - N*(k == Nm1);
                out[i][j][k] =  (x[i][j][km1] + x[i][j][kp1]+ x[i][jm1][k]
                                + x[i][jp1][k]+ x[im1][j][k] + x[ip1][j][k]
                                - 6.f * x[i][j][k]) * invh2;
            }
        }
    }
}
void Native::conditional_add_3d_nested_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    const float invh2 = N*N;
    const size_t Nm1 = N - 1ul;
    #pragma omp parallel for
    for(size_t i=0ul; i < N; i++)
    {
        const size_t im1 = (i - 1ul) + N*(i == 0ul);
        const size_t ip1 = (i + 1ul) - N*(i == Nm1);
        for(size_t j(0ul); j < N; j++)
        {
            const size_t jm1 = (j - 1ul) + N*(j == 0ul);
            const size_t jp1 = (j + 1ul) - N*(j == Nm1);
            #pragma omp simd
            for(size_t k=0ul; k < N; k++)
            {
                const size_t km1 = (k - 1ul) + N*(k == 0ul);
                const size_t kp1 = (k + 1ul) - N*(k == Nm1);
                out[i][j][k] =  (x[i][j][km1] + x[i][j][kp1]+ x[i][jm1][k]
                                + x[i][jp1][k]+ x[im1][j][k] + x[ip1][j][k]
                                - 6.f * x[i][j][k]) * invh2;
            }
        }
    }
}
void Native::ternary_3d_nested (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    const float invh2 = N*N;
    const size_t Nm1 = N - 1ul;
    #pragma omp parallel for
    for(size_t i=0ul; i < N; i++)
    {
        const size_t im1 = (i == 0ul) ? Nm1 : i - 1ul;
        const size_t ip1 = (i == Nm1) ? 0ul : i + 1ul;
        for(size_t j(0ul); j < N; j++)
        {
            const size_t jm1 = (j  == 0ul) ? Nm1: j - 1ul;
            const size_t jp1 = (j  == Nm1) ? 0ul : j + 1ul;
            for(size_t k(0ul); k < N; k++)
            {
                const size_t km1 = (k == 0ul) ? Nm1 : k - 1ul;
                const size_t kp1 = (k == Nm1) ? 0ul : k + 1ul;
                out[i][j][k] =  (x[i][j][km1] + x[i][j][kp1] + x[i][jm1][k]
                                + x[i][jp1][k]+ x[im1][j][k] + x[ip1][j][k]
                                - 6.f * x[i][j][k]) * invh2;
            }
        }
    }
}
void Native::ternary_3d_nested_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    const float invh2 = N*N;
    const size_t Nm1 = N - 1ul;
    #pragma omp parallel for
    for(size_t i=0ul; i < N; i++)
    {
        const size_t im1 = (i == 0ul) ? Nm1 : i - 1ul;
        const size_t ip1 = (i == Nm1) ? 0ul : i + 1ul;
        for(size_t j(0ul); j < N; j++)
        {
            const size_t jm1 = (j == 0ul) ? Nm1 : j - 1ul;
            const size_t jp1 = (j == Nm1) ? 0ul : j + 1ul;
            #pragma omp simd
            for(size_t k=0ul; k < N; k++)
            {
                const size_t km1 = (k == 0ul) ? Nm1 : k - 1ul;
                const size_t kp1 = (k == Nm1) ? 0ul : k + 1ul;
                out[i][j][k] =  (x[i][j][km1]+ x[i][j][kp1]+ x[i][jm1][k]
                                + x[i][jp1][k]+ x[im1][j][k]+ x[ip1][j][k]
                                - 6.f * x[i][j][k]) * invh2;
            }
        }
    }
}

void Native::interior_3d_flat(std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, size_t N) 
{
    const size_t Ni = N-2ul;
    const size_t N2i = Ni*Ni;
    const size_t N3i = N2i*Ni;
    const size_t N2 = N*N;
    const float invh2 = static_cast<float>(N2);
    #pragma omp parallel for
    for(size_t idx=0ul; idx < N3i; idx++)
    {   
        size_t i = idx / N2i;
        size_t j = (idx - N2i*i) / Ni;
        size_t k = idx - N2i*i - Ni*j;
        i++; j++; k++;
        out[i][j][k] =  ( x[i][j][k-1ul] + x[i][j][k+1ul] + x[i][j-1ul][k]
                        + x[i][j+1ul][k] + x[i-1ul][j][k] + x[i+1ul][j][k]
                        - 6.f * x[i][j][k]) * invh2;
    }
}
void Native::interior_3d_flat_simd(std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, size_t N) 
{
    const size_t Ni = N-2ul;
    const size_t N2i = Ni*Ni;
    const size_t N3i = N2i*Ni;
    const size_t N2 = N*N;
    const float invh2 = static_cast<float>(N2);
    #pragma omp parallel for simd
    for(size_t idx=0ul; idx < N3i; idx++)
    {   
        size_t i = idx / N2i;
        size_t j = (idx - N2i*i) / Ni;
        size_t k = idx - N2i*i - Ni*j;
        i++; j++; k++;
        out[i][j][k] =  ( x[i][j][k-1ul] + x[i][j][k+1ul] + x[i][j-1ul][k]
                        + x[i][j+1ul][k] + x[i-1ul][j][k] + x[i+1ul][j][k]
                        - 6.f * x[i][j][k]) * invh2;
    }
}
void Native::interior_3d_nested (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    const float invh2 = N*N;
    const size_t Nm1 = N-1ul;
    #pragma omp parallel for
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j(1ul); j < Nm1; j++)
    for(size_t k(1ul); k < Nm1; k++)
    {
        out[i][j][k] =  ( x[i][j][k-1ul] + x[i][j][k+1ul] + x[i][j-1ul][k]
                        + x[i][j+1ul][k] + x[i-1ul][j][k] + x[i+1ul][j][k]
                        - 6.f * x[i][j][k]) * invh2;
    }
}

void Native::interior_3d_nested_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x, const size_t N)
{
    const float invh2 = N*N;
    const size_t Nm1 = N-1ul;
    #pragma omp parallel for
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j(1ul); j < Nm1; j++)
    #pragma omp simd
    for(size_t k = 1ul; k < Nm1; k++)
    {
        out[i][j][k] =  (x[i][j][k-1ul]+ x[i][j][k+1ul]+ x[i][j-1ul][k]
                        + x[i][j+1ul][k]+ x[i-1ul][j][k]+ x[i+1ul][j][k]
                        - 6.f * x[i][j][k]) * invh2;
    }
}
template<size_t N>
void Native::interior_3d_nested_constexpr (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x)
{
    constexpr float invh2 = N*N;
    constexpr size_t Nm1 = N-1ul;
    #pragma omp parallel for
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j(1ul); j < Nm1; j++)
    for(size_t k(1ul); k < Nm1; k++)
    {
        out[i][j][k] =  ( x[i][j][k-1ul] + x[i][j][k+1ul] + x[i][j-1ul][k]
                        + x[i][j+1ul][k] + x[i-1ul][j][k] + x[i+1ul][j][k]
                        - 6.f * x[i][j][k]) * invh2;
    }
}
template<size_t N>
void Native::interior_3d_nested_constexpr_simd (std::vector<std::vector<std::vector<float>>>& out, const std::vector<std::vector<std::vector<float>>>& x)
{
    constexpr float invh2 = N*N;
    constexpr size_t Nm1 = N-1ul;
    #pragma omp parallel for    
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j=(1ul); j < Nm1; j++)
    #pragma omp simd
    for(size_t k=1ul; k < Nm1; k++)
    {
        out[i][j][k] =  ( x[i][j][k-1ul] + x[i][j][k+1ul] + x[i][j-1ul][k]
                        + x[i][j+1ul][k] + x[i-1ul][j][k] + x[i+1ul][j][k]
                        - 6.f * x[i][j][k]) * invh2;
    }
}

void Native::interior_1d_flat(std::vector<float>& out, const std::vector<float>& x, size_t N) 
{
    const size_t Ni = N-2ul;
    const size_t N2i = Ni*Ni;
    const size_t N3i = N2i*Ni;
    const size_t N2 = N*N;
    const float invh2 = static_cast<float>(N2);
    #pragma omp parallel for
    for(size_t idx=0ul; idx < N3i; idx++)
    {   
        const size_t i = idx / N2i;
        const size_t j = (idx - N2i*i) / Ni;
        const size_t k = idx - N2i*i - Ni*j;
        const size_t idx2 = (i+1ul) * N2 + (j+1ul) * N + k + 1ul;
        out[idx2] =  ( x[idx2-1ul] + x[idx2+1ul] + x[idx2-N]
                    + x[idx2+N] + x[idx2-N2]+ x[idx2+N2]
                    - 6.f * x[idx2]) * invh2;
    }
}
void Native::interior_1d_flat_simd(std::vector<float>& out, const std::vector<float>& x, size_t N) 
{
    const size_t Ni = N-2ul;
    const size_t N2i = Ni*Ni;
    const size_t N3i = N2i*Ni;
    const size_t N2 = N*N;
    const float invh2 = static_cast<float>(N2);
    #pragma omp parallel for simd
    for(size_t idx=0ul; idx < N3i; idx++)
    {   
        const size_t i = idx / N2i;
        const size_t j = (idx - N2i*i) / Ni;    
        const size_t k = idx - N2i*i - Ni*j;
        const size_t idx2 = (i+1ul) * N2 + (j+1ul) * N + k + 1ul;
        out[idx2] =  ( x[idx2-1ul] + x[idx2+1ul] + x[idx2-N]
                    + x[idx2+N] + x[idx2-N2]+ x[idx2+N2]
                    - 6.f * x[idx2]) * invh2;
    }
}

void Native::interior_1d_nested(std::vector<float>& out, const std::vector<float>& x, size_t N) 
{
    const size_t N2 = N*N;
    const size_t Nm1 = N-1ul;
    const float invh2 = static_cast<float>(N2);
    #pragma omp parallel for
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j(1ul); j < Nm1; j++)
    for(size_t k(1ul); k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1ul] + x[idx+1ul] + x[idx-N]
                    + x[idx+N] + x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}
void Native::interior_1d_nested_simd(std::vector<float>& out, const std::vector<float>& x, size_t N) 
{
    const size_t N2 = N*N;
    const size_t Nm1 = N-1ul;
    const float invh2 = static_cast<float>(N2);
    #pragma omp parallel for
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j(1ul); j < Nm1; j++)
    #pragma omp simd
    for(size_t k = 1ul; k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1ul]+ x[idx+1ul]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}

void Native::interior_1d_malloc_nested(float* out, const float* x, size_t N) 
{
    const size_t N2 = N*N;
    const size_t Nm1 = N-1u;
    const float invh2 = static_cast<float>(N2);
    #pragma omp parallel for
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j(1ul); j < Nm1; j++)
    for(size_t k(1ul); k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1u]+ x[idx+1u]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}

void Native::interior_1d_malloc_nested_simd(float* out, const float* x, size_t N) 
{
    const size_t N2 = N*N;
    const size_t Nm1 = N-1ul;
    const float invh2 = static_cast<float>(N2);
    #pragma omp parallel for
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j(1ul); j < Nm1; j++)
    #pragma omp simd
    for(size_t k = 1ul; k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1u]+ x[idx+1u]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}
void Native::interior_1d_malloc_nested_i32_max32_idx64(float* out, const float* x, const size_t N) 
{
    // Use uint32_t for i,j,k. uint64_t otherwise
    const uint32_t N1 = N;
    const uint32_t Nm1 = N-1ul;
    const uint32_t N2 = N*N;
    const float invh2 = N2;
    #pragma omp parallel for
    for(uint32_t i=1u; i < Nm1; i++)
    for(uint32_t j(1u); j < Nm1; j++)
    for(uint32_t k=1u; k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N1 + k;
        out[idx] =  ( x[idx-1]+ x[idx+1]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}
void Native::interior_1d_malloc_nested_i32_max32_idx64_simd(float* out, const float* x, const size_t N) 
{
    const uint32_t N1 = N;
    const uint32_t Nm1 = N-1ul;
    const uint32_t N2 = N*N;
    const float invh2 = N2;
    #pragma omp parallel for
    for(uint32_t i=1u; i < Nm1; i++)
    for(uint32_t j(1u); j < Nm1; j++)
    #pragma omp simd
    for(uint32_t k=1u; k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N1 + k;
        out[idx] =  ( x[idx-1]+ x[idx+1]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}
void Native::interior_1d_malloc_nested_i32_max32_idx64promotion(float* out, const float* x, const size_t N) 
{
    // Use uint32_t for i,j,k. uint64_t otherwise
    const uint32_t Nm1 = N-1ul;
    const uint32_t N2 = N*N;
    const float invh2 = N2;
    #pragma omp parallel for
    for(uint32_t i=1u; i < Nm1; i++)
    for(uint32_t j(1u); j < Nm1; j++)
    for(uint32_t k=1u; k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1]+ x[idx+1]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}

void Native::interior_1d_malloc_nested_i32_max32_idx64promotion_simd(float* out, const float* x, const size_t N) 
{
    const uint32_t Nm1 = N-1ul;
    const uint32_t N2 = N*N;
    const float invh2 = N2;
    #pragma omp parallel for
    for(uint32_t i=1u; i < Nm1; i++)
    for(uint32_t j(1u); j < Nm1; j++)
    #pragma omp simd
    for(uint32_t k=1u; k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1]+ x[idx+1]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}
void Native::interior_1d_malloc_nested_i32_max32_idx32(float* out, const float* x, const size_t N) 
{
    // Use uint32_t for all integers
    // The final result is slow. Probably due to the adressing which should be 64bits in a 64bits system
    const uint32_t N1 = N;
    const uint32_t Nm1 = N-1u;
    const uint32_t N2 = N*N;
    const float invh2 = N2;
    #pragma omp parallel for
    for(uint32_t i=1u; i < Nm1; i++)
    for(uint32_t j(1u); j < Nm1; j++)
    for(uint32_t k=1u; k < Nm1; k++)
    {
        const uint32_t idx = i * N2 + j * N1 + k;
        out[idx] =  ( x[idx-1u]+ x[idx+1u]+ x[idx-N1]
                    + x[idx+N1]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}
void Native::interior_1d_malloc_nested_i32_max32_idx32_simd(float* out, const float* x, const size_t N) 
{
    // Use uint32_t for all integers
    // The final result is slow. Probably due to the adressing which should be 64bits in a 64bits system
    const uint32_t N1 = N;
    const uint32_t Nm1 = N-1u;
    const uint32_t N2 = N*N;
    const float invh2 = N2;
    #pragma omp parallel for
    for(uint32_t i=1u; i < Nm1; i++)
    for(uint32_t j(1u); j < Nm1; j++)
    #pragma omp simd
    for(uint32_t k=1u; k < Nm1; k++)
    {
        const uint32_t idx = i * N2 + j * N1 + k;
        out[idx] =  ( x[idx-1u]+ x[idx+1u]+ x[idx-N1]
                    + x[idx+N1]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}
void Native::interior_1d_malloc_nested_i32_max64_idx64(float* out, const float* x, const size_t N) 
{
    // Use uint32_t for i,j,k. uint64_t otherwise
    // Problem in the for loop since Nm1 can be more than the max of uint32_t.
    // This prevents vectorization
    const size_t Nm1 = N-1ul;
    const size_t N2 = N*N;
    const float invh2 = N2;
    #pragma omp parallel for
    for(uint32_t i=1u; i < Nm1; i++)
    for(uint32_t j(1u); j < Nm1; j++)
    for(uint32_t k=1u; k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1]+ x[idx+1]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}
void Native::interior_1d_malloc_nested_i32_max64_idx64_simd(float* out, const float* x, const size_t N) 
{
    // Use uint32_t for i,j,k. uint64_t otherwise
    // Problem in the for loop since Nm1 can be more than the max of uint32_t.
    // This prevents vectorization normally, but "omp simd" allow for aggressive vectorization  
    const size_t Nm1 = N-1ul;
    const size_t N2 = N*N;
    const float invh2 = N2;
    #pragma omp parallel for
    for(uint32_t i=1u; i < Nm1; i++)
    for(uint32_t j(1u); j < Nm1; j++)
    #pragma omp simd
    for(uint32_t k=1u; k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1]+ x[idx+1]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}
void Native::interior_1d_malloc_nested_i32_max64_idx32(float* out, const float* x, const size_t N) 
{
    // Use uint32_t for i,j,k. uint64_t otherwise
    // Problem in the for loop since Nm1 can be more than the max of uint32_t.
    // This prevents vectorization
    const size_t Nm1 = N-1ul;
    const size_t N2 = N*N;
    const float invh2 = N2;
    #pragma omp parallel for
    for(uint32_t i=1u; i < Nm1; i++)
    for(uint32_t j(1u); j < Nm1; j++)
    for(uint32_t k=1u; k < Nm1; k++)
    {
        const uint32_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1]+ x[idx+1]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}
void Native::interior_1d_malloc_nested_i32_max64_idx32_simd(float* out, const float* x, const size_t N) 
{
    // Use uint32_t for i,j,k. uint64_t otherwise
    // Problem in the for loop since Nm1 can be more than the max of uint32_t.
    // This prevents vectorization
    const size_t Nm1 = N-1ul;
    const size_t N2 = N*N;
    const float invh2 = N2;
    #pragma omp parallel for
    for(uint32_t i=1u; i < Nm1; i++)
    for(uint32_t j(1u); j < Nm1; j++)
    #pragma omp simd
    for(uint32_t k=1u; k < Nm1; k++)
    {
        const uint32_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1]+ x[idx+1]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}
template<size_t N>
void Native::interior_1d_malloc_nested_constexpr(float* out, const float* x) 
{
    constexpr size_t N2 = N*N;
    constexpr size_t Nm1 = N-1ul;
    constexpr float invh2 = N2;
    #pragma omp parallel for
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j(1ul); j < Nm1; j++)
    for(size_t k(1ul); k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1ul]+ x[idx+1ul]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}

template<size_t N>
void Native::interior_1d_malloc_nested_constexpr_simd(float* out, const float* x) 
{
    constexpr size_t N2 = N*N;
    constexpr size_t Nm1 = N-1ul;
    constexpr float invh2 = N2;
    #pragma omp parallel for
    for(size_t i=1ul; i < Nm1; i++)
    for(size_t j(1ul); j < Nm1; j++)
    #pragma omp simd
    for(size_t k=1ul; k < Nm1; k++)
    {
        const size_t idx = i * N2 + j * N + k;
        out[idx] =  ( x[idx-1ul]+ x[idx+1ul]+ x[idx-N]
                    + x[idx+N]+ x[idx-N2]+ x[idx+N2]
                    - 6.f * x[idx]) * invh2;
    }
}

// Indexing helper: (i,j,k) -> flat index
inline size_t idx_ij(size_t i, size_t j, size_t N) {
    return j*N + i*N*N;
}

// The optimized 3D interior loop
void Native::interior_1d_aligned_nested(float* out, const float* x, size_t N) {
    const float invh2 = static_cast<float>(N * N);
    const size_t Nm1 = N - 1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 1; i < Nm1; ++i)
    for (size_t j = 1; j < Nm1; ++j) {

        // Pointers to rows for easier SIMD-friendly access
        const float* __restrict xp     = x + idx_ij(i, j, N);
        const float* __restrict xp_jm1 = x + idx_ij(i, j-1, N);
        const float* __restrict xp_jp1 = x + idx_ij(i, j+1, N);
        const float* __restrict xp_im1 = x + idx_ij(i-1, j, N);
        const float* __restrict xp_ip1 = x + idx_ij(i+1, j, N);
        float* __restrict outp         = out + idx_ij(i, j, N);

        #pragma omp simd aligned(outp,xp,xp_jm1,xp_jp1,xp_im1,xp_ip1:64)
        for (size_t k = 1; k < Nm1; ++k) {
            const float sum_neighbors = xp[k-1] + xp[k+1] 
                + xp_jm1[k] + xp_jp1[k] 
                + xp_im1[k] + xp_ip1[k];

            const float tmp = std::fma(-6.f,xp[k], sum_neighbors);
            outp[k] = tmp*invh2;
        }
    }
}