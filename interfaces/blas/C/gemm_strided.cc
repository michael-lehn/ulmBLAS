//
//  GEMM with arbitrary row and column strides (ulmblas_strided.h): the C
//  functions call ulmBLAS::gemm, which takes a row and a column stride for
//  each matrix.  Names with the prefix of the configuration (ULMBLAS(x)):
//  ulmblas_dgemm_strided in libcblas.a.
//
//  The integer functions compute with the unsigned type of the same size:
//  the result is the same bit pattern as with signed wraparound (Julia's
//  Int), without the undefined behaviour of signed overflow in C++.
//
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <ulmblas/ulmblas.h>

namespace {

template <typename T, typename U>
void
gemmUnsigned(std::int64_t m, std::int64_t n, std::int64_t k, T alpha,
             const T *A, std::int64_t incRowA, std::int64_t incColA, const T *B,
             std::int64_t incRowB, std::int64_t incColB, T beta, T *C,
             std::int64_t incRowC, std::int64_t incColC)
{
    //  U*U is int for 8 and 16 bit: compute in its unsigned type W (the low
    //  bits are the same), so that signed overflow cannot occur
    typedef typename std::make_unsigned<decltype(U(0) * U(0))>::type W;

    ulmBLAS::gemm<std::ptrdiff_t>(
        m, n, k, static_cast<W>(static_cast<U>(alpha)), false,
        reinterpret_cast<const U *>(A), incRowA, incColA, false,
        reinterpret_cast<const U *>(B), incRowB, incColB,
        static_cast<W>(static_cast<U>(beta)), reinterpret_cast<U *>(C), incRowC,
        incColC);
}

} // namespace

extern "C" {

void
ULMBLAS(sgemm_strided)(std::int64_t m, std::int64_t n, std::int64_t k,
                       float alpha, const float *A, std::int64_t incRowA,
                       std::int64_t incColA, const float *B,
                       std::int64_t incRowB, std::int64_t incColB, float beta,
                       float *C, std::int64_t incRowC, std::int64_t incColC)
{
    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k, alpha, false, A, incRowA, incColA,
                                  false, B, incRowB, incColB, beta, C, incRowC,
                                  incColC);
}

void
ULMBLAS(dgemm_strided)(std::int64_t m, std::int64_t n, std::int64_t k,
                       double alpha, const double *A, std::int64_t incRowA,
                       std::int64_t incColA, const double *B,
                       std::int64_t incRowB, std::int64_t incColB, double beta,
                       double *C, std::int64_t incRowC, std::int64_t incColC)
{
    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k, alpha, false, A, incRowA, incColA,
                                  false, B, incRowB, incColB, beta, C, incRowC,
                                  incColC);
}

void
ULMBLAS(cgemm_strided)(std::int64_t m, std::int64_t n, std::int64_t k,
                       const float *alpha_, int conjA, const float *A_,
                       std::int64_t incRowA, std::int64_t incColA, int conjB,
                       const float *B_, std::int64_t incRowB,
                       std::int64_t incColB, const float *beta_, float *C_,
                       std::int64_t incRowC, std::int64_t incColC)
{
    typedef std::complex<float> fcomplex;

    fcomplex alpha(alpha_[0], alpha_[1]);
    fcomplex beta(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *B = reinterpret_cast<const fcomplex *>(B_);
    fcomplex *C = reinterpret_cast<fcomplex *>(C_);

    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k, alpha, conjA != 0, A, incRowA,
                                  incColA, conjB != 0, B, incRowB, incColB,
                                  beta, C, incRowC, incColC);
}

void
ULMBLAS(zgemm_strided)(std::int64_t m, std::int64_t n, std::int64_t k,
                       const double *alpha_, int conjA, const double *A_,
                       std::int64_t incRowA, std::int64_t incColA, int conjB,
                       const double *B_, std::int64_t incRowB,
                       std::int64_t incColB, const double *beta_, double *C_,
                       std::int64_t incRowC, std::int64_t incColC)
{
    typedef std::complex<double> dcomplex;

    dcomplex alpha(alpha_[0], alpha_[1]);
    dcomplex beta(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *B = reinterpret_cast<const dcomplex *>(B_);
    dcomplex *C = reinterpret_cast<dcomplex *>(C_);

    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k, alpha, conjA != 0, A, incRowA,
                                  incColA, conjB != 0, B, incRowB, incColB,
                                  beta, C, incRowC, incColC);
}

void
ULMBLAS(i8gemm_strided)(std::int64_t m, std::int64_t n, std::int64_t k,
                        std::int8_t alpha, const std::int8_t *A,
                        std::int64_t incRowA, std::int64_t incColA,
                        const std::int8_t *B, std::int64_t incRowB,
                        std::int64_t incColB, std::int8_t beta, std::int8_t *C,
                        std::int64_t incRowC, std::int64_t incColC)
{
    gemmUnsigned<std::int8_t, std::uint8_t>(m, n, k, alpha, A, incRowA, incColA,
                                            B, incRowB, incColB, beta, C,
                                            incRowC, incColC);
}

void
ULMBLAS(i16gemm_strided)(std::int64_t m, std::int64_t n, std::int64_t k,
                         std::int16_t alpha, const std::int16_t *A,
                         std::int64_t incRowA, std::int64_t incColA,
                         const std::int16_t *B, std::int64_t incRowB,
                         std::int64_t incColB, std::int16_t beta,
                         std::int16_t *C, std::int64_t incRowC,
                         std::int64_t incColC)
{
    gemmUnsigned<std::int16_t, std::uint16_t>(m, n, k, alpha, A, incRowA,
                                              incColA, B, incRowB, incColB,
                                              beta, C, incRowC, incColC);
}

void
ULMBLAS(i32gemm_strided)(std::int64_t m, std::int64_t n, std::int64_t k,
                         std::int32_t alpha, const std::int32_t *A,
                         std::int64_t incRowA, std::int64_t incColA,
                         const std::int32_t *B, std::int64_t incRowB,
                         std::int64_t incColB, std::int32_t beta,
                         std::int32_t *C, std::int64_t incRowC,
                         std::int64_t incColC)
{
    gemmUnsigned<std::int32_t, std::uint32_t>(m, n, k, alpha, A, incRowA,
                                              incColA, B, incRowB, incColB,
                                              beta, C, incRowC, incColC);
}

void
ULMBLAS(i64gemm_strided)(std::int64_t m, std::int64_t n, std::int64_t k,
                         std::int64_t alpha, const std::int64_t *A,
                         std::int64_t incRowA, std::int64_t incColA,
                         const std::int64_t *B, std::int64_t incRowB,
                         std::int64_t incColB, std::int64_t beta,
                         std::int64_t *C, std::int64_t incRowC,
                         std::int64_t incColC)
{
    gemmUnsigned<std::int64_t, std::uint64_t>(m, n, k, alpha, A, incRowA,
                                              incColA, B, incRowB, incColB,
                                              beta, C, incRowC, incColC);
}

} // extern "C"
