//
//  GEMM with arbitrary row and column strides (ulmblas_strided.h): the C
//  functions call ulmBLAS::gemm, which takes a row and a column stride for
//  each matrix.  Names with the prefix of the configuration (ULMBLAS(x)):
//  ulmblas_dgemm_strided in libcblas.a.
//
#include <complex>
#include <cstddef>
#include <cstdint>
#include <ulmblas/ulmblas.h>

extern "C" {

void
ULMBLAS(sgemm_strided)(std::int64_t  m,
                       std::int64_t  n,
                       std::int64_t  k,
                       float         alpha,
                       const float   *A,
                       std::int64_t  incRowA,
                       std::int64_t  incColA,
                       const float   *B,
                       std::int64_t  incRowB,
                       std::int64_t  incColB,
                       float         beta,
                       float         *C,
                       std::int64_t  incRowC,
                       std::int64_t  incColC)
{
    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                  alpha,
                                  false, A, incRowA, incColA,
                                  false, B, incRowB, incColB,
                                  beta,
                                  C, incRowC, incColC);
}

void
ULMBLAS(dgemm_strided)(std::int64_t  m,
                       std::int64_t  n,
                       std::int64_t  k,
                       double        alpha,
                       const double  *A,
                       std::int64_t  incRowA,
                       std::int64_t  incColA,
                       const double  *B,
                       std::int64_t  incRowB,
                       std::int64_t  incColB,
                       double        beta,
                       double        *C,
                       std::int64_t  incRowC,
                       std::int64_t  incColC)
{
    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                  alpha,
                                  false, A, incRowA, incColA,
                                  false, B, incRowB, incColB,
                                  beta,
                                  C, incRowC, incColC);
}

void
ULMBLAS(cgemm_strided)(std::int64_t  m,
                       std::int64_t  n,
                       std::int64_t  k,
                       const float   *alpha_,
                       int           conjA,
                       const float   *A_,
                       std::int64_t  incRowA,
                       std::int64_t  incColA,
                       int           conjB,
                       const float   *B_,
                       std::int64_t  incRowB,
                       std::int64_t  incColB,
                       const float   *beta_,
                       float         *C_,
                       std::int64_t  incRowC,
                       std::int64_t  incColC)
{
    typedef std::complex<float> fcomplex;

    fcomplex alpha(alpha_[0], alpha_[1]);
    fcomplex beta(beta_[0], beta_[1]);

    const fcomplex *A = reinterpret_cast<const fcomplex *>(A_);
    const fcomplex *B = reinterpret_cast<const fcomplex *>(B_);
    fcomplex       *C = reinterpret_cast<fcomplex *>(C_);

    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                  alpha,
                                  conjA!=0, A, incRowA, incColA,
                                  conjB!=0, B, incRowB, incColB,
                                  beta,
                                  C, incRowC, incColC);
}

void
ULMBLAS(zgemm_strided)(std::int64_t  m,
                       std::int64_t  n,
                       std::int64_t  k,
                       const double  *alpha_,
                       int           conjA,
                       const double  *A_,
                       std::int64_t  incRowA,
                       std::int64_t  incColA,
                       int           conjB,
                       const double  *B_,
                       std::int64_t  incRowB,
                       std::int64_t  incColB,
                       const double  *beta_,
                       double        *C_,
                       std::int64_t  incRowC,
                       std::int64_t  incColC)
{
    typedef std::complex<double> dcomplex;

    dcomplex alpha(alpha_[0], alpha_[1]);
    dcomplex beta(beta_[0], beta_[1]);

    const dcomplex *A = reinterpret_cast<const dcomplex *>(A_);
    const dcomplex *B = reinterpret_cast<const dcomplex *>(B_);
    dcomplex       *C = reinterpret_cast<dcomplex *>(C_);

    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k,
                                  alpha,
                                  conjA!=0, A, incRowA, incColA,
                                  conjB!=0, B, incRowB, incColB,
                                  beta,
                                  C, incRowC, incColC);
}

} // extern "C"
