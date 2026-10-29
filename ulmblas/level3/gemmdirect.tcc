#ifndef ULMBLAS_LEVEL3_GEMMDIRECT_TCC
#define ULMBLAS_LEVEL3_GEMMDIRECT_TCC 1

#include <ulmblas/level3/gemmdirect.h>

#if defined(USE_NEON)
#   include <arm_neon.h>
#endif

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
bool
gemm_direct(IndexType    ,
            IndexType    ,
            IndexType    ,
            const Alpha  &,
            const TA     *,
            IndexType    ,
            IndexType    ,
            const TB     *,
            IndexType    ,
            IndexType    ,
            const Beta   &,
            TC           *,
            IndexType    ,
            IndexType    )
{
    return false;
}

#if defined(USE_NEON)

//
//  Small problems with NEON (AArch64), for double, A with unit row stride
//  (A or A^T stored column major) and C with unit row stride (the C and
//  Fortran interfaces pass C so).  For a small problem packing and the
//  fixed costs of the GEMM algorithm and of the micro kernel (with the SME2
//  kernel: streaming mode, ZA, loading and storing the tile of C through
//  ZA) cost more than the multiplication itself.
//
//  The tiles of C have 2*R rows (R = 4, 2, 1) and N = 1 ... 4 columns, the
//  accumulators are registers: per value of k one column of A (2*R
//  values) is loaded and multiplied with N values of a row of B.  Each
//  element of C is computed as alpha*(sum over l) + beta*C(i,j), the sum in
//  the order of l (as the reference implementation); C is not read if beta
//  is zero.
//
//  Used for m*n*k <= ULMBLAS_GEMM_DIRECT_MAX_MNK (default 8192; 0 disables
//  it, as in the test configurations with small block sizes).  On an Apple
//  M5 the direct path reaches 58 GFLOP/s (84% of the NEON FMA peak) and is
//  faster than the SME2 path for all measured shapes up to m*n*k = 8192
//  (break-even for 20 x 20 x 20, 16 x 16 x 32, 32 x 32 x 8), for squares
//  n = 8: 0.03 instead of 0.21 us.  Skinny shapes gain far more (200 x 4 x
//  4: 16x) and would also gain beyond the threshold (2 x 2 x 4000: 4x, 64 x
//  64 x 4: 1.4x), which is not used yet.
//
namespace direct {

#ifndef ULMBLAS_GEMM_DIRECT_MAX_MNK
#define ULMBLAS_GEMM_DIRECT_MAX_MNK 8192
#endif

template <int R, int N>
inline void
tile(long k, double alpha, const double *A, long ldA,
     const double *B, long incRowB, long incColB,
     double beta, double *C, long ldC)
{
    float64x2_t c[N][R];

    for (int j=0; j<N; ++j) {
        for (int r=0; r<R; ++r) {
            c[j][r] = vdupq_n_f64(0.0);
        }
    }
    for (long l=0; l<k; ++l) {
        float64x2_t a[R];
        for (int r=0; r<R; ++r) {
            a[r] = vld1q_f64(A + 2*r);
        }
        for (int j=0; j<N; ++j) {
            const float64x2_t b = vld1q_dup_f64(B + j*incColB);
            for (int r=0; r<R; ++r) {
                c[j][r] = vfmaq_f64(c[j][r], a[r], b);
            }
        }
        A += ldA;
        B += incRowB;
    }
    for (int j=0; j<N; ++j) {
        for (int r=0; r<R; ++r) {
            double      *Cp = C + j*ldC + 2*r;
            float64x2_t x   = vmulq_n_f64(c[j][r], alpha);
            if (beta!=0.0) {
                x = vfmaq_n_f64(x, vld1q_f64(Cp), beta);
            }
            vst1q_f64(Cp, x);
        }
    }
}

//  The last row (m odd).
template <int N>
inline void
row(long k, double alpha, const double *A, long ldA,
    const double *B, long incRowB, long incColB,
    double beta, double *C, long ldC)
{
    double c[N];

    for (int j=0; j<N; ++j) {
        c[j] = 0.0;
    }
    for (long l=0; l<k; ++l) {
        for (int j=0; j<N; ++j) {
            c[j] += A[0]*B[j*incColB];
        }
        A += ldA;
        B += incRowB;
    }
    for (int j=0; j<N; ++j) {
        C[j*ldC] = (beta!=0.0) ? alpha*c[j] + beta*C[j*ldC] : alpha*c[j];
    }
}

//  Columns j to j+N-1 of C.
template <int N>
inline void
columns(long m, long k, double alpha, const double *A, long ldA,
        const double *B, long incRowB, long incColB,
        double beta, double *C, long ldC)
{
    long i = 0;

    for (; i+8<=m; i+=8) {
        tile<4, N>(k, alpha, A+i, ldA, B, incRowB, incColB, beta, C+i, ldC);
    }
    if (i+4<=m) {
        tile<2, N>(k, alpha, A+i, ldA, B, incRowB, incColB, beta, C+i, ldC);
        i += 4;
    }
    if (i+2<=m) {
        tile<1, N>(k, alpha, A+i, ldA, B, incRowB, incColB, beta, C+i, ldC);
        i += 2;
    }
    if (i<m) {
        row<N>(k, alpha, A+i, ldA, B, incRowB, incColB, beta, C+i, ldC);
    }
}

} // namespace direct

template <typename IndexType>
bool
gemm_direct(IndexType     m,
            IndexType     n,
            IndexType     k,
            const double  &alpha,
            const double  *A,
            IndexType     incRowA,
            IndexType     incColA,
            const double  *B,
            IndexType     incRowB,
            IndexType     incColB,
            const double  &beta,
            double        *C,
            IndexType     incRowC,
            IndexType     incColC)
{
    if (incRowA!=IndexType(1) || incRowC!=IndexType(1)
     || double(m)*double(n)*double(k) > ULMBLAS_GEMM_DIRECT_MAX_MNK)
    {
        return false;
    }

    const long ldA = incColA, ldC = incColC;
    const long iRB = incRowB, iCB = incColB;
    long       j   = 0;

    for (; j+4<=n; j+=4) {
        direct::columns<4>(m, k, alpha, A, ldA, B+j*iCB, iRB, iCB, beta,
                           C+j*ldC, ldC);
    }
    switch (n-j) {
        case 3:
            direct::columns<3>(m, k, alpha, A, ldA, B+j*iCB, iRB, iCB, beta,
                               C+j*ldC, ldC);
            break;
        case 2:
            direct::columns<2>(m, k, alpha, A, ldA, B+j*iCB, iRB, iCB, beta,
                               C+j*ldC, ldC);
            break;
        case 1:
            direct::columns<1>(m, k, alpha, A, ldA, B+j*iCB, iRB, iCB, beta,
                               C+j*ldC, ldC);
            break;
    }
    return true;
}

#endif // USE_NEON

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMMDIRECT_TCC
