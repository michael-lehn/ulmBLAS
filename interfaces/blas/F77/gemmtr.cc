#include <algorithm>
#include <cctype>
#include <complex>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/ulmblas.h>

//
//  ?GEMMTR of the reference BLAS (LAPACK 3.12): the lower or upper
//  triangle of the n x n matrix C is set to alpha*op(A)*op(B) + beta*C.
//  As the reference, TRANSA and TRANSB are 'N', 'T' or 'C' ('C' is the
//  conjugate transpose for complex matrices, the transpose otherwise).
//
namespace {

template <typename T>
void
gemmtr(const char *name, const char *upLo_, const char *transA_,
       const char *transB_, const ULMBLAS_INDEX_TYPE *n_,
       const ULMBLAS_INDEX_TYPE *k_, const T &alpha, const T *A,
       const ULMBLAS_INDEX_TYPE *ldA_, const T *B,
       const ULMBLAS_INDEX_TYPE *ldB_, const T &beta, T *C,
       const ULMBLAS_INDEX_TYPE *ldC_)
{
    const char upLo = toupper(*upLo_);
    const char transA = toupper(*transA_);
    const char transB = toupper(*transB_);
    ULMBLAS_INDEX_TYPE n = *n_;
    ULMBLAS_INDEX_TYPE k = *k_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

    ULMBLAS_INDEX_TYPE numRowsA = (transA == 'N') ? n : k;
    ULMBLAS_INDEX_TYPE numRowsB = (transB == 'N') ? k : n;

    ULMBLAS_INDEX_TYPE info = 0;

    if (upLo != 'L' && upLo != 'U') {
	info = 1;
    } else if (transA != 'N' && transA != 'T' && transA != 'C') {
	info = 2;
    } else if (transB != 'N' && transB != 'T' && transB != 'C') {
	info = 3;
    } else if (n < 0) {
	info = 4;
    } else if (k < 0) {
	info = 5;
    } else if (ldA < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsA)) {
	info = 8;
    } else if (ldB < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsB)) {
	info = 10;
    } else if (ldC < std::max<ULMBLAS_INDEX_TYPE>(1, n)) {
	info = 13;
    }

    if (info != 0) {
	F77BLAS(xerbla)(name, &info, 7);
	return;
    }

    //  op(A) is n x k, op(B) is k x n: transposition by the strides
    const std::ptrdiff_t incRowA = (transA == 'N') ? 1 : ldA;
    const std::ptrdiff_t incColA = (transA == 'N') ? ldA : 1;
    const std::ptrdiff_t incRowB = (transB == 'N') ? 1 : ldB;
    const std::ptrdiff_t incColB = (transB == 'N') ? ldB : 1;

    ulmBLAS::gemmtr<std::ptrdiff_t>(upLo == 'L', n, k, alpha, transA == 'C', A,
                                    incRowA, incColA, transB == 'C', B, incRowB,
                                    incColB, beta, C, 1, ldC);
}

} // namespace

extern "C" {

void
F77BLAS(sgemmtr)(const char *upLo_, const char *transA_, const char *transB_,
                 const ULMBLAS_INDEX_TYPE *n_, const ULMBLAS_INDEX_TYPE *k_,
                 const float *alpha_, const float *A,
                 const ULMBLAS_INDEX_TYPE *ldA_, const float *B,
                 const ULMBLAS_INDEX_TYPE *ldB_, const float *beta_, float *C,
                 const ULMBLAS_INDEX_TYPE *ldC_)
{
    //  for real matrices 'C' means the transpose: no conjugation
    const char tA = (toupper(*transA_) == 'C') ? 'T' : *transA_;
    const char tB = (toupper(*transB_) == 'C') ? 'T' : *transB_;
    gemmtr("SGEMMTR", upLo_, &tA, &tB, n_, k_, *alpha_, A, ldA_, B, ldB_,
           *beta_, C, ldC_);
}

void
F77BLAS(dgemmtr)(const char *upLo_, const char *transA_, const char *transB_,
                 const ULMBLAS_INDEX_TYPE *n_, const ULMBLAS_INDEX_TYPE *k_,
                 const double *alpha_, const double *A,
                 const ULMBLAS_INDEX_TYPE *ldA_, const double *B,
                 const ULMBLAS_INDEX_TYPE *ldB_, const double *beta_, double *C,
                 const ULMBLAS_INDEX_TYPE *ldC_)
{
    const char tA = (toupper(*transA_) == 'C') ? 'T' : *transA_;
    const char tB = (toupper(*transB_) == 'C') ? 'T' : *transB_;
    gemmtr("DGEMMTR", upLo_, &tA, &tB, n_, k_, *alpha_, A, ldA_, B, ldB_,
           *beta_, C, ldC_);
}

void
F77BLAS(cgemmtr)(const char *upLo_, const char *transA_, const char *transB_,
                 const ULMBLAS_INDEX_TYPE *n_, const ULMBLAS_INDEX_TYPE *k_,
                 const float *alpha_, const float *A_,
                 const ULMBLAS_INDEX_TYPE *ldA_, const float *B_,
                 const ULMBLAS_INDEX_TYPE *ldB_, const float *beta_, float *C_,
                 const ULMBLAS_INDEX_TYPE *ldC_)
{
    typedef std::complex<float> fcomplex;

    gemmtr(
        "CGEMMTR", upLo_, transA_, transB_, n_, k_,
        fcomplex(alpha_[0], alpha_[1]), reinterpret_cast<const fcomplex *>(A_),
        ldA_, reinterpret_cast<const fcomplex *>(B_), ldB_,
        fcomplex(beta_[0], beta_[1]), reinterpret_cast<fcomplex *>(C_), ldC_);
}

void
F77BLAS(zgemmtr)(const char *upLo_, const char *transA_, const char *transB_,
                 const ULMBLAS_INDEX_TYPE *n_, const ULMBLAS_INDEX_TYPE *k_,
                 const double *alpha_, const double *A_,
                 const ULMBLAS_INDEX_TYPE *ldA_, const double *B_,
                 const ULMBLAS_INDEX_TYPE *ldB_, const double *beta_,
                 double *C_, const ULMBLAS_INDEX_TYPE *ldC_)
{
    typedef std::complex<double> dcomplex;

    gemmtr(
        "ZGEMMTR", upLo_, transA_, transB_, n_, k_,
        dcomplex(alpha_[0], alpha_[1]), reinterpret_cast<const dcomplex *>(A_),
        ldA_, reinterpret_cast<const dcomplex *>(B_), ldB_,
        dcomplex(beta_[0], beta_[1]), reinterpret_cast<dcomplex *>(C_), ldC_);
}

} // extern "C"
