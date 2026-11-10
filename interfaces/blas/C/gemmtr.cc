#include BLAS_HEADER
#include <algorithm>
#include <complex>
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/ulmblas.h>

//
//  ?gemmtr (BLAS of LAPACK 3.12): the lower or upper triangle of the n x n
//  matrix C is set to alpha*op(A)*op(B) + beta*C.  As the reference CBLAS,
//  the transposition is CblasNoTrans, CblasTrans or CblasConjTrans (the
//  conjugate transpose for complex matrices, the transpose otherwise; the
//  ATLAS extension AtlasConj of gemm is not accepted).
//
//  set before calling cblas_xerbla (row major: positions of the column major
//  call, translated by cblas_xerbla), see cblas_globals.cc
extern "C" int RowMajorStrg;

namespace {

//  column major: C(triangle) = alpha*op(A)*op(B) + beta*C(triangle)
template <typename T>
void
gemmtr(bool complex, enum CBLAS_UPLO upLo, enum CBLAS_TRANSPOSE transA,
       enum CBLAS_TRANSPOSE transB, ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE k,
       const T &alpha, const T *A, ULMBLAS_INDEX_TYPE ldA, const T *B,
       ULMBLAS_INDEX_TYPE ldB, const T &beta, T *C, ULMBLAS_INDEX_TYPE ldC)
{
    const bool tA = (transA != CblasNoTrans);
    const bool tB = (transB != CblasNoTrans);

    ulmBLAS::gemmtr<std::ptrdiff_t>(
        upLo == CblasLower, n, k, alpha, complex && transA == CblasConjTrans, A,
        tA ? ldA : 1, tA ? 1 : ldA, complex && transB == CblasConjTrans, B,
        tB ? ldB : 1, tB ? 1 : ldB, beta, C, 1, ldC);
}

//  argument checks as the reference CBLAS, then the column major routine;
//  row major: C^T = op(B)^T*op(A)^T with the other triangle
template <typename T>
void
cblas_gemmtr(const char *name, bool complex, enum CBLAS_ORDER order,
             enum CBLAS_UPLO upLo, enum CBLAS_TRANSPOSE transA,
             enum CBLAS_TRANSPOSE transB, ULMBLAS_INDEX_TYPE n,
             ULMBLAS_INDEX_TYPE k, const T &alpha, const T *A,
             ULMBLAS_INDEX_TYPE ldA, const T *B, ULMBLAS_INDEX_TYPE ldB,
             const T &beta, T *C, ULMBLAS_INDEX_TYPE ldC)
{
    auto valid = [](enum CBLAS_TRANSPOSE trans) {
	return trans == CblasNoTrans || trans == CblasTrans ||
	       trans == CblasConjTrans;
    };
    const bool tA = (transA != CblasNoTrans);
    const bool tB = (transB != CblasNoTrans);

    //  row major: the column major routine gets B, A
    ULMBLAS_INDEX_TYPE numRowsA = tA ? k : n;
    ULMBLAS_INDEX_TYPE numRowsB = tB ? n : k;
    if (order == CblasRowMajor) {
	numRowsA = tA ? n : k;
	numRowsB = tB ? k : n;
    }

    int info = 0;
    if (order != CblasColMajor && order != CblasRowMajor) {
	info = 1;
    } else if (upLo != CblasUpper && upLo != CblasLower) {
	info = 2;
    } else if (!valid(transA)) {
	info = 3;
    } else if (!valid(transB)) {
	info = 4;
    } else if (n < 0) {
	info = 5;
    } else if (k < 0) {
	info = 6;
    } else if (order == CblasColMajor) {
	if (ldA < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsA)) {
	    info = 9;
	} else if (ldB < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsB)) {
	    info = 11;
	} else if (ldC < std::max<ULMBLAS_INDEX_TYPE>(1, n)) {
	    info = 14;
	}
    } else {
	//  positions in the call of the column major routine (B first);
	//  cblas_xerbla translates them (RowMajorStrg)
	if (ldB < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsB)) {
	    info = 9;
	} else if (ldA < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsA)) {
	    info = 11;
	} else if (ldC < std::max<ULMBLAS_INDEX_TYPE>(1, n)) {
	    info = 14;
	}
    }

    if (info != 0) {
	RowMajorStrg = (order == CblasRowMajor && info > 6) ? 1 : 0;
	CBLAS(xerbla)(info, name, "... bla bla ...");
	return;
    }

    if (order == CblasColMajor) {
	gemmtr(complex, upLo, transA, transB, n, k, alpha, A, ldA, B, ldB, beta,
	       C, ldC);
    } else {
	gemmtr(complex, upLo == CblasUpper ? CblasLower : CblasUpper, transB,
	       transA, n, k, alpha, B, ldB, A, ldA, beta, C, ldC);
    }
}

typedef std::complex<float> fcomplex;
typedef std::complex<double> dcomplex;

} // namespace

extern "C" {

void
ULMBLAS(sgemmtr)(enum CBLAS_UPLO upLo, enum CBLAS_TRANSPOSE transA,
                 enum CBLAS_TRANSPOSE transB, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE k, float alpha, const float *A,
                 ULMBLAS_INDEX_TYPE ldA, const float *B, ULMBLAS_INDEX_TYPE ldB,
                 float beta, float *C, ULMBLAS_INDEX_TYPE ldC)
{
    gemmtr(false, upLo, transA, transB, n, k, alpha, A, ldA, B, ldB, beta, C,
           ldC);
}

void
ULMBLAS(dgemmtr)(enum CBLAS_UPLO upLo, enum CBLAS_TRANSPOSE transA,
                 enum CBLAS_TRANSPOSE transB, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE k, double alpha, const double *A,
                 ULMBLAS_INDEX_TYPE ldA, const double *B,
                 ULMBLAS_INDEX_TYPE ldB, double beta, double *C,
                 ULMBLAS_INDEX_TYPE ldC)
{
    gemmtr(false, upLo, transA, transB, n, k, alpha, A, ldA, B, ldB, beta, C,
           ldC);
}

void
ULMBLAS(cgemmtr)(enum CBLAS_UPLO upLo, enum CBLAS_TRANSPOSE transA,
                 enum CBLAS_TRANSPOSE transB, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE k, const void *alpha, const void *A,
                 ULMBLAS_INDEX_TYPE ldA, const void *B, ULMBLAS_INDEX_TYPE ldB,
                 const void *beta, void *C, ULMBLAS_INDEX_TYPE ldC)
{
    gemmtr(
        true, upLo, transA, transB, n, k, *static_cast<const fcomplex *>(alpha),
        static_cast<const fcomplex *>(A), ldA, static_cast<const fcomplex *>(B),
        ldB, *static_cast<const fcomplex *>(beta), static_cast<fcomplex *>(C),
        ldC);
}

void
ULMBLAS(zgemmtr)(enum CBLAS_UPLO upLo, enum CBLAS_TRANSPOSE transA,
                 enum CBLAS_TRANSPOSE transB, ULMBLAS_INDEX_TYPE n,
                 ULMBLAS_INDEX_TYPE k, const void *alpha, const void *A,
                 ULMBLAS_INDEX_TYPE ldA, const void *B, ULMBLAS_INDEX_TYPE ldB,
                 const void *beta, void *C, ULMBLAS_INDEX_TYPE ldC)
{
    gemmtr(
        true, upLo, transA, transB, n, k, *static_cast<const dcomplex *>(alpha),
        static_cast<const dcomplex *>(A), ldA, static_cast<const dcomplex *>(B),
        ldB, *static_cast<const dcomplex *>(beta), static_cast<dcomplex *>(C),
        ldC);
}

void
CBLAS(sgemmtr)(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
               enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
               ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE k, float alpha,
               const float *A, ULMBLAS_INDEX_TYPE ldA, const float *B,
               ULMBLAS_INDEX_TYPE ldB, float beta, float *C,
               ULMBLAS_INDEX_TYPE ldC)
{
    cblas_gemmtr("cblas_sgemmtr", false, order, upLo, transA, transB, n, k,
                 alpha, A, ldA, B, ldB, beta, C, ldC);
}

void
CBLAS(dgemmtr)(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
               enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
               ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE k, double alpha,
               const double *A, ULMBLAS_INDEX_TYPE ldA, const double *B,
               ULMBLAS_INDEX_TYPE ldB, double beta, double *C,
               ULMBLAS_INDEX_TYPE ldC)
{
    cblas_gemmtr("cblas_dgemmtr", false, order, upLo, transA, transB, n, k,
                 alpha, A, ldA, B, ldB, beta, C, ldC);
}

void
CBLAS(cgemmtr)(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
               enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
               ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE k, const void *alpha,
               const void *A, ULMBLAS_INDEX_TYPE ldA, const void *B,
               ULMBLAS_INDEX_TYPE ldB, const void *beta, void *C,
               ULMBLAS_INDEX_TYPE ldC)
{
    cblas_gemmtr(
        "cblas_cgemmtr", true, order, upLo, transA, transB, n, k,
        *static_cast<const fcomplex *>(alpha), static_cast<const fcomplex *>(A),
        ldA, static_cast<const fcomplex *>(B), ldB,
        *static_cast<const fcomplex *>(beta), static_cast<fcomplex *>(C), ldC);
}

void
CBLAS(zgemmtr)(enum CBLAS_ORDER order, enum CBLAS_UPLO upLo,
               enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
               ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE k, const void *alpha,
               const void *A, ULMBLAS_INDEX_TYPE ldA, const void *B,
               ULMBLAS_INDEX_TYPE ldB, const void *beta, void *C,
               ULMBLAS_INDEX_TYPE ldC)
{
    cblas_gemmtr(
        "cblas_zgemmtr", true, order, upLo, transA, transB, n, k,
        *static_cast<const dcomplex *>(alpha), static_cast<const dcomplex *>(A),
        ldA, static_cast<const dcomplex *>(B), ldB,
        *static_cast<const dcomplex *>(beta), static_cast<dcomplex *>(C), ldC);
}

} // extern "C"
