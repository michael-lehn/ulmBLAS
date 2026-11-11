//
//  SBGEMM: C <- alpha*op(A)*op(B) + beta*C with A and B in bfloat16 (as
//  OpenBLAS's sbgemm), see ../C/sbgemm.cc.
//
#include <algorithm>
#include <cctype>
#include <cstdint>
#include BLAS_HEADER
#include <interfaces/blas/F77/xerbla.h>
#include <ulmblas/auxiliary/bfloat16.h>
#include <ulmblas/ulmblas.h>

using ulmBLAS::bfloat16;

extern "C" {

void
F77BLAS(sbgemm)(const char *transA_, const char *transB_,
                const ULMBLAS_INDEX_TYPE *m_, const ULMBLAS_INDEX_TYPE *n_,
                const ULMBLAS_INDEX_TYPE *k_, const float *alpha_,
                const uint16_t *A_, const ULMBLAS_INDEX_TYPE *ldA_,
                const uint16_t *B_, const ULMBLAS_INDEX_TYPE *ldB_,
                const float *beta_, float *C, const ULMBLAS_INDEX_TYPE *ldC_)
{
    //
    //  Dereference scalar parameters
    //
    bool transA = (toupper(*transA_) == 'T' || toupper(*transA_) == 'C');
    bool transB = (toupper(*transB_) == 'T' || toupper(*transB_) == 'C');
    ULMBLAS_INDEX_TYPE m = *m_;
    ULMBLAS_INDEX_TYPE n = *n_;
    ULMBLAS_INDEX_TYPE k = *k_;
    float alpha = *alpha_;
    ULMBLAS_INDEX_TYPE ldA = *ldA_;
    ULMBLAS_INDEX_TYPE ldB = *ldB_;
    float beta = *beta_;
    ULMBLAS_INDEX_TYPE ldC = *ldC_;

    //
    //  Set  numRowsA and numRowsB as the number of rows of A and B
    //
    ULMBLAS_INDEX_TYPE numRowsA = (!transA) ? m : k;
    ULMBLAS_INDEX_TYPE numRowsB = (!transB) ? k : n;

    //
    //  Test the input parameters
    //
    ULMBLAS_INDEX_TYPE info = 0;

    if (toupper(*transA_) != 'N' && toupper(*transA_) != 'T' &&
        toupper(*transA_) != 'C' && toupper(*transA_) != 'R') {
	info = 1;
    } else if (toupper(*transB_) != 'N' && toupper(*transB_) != 'T' &&
               toupper(*transB_) != 'C' && toupper(*transB_) != 'R') {
	info = 2;
    } else if (m < 0) {
	info = 3;
    } else if (n < 0) {
	info = 4;
    } else if (k < 0) {
	info = 5;
    } else if (ldA < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsA)) {
	info = 8;
    } else if (ldB < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsB)) {
	info = 10;
    } else if (ldC < std::max<ULMBLAS_INDEX_TYPE>(1, m)) {
	info = 13;
    }

    if (info != 0) {
	F77BLAS(xerbla)("SBGEMM", &info, 6);
	return;
    }

    const bfloat16 *A = reinterpret_cast<const bfloat16 *>(A_);
    const bfloat16 *B = reinterpret_cast<const bfloat16 *>(B_);

    //
    //  Start the operations.
    //
    if (!transB) {
	if (!transA) {
	    //
	    //          Form  C := alpha*A*B + beta*C.
	    //
	    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k, alpha, false, A, 1, ldA,
	                                  false, B, 1, ldB, beta, C, 1, ldC);
	} else {
	    //
	    //          Form  C := alpha*A**T*B + beta*C
	    //
	    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k, alpha, false, A, ldA, 1,
	                                  false, B, 1, ldB, beta, C, 1, ldC);
	}
    } else {
	if (!transA) {
	    //
	    //          Form  C := alpha*A*B**T + beta*C
	    //
	    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k, alpha, false, A, 1, ldA,
	                                  false, B, ldB, 1, beta, C, 1, ldC);
	} else {
	    //
	    //          Form  C := alpha*A**T*B**T + beta*C
	    //
	    ulmBLAS::gemm<std::ptrdiff_t>(m, n, k, alpha, false, A, ldA, 1,
	                                  false, B, ldB, 1, beta, C, 1, ldC);
	}
    }
}

} // extern "C"
