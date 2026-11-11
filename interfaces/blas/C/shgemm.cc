//
//  shgemm: C <- alpha*op(A)*op(B) + beta*C with A and B in float16 (IEEE
//  half precision), C, alpha and beta in float (as OpenBLAS's shgemm).
//  The products and sums are computed in float: the panels of A and B are
//  converted to float when they are packed, the micro kernels are those
//  of sgemm.
//
#include BLAS_HEADER
#include <interfaces/blas/C/transpose.h>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/auxiliary/float16.h>
#include <ulmblas/ulmblas.h>

using ulmBLAS::float16;

extern "C" {

void
ULMBLAS(shgemm)(enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
                ULMBLAS_INDEX_TYPE m, ULMBLAS_INDEX_TYPE n,
                ULMBLAS_INDEX_TYPE k, float alpha, const ulmblas_float16 *A_,
                ULMBLAS_INDEX_TYPE ldA, const ulmblas_float16 *B_,
                ULMBLAS_INDEX_TYPE ldB, float beta, float *C,
                ULMBLAS_INDEX_TYPE ldC)
{
    const float16 *A = reinterpret_cast<const float16 *>(A_);
    const float16 *B = reinterpret_cast<const float16 *>(B_);

    //
    //  Start the operations.
    //
    if (transB == CblasNoTrans || transB == AtlasConj) {
	if (transA == CblasNoTrans || transA == AtlasConj) {
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
	if (transA == CblasNoTrans || transA == AtlasConj) {
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

void
CBLAS(shgemm)(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA,
              enum CBLAS_TRANSPOSE transB, ULMBLAS_INDEX_TYPE m,
              ULMBLAS_INDEX_TYPE n, ULMBLAS_INDEX_TYPE k, float alpha,
              const ulmblas_float16 *A_, ULMBLAS_INDEX_TYPE ldA,
              const ulmblas_float16 *B_, ULMBLAS_INDEX_TYPE ldB, float beta,
              float *C, ULMBLAS_INDEX_TYPE ldC)
{
    //
    //  Set  numRowsA and numRowsB as the number of rows of A and B
    //
    ULMBLAS_INDEX_TYPE numRowsA;
    ULMBLAS_INDEX_TYPE numRowsB;

    if (order == CblasColMajor) {
	numRowsA = (transA == CblasNoTrans || transA == AtlasConj) ? m : k;
	numRowsB = (transB == CblasNoTrans || transB == AtlasConj) ? k : n;
    } else {
	numRowsB = (transB == CblasNoTrans || transB == AtlasConj) ? n : k;
	numRowsA = (transA == CblasNoTrans || transA == AtlasConj) ? k : m;
    }

    //
    //  Test the input parameters
    //
    int info = 0;
    if (order != CblasColMajor && order != CblasRowMajor) {
	info = 1;
    } else if (transA != CblasNoTrans && transA != CblasTrans &&
               transA != AtlasConj && transA != CblasConjTrans) {
	info = 2;
    } else if (transB != CblasNoTrans && transB != CblasTrans &&
               transB != AtlasConj && transB != CblasConjTrans) {
	info = 3;
    } else {
	if (order == CblasColMajor) {
	    if (m < 0) {
		info = 4;
	    } else if (n < 0) {
		info = 5;
	    } else if (k < 0) {
		info = 6;
	    } else if (ldA < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsA)) {
		info = 9;
	    } else if (ldB < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsB)) {
		info = 11;
	    } else if (ldC < std::max<ULMBLAS_INDEX_TYPE>(1, m)) {
		info = 14;
	    }
	} else {
	    if (n < 0) {
		info = 4;
	    } else if (m < 0) {
		info = 5;
	    } else if (k < 0) {
		info = 6;
	    } else if (ldB < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsB)) {
		info = 9;
	    } else if (ldA < std::max<ULMBLAS_INDEX_TYPE>(1, numRowsA)) {
		info = 11;
	    } else if (ldC < std::max<ULMBLAS_INDEX_TYPE>(1, n)) {
		info = 14;
	    }
	}
    }

    if (info != 0) {
	extern int RowMajorStrg;

	RowMajorStrg = (order == CblasRowMajor) ? 1 : 0;
	CBLAS(xerbla)(info, "cblas_shgemm", "... bla bla ...");
	return;
    }

    if (order == CblasColMajor) {
	ULMBLAS(shgemm)(transA, transB, m, n, k, alpha, A_, ldA, B_, ldB, beta,
	                C, ldC);
    } else {
	ULMBLAS(shgemm)(transB, transA, n, m, k, alpha, B_, ldB, A_, ldA, beta,
	                C, ldC);
    }
}

} // extern "C"
