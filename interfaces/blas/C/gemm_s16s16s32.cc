//
//  cblas_gemm_s16s16s32: 16-bit integer GEMM with the signature and the
//  semantics of oneMKL's function of that name,
//
//      C <- alpha*(op(A) + ao)*(op(B) + bo) + beta*C + co,
//
//  A, B int16_t.  As oneMKL (measured with oneMKL 2026.2) the sum
//  S = (op(A) + ao)*(op(B) + bo) is an int32 with wraparound (a product of
//  two int16 values alone may need 31 bits); then alpha, beta and the
//  conversion to int32 in double, rounded to nearest (halfway cases away
//  from zero) and saturated to the range of int32.  co: one value
//  (CblasFixOffset), one per row (CblasColOffset) or one per column
//  (CblasRowOffset).
//
//  S = op(A)*op(B) + ao*colsum(B) + bo*rowsum(A) + k*ao*bo modulo 2^32;
//  op(A)*op(B) by ulmBLAS::i16gemm for blocks of columns into a local
//  buffer.
//
#include BLAS_HEADER
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/level3/i8gemm.h>

namespace {

//  s[i] = sum_l X(i,l) modulo 2^32, X(i,l) at X[i*incR + l*incC]
void
sums16(std::int64_t r, std::int64_t k, const std::int16_t *X,
       std::int64_t incR, std::int64_t incC, std::uint32_t *s)
{
    for (std::int64_t i = 0; i < r; ++i) {
	s[i] = 0;
    }
    if (incC == 1) {
	for (std::int64_t i = 0; i < r; ++i) {
	    const std::int16_t *x = X + i * incR;
	    std::uint32_t t = 0;
	    for (std::int64_t l = 0; l < k; ++l) {
		t += std::uint32_t(std::int32_t(x[l]));
	    }
	    s[i] = t;
	}
	return;
    }
    for (std::int64_t l = 0; l < k; ++l) {
	const std::int16_t *x = X + l * incC;
	for (std::int64_t i = 0; i < r; ++i) {
	    s[i] += std::uint32_t(std::int32_t(x[i * incR]));
	}
    }
}

//  rounded to nearest (halfway cases away from zero) and saturated; NaN
//  gives INT32_MIN
inline std::int32_t
toInt32(double x)
{
    x = std::fmin(std::fmax(x, -2147483648.0), 2147483647.0);
    return std::int32_t(std::round(x));
}

} // namespace

extern "C" {

void
CBLAS(gemm_s16s16s32)(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA,
                      enum CBLAS_TRANSPOSE transB, enum CBLAS_OFFSET offsetC,
                      ULMBLAS_INDEX_TYPE m, ULMBLAS_INDEX_TYPE n,
                      ULMBLAS_INDEX_TYPE k, float alpha, const int16_t *A,
                      ULMBLAS_INDEX_TYPE ldA, int16_t ao, const int16_t *B,
                      ULMBLAS_INDEX_TYPE ldB, int16_t bo, float beta,
                      int32_t *C, ULMBLAS_INDEX_TYPE ldC, const int32_t *co)
{
    const bool colMajor = order == CblasColMajor;
    const bool tA = transA == CblasTrans || transA == CblasConjTrans;
    const bool tB = transB == CblasTrans || transB == CblasConjTrans;

    const ULMBLAS_INDEX_TYPE ldAmin = colMajor ? (tA ? k : m) : (tA ? m : k);
    const ULMBLAS_INDEX_TYPE ldBmin = colMajor ? (tB ? n : k) : (tB ? k : n);
    const ULMBLAS_INDEX_TYPE ldCmin = colMajor ? m : n;

    int info = 0;
    if (order != CblasColMajor && order != CblasRowMajor) {
	info = 1;
    } else if (transA != CblasNoTrans && transA != CblasTrans &&
               transA != CblasConjTrans) {
	info = 2;
    } else if (transB != CblasNoTrans && transB != CblasTrans &&
               transB != CblasConjTrans) {
	info = 3;
    } else if (offsetC != CblasFixOffset && offsetC != CblasColOffset &&
               offsetC != CblasRowOffset) {
	info = 4;
    } else if (m < 0) {
	info = 5;
    } else if (n < 0) {
	info = 6;
    } else if (k < 0) {
	info = 7;
    } else if (ldA < std::max<ULMBLAS_INDEX_TYPE>(1, ldAmin)) {
	info = 10;
    } else if (ldB < std::max<ULMBLAS_INDEX_TYPE>(1, ldBmin)) {
	info = 13;
    } else if (ldC < std::max<ULMBLAS_INDEX_TYPE>(1, ldCmin)) {
	info = 17;
    }
    if (info != 0) {
	extern int RowMajorStrg;

	RowMajorStrg = colMajor ? 0 : 1;
	CBLAS(xerbla)(info, "cblas_gemm_s16s16s32", "");
	return;
    }
    if (m == 0 || n == 0) {
	return;
    }

    const std::int64_t lda = ldA, ldb = ldB, ldc = ldC;
    std::int64_t rA, cA, rB, cB, rC, cC;
    if (colMajor) {
	rA = tA ? lda : 1;
	cA = tA ? 1 : lda;
	rB = tB ? ldb : 1;
	cB = tB ? 1 : ldb;
	rC = 1;
	cC = ldc;
    } else {
	rA = tA ? 1 : lda;
	cA = tA ? lda : 1;
	rB = tB ? 1 : ldb;
	cB = tB ? ldb : 1;
	rC = ldc;
	cC = 1;
    }

    //  MemoryPool (posix_memalign, abort on failure): no functions that throw
    static ulmBLAS::MemoryPool<std::uint32_t> sumPool;
    static ulmBLAS::MemoryPool<std::int32_t> bufferPool;

    //  per row bo*rowsum(A), per column ao*colsum(B) + k*ao*bo, modulo 2^32
    std::uint32_t *rowAdd = sumPool.allocate(m);
    std::uint32_t *colAdd = sumPool.allocate(n);
    if (bo != 0) {
	sums16(m, k, A, rA, cA, rowAdd);
    }
    for (std::int64_t i = 0; i < m; ++i) {
	rowAdd[i] = bo != 0 ? std::uint32_t(std::int32_t(bo)) * rowAdd[i] : 0;
    }
    if (ao != 0) {
	sums16(n, k, B, cB, rB, colAdd);
    }
    const std::uint32_t aobok = std::uint32_t(std::int32_t(ao)) *
                                std::uint32_t(std::int32_t(bo)) *
                                std::uint32_t(std::uint64_t(k));
    for (std::int64_t j = 0; j < n; ++j) {
	colAdd[j] = (ao != 0 ? std::uint32_t(std::int32_t(ao)) * colAdd[j] : 0) +
	            aobok;
    }

    const std::int64_t NB = 1024;
    std::int32_t *S_ = bufferPool.allocate(m * std::min<std::int64_t>(n, NB));
    const double a = alpha, b = beta;

    for (std::int64_t j0 = 0; j0 < n; j0 += NB) {
	const std::int64_t nb = std::min<std::int64_t>(n - j0, NB);
	ulmBLAS::i16gemm(m, nb, k, 1, A, rA, cA, B + j0 * cB, rB, cB, 0, S_, 1,
	                 m);
	for (std::int64_t jj = 0; jj < nb; ++jj) {
	    const std::int64_t j = j0 + jj;
	    const std::int32_t *S = S_ + jj * m;
	    std::int32_t *Cj = C + j * cC;
	    const double coj = offsetC == CblasFixOffset   ? double(co[0])
	                       : offsetC == CblasRowOffset ? double(co[j])
	                                                   : 0.0;
	    for (std::int64_t i = 0; i < m; ++i) {
		//  separate statements: no fused multiply-adds
		const std::int32_t s = std::int32_t(std::uint32_t(S[i]) +
		                                    colAdd[j] + rowAdd[i]);
		double x = a * double(s);
		if (beta != 0.0f) {
		    const double y = b * double(Cj[i * rC]);
		    x = x + y;
		}
		const double o =
		    coj + (offsetC == CblasColOffset ? double(co[i]) : 0.0);
		x = x + o;
		Cj[i * rC] = toInt32(x);
	    }
	}
    }
    bufferPool.release(S_);
    sumPool.release(rowAdd);
    sumPool.release(colAdd);
}

} // extern "C"
