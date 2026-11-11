//
//  cblas_gemm_s8u8s32: 8-bit integer GEMM with the signature and the
//  semantics of oneMKL's function of that name,
//
//      C <- alpha*(op(A) + ao)*(op(B) + bo) + beta*C + co.
//
//  Column major: A int8_t and B uint8_t; row major: A uint8_t and B int8_t.
//  co: one value (CblasFixOffset), one per row (CblasColOffset, m values)
//  or one per column (CblasRowOffset, n values).
//
//  The product S = op(A)*op(B) is computed exactly by ulmBLAS::i8gemm, for
//  blocks of columns into a local buffer (in int32 for at most 65536 values
//  of k, more are summed in int64).  The offsets ao and bo are added with the
//  row sums of A and the column sums of B:
//
//      (A + ao)(B + bo) = S + ao*colsum(B) + bo*rowsum(A) + k*ao*bo.
//
//  alpha, beta and the conversion to int32 in double: rounded to nearest
//  (halfway cases away from zero, as oneMKL) and saturated to the range of
//  int32.  NaN (alpha or beta NaN) gives INT32_MIN.
//
#include BLAS_HEADER
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <interfaces/blas/C/xerbla.h>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/level3/i8gemm.h>

namespace {

//  s[i] = sum_l X(i,l), 0 <= i < r, X(i,l) at X[i*incR + l*incC]; in int32
//  for blocks of KB values of l (|sum| <= KB*255 < 2^31), then in int64
template <typename T>
void
sums(std::int64_t r, std::int64_t k, const T *X, std::int64_t incR,
     std::int64_t incC, std::int64_t *s)
{
    const std::int64_t KB = 1 << 22;
    if (incC == 1) {
	for (std::int64_t i = 0; i < r; ++i) {
	    const T *x = X + i * incR;
	    std::int64_t t = 0;
	    for (std::int64_t l0 = 0; l0 < k; l0 += KB) {
		const std::int64_t l1 = std::min(k, l0 + KB);
		std::int32_t u = 0;
		for (std::int64_t l = l0; l < l1; ++l) {
		    u += x[l];
		}
		t += u;
	    }
	    s[i] = t;
	}
	return;
    }
    //  per block of 256 rows the int32 sums in a local array
    for (std::int64_t i0 = 0; i0 < r; i0 += 256) {
	const std::int64_t rb = std::min<std::int64_t>(r - i0, 256);
	std::int64_t t[256] = {0};
	for (std::int64_t l0 = 0; l0 < k; l0 += KB) {
	    const std::int64_t l1 = std::min(k, l0 + KB);
	    std::int32_t u[256] = {0};
	    for (std::int64_t l = l0; l < l1; ++l) {
		const T *x = X + l * incC + i0 * incR;
		if (incR == 1) {
		    for (std::int64_t i = 0; i < rb; ++i) {
			u[i] += x[i];
		    }
		} else {
		    for (std::int64_t i = 0; i < rb; ++i) {
			u[i] += x[i * incR];
		    }
		}
	    }
	    for (std::int64_t i = 0; i < rb; ++i) {
		t[i] += u[i];
	    }
	}
	for (std::int64_t i = 0; i < rb; ++i) {
	    s[i0 + i] = t[i];
	}
    }
}

//  rounded to nearest (halfway cases away from zero) and saturated; NaN
//  gives INT32_MIN (fmax returns the other operand)
inline std::int32_t
toInt32(double x)
{
    x = std::fmin(std::fmax(x, -2147483648.0), 2147483647.0);
    return std::int32_t(std::round(x));
}

//
//  One column of C: c_i <- alpha*(S_i + add + rowAdd_i) + beta*c_i + co +
//  rowCo_i.  S_i + add + rowAdd_i is an integer below 2^53 in magnitude,
//  exact in double.  The products and sums are separate statements, so
//  that they are not contracted into fused multiply-adds (the result does
//  not depend on the compiler).
//
//
//  The same for alpha = 1 and beta = 0 or 1 in integers (the result is the
//  same: the double sums of integers below 2^53 are exact, no rounding).
//
template <typename TS>
void
epilogueInt(std::int64_t m, const TS *S, std::int64_t add,
            const std::int64_t *rowAdd, bool beta1, std::int32_t *C,
            std::int64_t incC, std::int64_t co, const std::int64_t *rowCo)
{
    for (std::int64_t i = 0; i < m; ++i) {
	std::int64_t x = std::int64_t(S[i]) + add + rowAdd[i] + co + rowCo[i];
	if (beta1) {
	    x += C[i * incC];
	}
	x = std::min<std::int64_t>(std::max<std::int64_t>(x, INT32_MIN),
	                           INT32_MAX);
	C[i * incC] = std::int32_t(x);
    }
}

template <typename TS>
void
epilogue(std::int64_t m, const TS *S, double add, const double *rowAdd,
         double alpha, double beta, std::int32_t *C, std::int64_t incC,
         double co, const double *rowCo)
{
    for (std::int64_t i = 0; i < m; ++i) {
	const double s = double(S[i]) + add + rowAdd[i];
	double x = alpha * s;
	if (beta != 0.0) {
	    const double y = beta * double(C[i * incC]);
	    x = x + y;
	}
	const double o = co + rowCo[i];
	x = x + o;
	C[i * incC] = toInt32(x);
    }
}

} // namespace

extern "C" {

void
CBLAS(gemm_s8u8s32)(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA,
                    enum CBLAS_TRANSPOSE transB, enum CBLAS_OFFSET offsetC,
                    ULMBLAS_INDEX_TYPE m, ULMBLAS_INDEX_TYPE n,
                    ULMBLAS_INDEX_TYPE k, float alpha, const void *A,
                    ULMBLAS_INDEX_TYPE ldA, int8_t ao, const void *B,
                    ULMBLAS_INDEX_TYPE ldB, int8_t bo, float beta, int32_t *C,
                    ULMBLAS_INDEX_TYPE ldC, const int32_t *co)
{
    const bool colMajor = order == CblasColMajor;
    const bool tA = transA == CblasTrans || transA == CblasConjTrans;
    const bool tB = transB == CblasTrans || transB == CblasConjTrans;

    //  the leading dimensions are at least the number of rows (column
    //  major) or columns (row major) of the arrays as stored
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
	CBLAS(xerbla)(info, "cblas_gemm_s8u8s32", "");
	return;
    }
    if (m == 0 || n == 0) {
	return;
    }

    //  strides of op(A) (m x k), op(B) (k x n) and C
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
    const bool unsignedA = !colMajor, unsignedB = colMajor;
    const std::int64_t KS = 65536;

    //  alpha = 1, beta = 0, no offsets: S directly into C (|S| < 2^31 for k
    //  <= KS, so neither wraparound nor saturation)
    if (alpha == 1.0f && beta == 0.0f && ao == 0 && bo == 0 && k <= KS &&
        offsetC == CblasFixOffset && co[0] == 0) {
	ulmBLAS::i8gemm(m, n, k, 1, unsignedA, A, rA, cA, unsignedB, B, rB, cB,
	                0, C, rC, cC);
	return;
    }

    //  MemoryPool (posix_memalign, abort on failure) instead of std::vector,
    //  so that the library does not depend on functions that throw
    static ulmBLAS::MemoryPool<std::int64_t> sumPool;
    static ulmBLAS::MemoryPool<std::int32_t> bufferPool;
    static ulmBLAS::MemoryPool<double> rowPool;

    //  per row: bo*rowsum(A) and the offset of the row (CblasColOffset), in
    //  double and in int64 (alpha = 1, beta = 0 or 1)
    const bool intPath = alpha == 1.0f && (beta == 0.0f || beta == 1.0f);
    double *rowAdd = rowPool.allocate(m);
    double *rowCo = rowPool.allocate(m);
    std::int64_t *rowAddI = sumPool.allocate(m);
    std::int64_t *rowCoI = sumPool.allocate(m);
    std::int64_t *sum = sumPool.allocate(std::max<std::int64_t>(m, n));
    if (bo != 0) {
	if (unsignedA) {
	    sums(m, k, static_cast<const std::uint8_t *>(A), rA, cA, sum);
	} else {
	    sums(m, k, static_cast<const std::int8_t *>(A), rA, cA, sum);
	}
    }
    for (std::int64_t i = 0; i < m; ++i) {
	rowAddI[i] = bo != 0 ? std::int64_t(bo) * sum[i] : 0;
	rowCoI[i] = offsetC == CblasColOffset ? co[i] : 0;
	rowAdd[i] = double(rowAddI[i]);
	rowCo[i] = double(rowCoI[i]);
    }
    //  per column: ao*colsum(B) + k*ao*bo (in colAdd, the sums reused)
    std::int64_t *colAdd = sum;
    const std::int64_t aobok = std::int64_t(ao) * std::int64_t(bo) * k;
    if (ao != 0) {
	if (unsignedB) {
	    sums(n, k, static_cast<const std::uint8_t *>(B), cB, rB, colAdd);
	} else {
	    sums(n, k, static_cast<const std::int8_t *>(B), cB, rB, colAdd);
	}
    }
    for (std::int64_t j = 0; j < n; ++j) {
	colAdd[j] = (ao != 0 ? std::int64_t(ao) * colAdd[j] : 0) + aobok;
    }

    //  S for blocks of NB columns into S_ (m x nb, column major).  For k >
    //  KS the sums over blocks of KS values of k (|sum| <= KS*128*255 <
    //  2^31, no wraparound) are added in int64 (S64_).
    const std::int64_t NB = 1024;
    const std::int64_t mnb = m * std::min<std::int64_t>(n, NB);
    std::int32_t *S_ = bufferPool.allocate(mnb);
    std::int64_t *S64_ = (k > KS) ? sumPool.allocate(mnb) : nullptr;
    const double a = alpha, b = beta;

    for (std::int64_t j0 = 0; j0 < n; j0 += NB) {
	const std::int64_t nb = std::min<std::int64_t>(n - j0, NB);
	const std::uint8_t *Bj = static_cast<const std::uint8_t *>(B) + j0 * cB;

	for (std::int64_t l0 = 0; l0 < k || l0 == 0; l0 += KS) {
	    const std::int64_t kb = std::min<std::int64_t>(k - l0, KS);
	    ulmBLAS::i8gemm(m, nb, kb, 1, unsignedA,
	                    static_cast<const std::uint8_t *>(A) + l0 * cA, rA,
	                    cA, unsignedB, Bj + l0 * rB, rB, cB, 0, S_, 1, m);
	    if (S64_) {
		for (std::int64_t q = 0; q < m * nb; ++q) {
		    S64_[q] = (l0 == 0 ? 0 : S64_[q]) + S_[q];
		}
	    }
	}

	for (std::int64_t jj = 0; jj < nb; ++jj) {
	    const std::int64_t j = j0 + jj;
	    const double addj = double(colAdd[j]);
	    const double coj = offsetC == CblasFixOffset   ? double(co[0])
	                       : offsetC == CblasRowOffset ? double(co[j])
	                                                   : 0.0;
	    std::int32_t *Cj = C + j * cC;
	    if (intPath) {
		const std::int64_t coI = offsetC == CblasFixOffset   ? co[0]
		                         : offsetC == CblasRowOffset ? co[j]
		                                                     : 0;
		if (S64_) {
		    epilogueInt(m, S64_ + jj * m, colAdd[j], rowAddI,
		                beta == 1.0f, Cj, rC, coI, rowCoI);
		} else {
		    epilogueInt(m, S_ + jj * m, colAdd[j], rowAddI,
		                beta == 1.0f, Cj, rC, coI, rowCoI);
		}
	    } else if (S64_) {
		epilogue(m, S64_ + jj * m, addj, rowAdd, a, b, Cj, rC, coj,
		         rowCo);
	    } else {
		epilogue(m, S_ + jj * m, addj, rowAdd, a, b, Cj, rC, coj,
		         rowCo);
	    }
	}
    }
    bufferPool.release(S_);
    if (S64_) {
	sumPool.release(S64_);
    }
    sumPool.release(sum);
    sumPool.release(rowAddI);
    sumPool.release(rowCoI);
    rowPool.release(rowAdd);
    rowPool.release(rowCo);
}

} // extern "C"
