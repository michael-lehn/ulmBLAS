/*
 *  8-bit integer GEMM, against plain loops (exact comparison):
 *
 *  - ulmblas_i8i32gemm_strided: C (int32) <- alpha*A*B + beta*C modulo
 *    2^32, A and B signed or unsigned, column major, row major, negative
 *    and non-unit strides.
 *  - cblas_gemm_s8u8s32 (oneMKL's signature): both orders, all
 *    transpositions, offsets ao, bo, the three kinds of offsets of C,
 *    float alpha and beta (rounded to nearest, halfway away from zero,
 *    saturated), k beyond 65536 (int32 sums would wrap).
 *
 *  Elements outside C must not change.  Sizes around the tile (32) and
 *  block sizes.  Exit status 1 on failure.  Used by check-shared.sh.
 */
#include <cblas.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ulmblas_i8i32gemm_strided(int64_t m, int64_t n, int64_t k, int32_t alpha,
                               int unsignedA, const void *A, int64_t incRowA,
                               int64_t incColA, int unsignedB, const void *B,
                               int64_t incRowB, int64_t incColB, int32_t beta,
                               int32_t *C, int64_t incRowC, int64_t incColC);

static uint64_t seed = 0x8a5cd789635d2dffULL;

static uint32_t
rnd(void)
{
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(seed >> 33);
}

static int failures = 0;

static int64_t
val(const uint8_t *X, int64_t q, int isUnsigned)
{
    return isUnsigned ? (int64_t)X[q] : (int64_t)(int8_t)X[q];
}

/*  the extent of an r x c operand with strides rs, cs */
static void
extent(int64_t r, int64_t c, int64_t rs, int64_t cs, int64_t *lo, int64_t *hi)
{
    *lo = (rs < 0 ? rs * (r - 1) : 0) + (cs < 0 ? cs * (c - 1) : 0);
    *hi = (rs > 0 ? rs * (r - 1) : 0) + (cs > 0 ? cs * (c - 1) : 0);
}

static void
strided(int64_t m, int64_t n, int64_t k, int ua, int ub, int layout,
        int32_t alpha, int32_t beta)
{
    int64_t rA, cA, rB, cB, rC, cC, loA, hiA, loB, hiB, loC, hiC;
    int64_t M = m > 0 ? m : 1, N = n > 0 ? n : 1, K = k > 0 ? k : 1;
    switch (layout) {
    case 0: /* column major */
	rA = 1, cA = M + 1, rB = 1, cB = K + 3, rC = 1, cC = M + 2;
	break;
    case 1: /* row major */
	rA = K + 1, cA = 1, rB = N, cB = 1, rC = N + 1, cC = 1;
	break;
    case 2: /* A row major, B column major (k contiguous for both) */
	rA = K, cA = 1, rB = 1, cB = K, rC = 1, cC = M;
	break;
    default: /* negative and non-unit strides, C not unit stride */
	rA = -1, cA = M, rB = 3, cB = 3 * K + 1, rC = 2, cC = 2 * M + 1;
	break;
    }
    extent(M, K, rA, cA, &loA, &hiA);
    extent(K, N, rB, cB, &loB, &hiB);
    extent(M, N, rC, cC, &loC, &hiC);
    uint8_t *Ab = malloc(hiA - loA + 1), *Bb = malloc(hiB - loB + 1);
    int32_t *Cb = malloc((hiC - loC + 1) * 4), *C0 = malloc((hiC - loC + 1) * 4);
    char *in = calloc(hiC - loC + 1, 1);
    for (int64_t q = 0; q <= hiA - loA; ++q)
	Ab[q] = (uint8_t)rnd();
    for (int64_t q = 0; q <= hiB - loB; ++q)
	Bb[q] = (uint8_t)rnd();
    for (int64_t q = 0; q <= hiC - loC; ++q)
	Cb[q] = C0[q] = (int32_t)rnd();
    const uint8_t *A = Ab - loA, *B = Bb - loB;
    int32_t *C = Cb - loC;

    ulmblas_i8i32gemm_strided(m, n, k, alpha, ua, A, rA, cA, ub, B, rB, cB,
                              beta, C, rC, cC);

    for (int64_t j = 0; j < n; ++j) {
	for (int64_t i = 0; i < m; ++i) {
	    uint32_t s = 0;
	    for (int64_t l = 0; l < k; ++l) {
		s += (uint32_t)(val(A, i * rA + l * cA, ua) *
		                val(B, l * rB + j * cB, ub));
	    }
	    int64_t q = i * rC + j * cC - loC;
	    uint32_t ref = (uint32_t)alpha * s;
	    if (beta != 0) {
		ref += (uint32_t)beta * (uint32_t)C0[q];
	    }
	    in[q] = 1;
	    if ((uint32_t)Cb[q] != ref) {
		if (failures++ < 10) {
		    printf("i8i32gemm_strided m=%lld n=%lld k=%lld ua=%d "
		           "ub=%d layout=%d alpha=%d beta=%d: C(%lld,%lld) = "
		           "%d, expected %d\n",
		           (long long)m, (long long)n, (long long)k, ua, ub,
		           layout, alpha, beta, (long long)i, (long long)j,
		           Cb[q], (int32_t)ref);
		}
		goto done;
	    }
	}
    }
    for (int64_t q = 0; q <= hiC - loC; ++q) {
	if (!in[q] && Cb[q] != C0[q]) {
	    if (failures++ < 10) {
		printf("i8i32gemm_strided m=%lld n=%lld k=%lld layout=%d: "
		       "element outside C modified\n",
		       (long long)m, (long long)n, (long long)k, layout);
	    }
	    break;
	}
    }
done:
    free(Ab), free(Bb), free(Cb), free(C0), free(in);
}

/*  rounded to nearest, halfway cases away from zero, saturated */
static int32_t
toInt32(double x)
{
    if (!(x > -2147483648.5))
	return INT32_MIN;
    if (x >= 2147483647.0)
	return INT32_MAX;
    return (int32_t)round(x);
}

static void
mkl(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE tA_, enum CBLAS_TRANSPOSE tB_,
    enum CBLAS_OFFSET offC, int m, int n, int k, float alpha, int8_t ao,
    int8_t bo, float beta, int big)
{
    int col = order == CblasColMajor;
    int tA = tA_ != CblasNoTrans, tB = tB_ != CblasNoTrans;
    /*  the arrays as stored: rows x cols (column major) or rows x cols (row
     *  major), with padding of the leading dimension */
    int ra = col ? (tA ? k : m) : (tA ? k : m);
    int ca = col ? (tA ? m : k) : (tA ? m : k);
    int rb = tB ? n : k, cb = tB ? k : n;
    int ldA = (col ? ra : ca) + 3, ldB = (col ? rb : cb) + 1,
        ldC = (col ? m : n) + 2;
    size_t sA = (size_t)ldA * (col ? (ca > 0 ? ca : 1) : (ra > 0 ? ra : 1));
    size_t sB = (size_t)ldB * (col ? (cb > 0 ? cb : 1) : (rb > 0 ? rb : 1));
    size_t sC = (size_t)ldC * (col ? (n > 0 ? n : 1) : (m > 0 ? m : 1));
    uint8_t *A = malloc(sA), *B = malloc(sB);
    int32_t *C = malloc(sC * 4), *C0 = malloc(sC * 4), co[4096];
    for (size_t q = 0; q < sA; ++q)
	A[q] = big ? 0x7f : (uint8_t)rnd();
    for (size_t q = 0; q < sB; ++q)
	B[q] = big ? 0xff : (uint8_t)rnd();
    for (size_t q = 0; q < sC; ++q)
	C[q] = C0[q] = (int32_t)(rnd() % 2000001) - 1000000;
    for (int q = 0; q < 4096; ++q)
	co[q] = (int32_t)(rnd() % 20001) - 10000;
    /*  strides of op(A), op(B), C */
    int64_t rA = col ? (tA ? ldA : 1) : (tA ? 1 : ldA);
    int64_t cA = col ? (tA ? 1 : ldA) : (tA ? ldA : 1);
    int64_t rB = col ? (tB ? ldB : 1) : (tB ? 1 : ldB);
    int64_t cB = col ? (tB ? 1 : ldB) : (tB ? ldB : 1);
    int64_t rC = col ? 1 : ldC, cC = col ? ldC : 1;
    int ua = !col, ub = col;

    cblas_gemm_s8u8s32(order, tA_, tB_, offC, m, n, k, alpha, A, ldA, ao, B,
                       ldB, bo, beta, C, ldC, co);

    char *in = calloc(sC, 1);
    for (int j = 0; j < n; ++j) {
	for (int i = 0; i < m; ++i) {
	    int64_t s = 0;
	    for (int l = 0; l < k; ++l) {
		s += (val(A, i * rA + l * cA, ua) + ao) *
		     (val(B, l * rB + j * cB, ub) + bo);
	    }
	    int64_t q = i * rC + j * cC;
	    /*  as specified: alpha*s + beta*c + co in double, without fused
	     *  multiply-adds (volatile) */
	    volatile double x = (double)alpha * (double)s;
	    if (beta != 0.0f) {
		volatile double y = (double)beta * (double)C0[q];
		x = x + y;
	    }
	    x = x + (double)(offC == CblasFixOffset   ? co[0]
	                     : offC == CblasColOffset ? co[i]
	                                              : co[j]);
	    int32_t ref = toInt32(x);
	    in[q] = 1;
	    if (C[q] != ref) {
		if (failures++ < 10) {
		    printf("gemm_s8u8s32 %s tA=%d tB=%d off=%d m=%d n=%d k=%d "
		           "alpha=%g ao=%d bo=%d beta=%g: C(%d,%d) = %d, "
		           "expected %d\n",
		           col ? "col" : "row", tA, tB, (int)offC, m, n, k,
		           alpha, ao, bo, beta, i, j, C[q], ref);
		}
		goto done;
	    }
	}
    }
    for (size_t q = 0; q < sC; ++q) {
	if (!in[q] && C[q] != C0[q]) {
	    if (failures++ < 10)
		printf("gemm_s8u8s32 m=%d n=%d k=%d: element outside C "
		       "modified\n",
		       m, n, k);
	    break;
	}
    }
done:
    free(A), free(B), free(C), free(C0), free(in);
}

int
main(void)
{
    static const int64_t size[][3] = {
        {1, 1, 1},     {2, 3, 5},       {7, 5, 3},     {31, 33, 4},
        {32, 32, 4},   {33, 31, 7},     {64, 64, 64},  {97, 113, 301},
        {100, 40, 17}, {530, 70, 4100}, {40, 1100, 9}, {513, 1030, 4097},
        {0, 5, 3},     {4, 0, 3},       {4, 5, 0}};
    static const int32_t ab[][2] = {{1, 0}, {1, 1}, {-3, 5}, {7, 0}, {0, 3}};
    int c = 0, cases = 0;
    for (size_t s = 0; s < sizeof(size) / sizeof(size[0]); ++s) {
	for (int u = 0; u < 4; ++u) {
	    for (int lay = 0; lay < 4; ++lay, ++c, ++cases) {
		strided(size[s][0], size[s][1], size[s][2], u & 1, u >> 1, lay,
		        ab[c % 5][0], ab[c % 5][1]);
	    }
	}
    }

    static const int msize[][3] = {{1, 1, 1},  {3, 2, 5},   {33, 31, 7},
                                   {64, 1, 40}, {70, 90, 133}, {0, 3, 2},
                                   {3, 0, 2},   {3, 2, 0}};
    static const float alphas[] = {1.0f, 0.5f, -2.5f, 1e9f};
    static const float betas[] = {0.0f, 1.0f, -0.5f};
    static const enum CBLAS_OFFSET offs[] = {CblasFixOffset, CblasColOffset,
                                             CblasRowOffset};
    c = 0;
    for (size_t s = 0; s < sizeof(msize) / sizeof(msize[0]); ++s) {
	for (int o = 0; o < 2; ++o) {
	    for (int t = 0; t < 4; ++t, ++c, ++cases) {
		mkl(o ? CblasRowMajor : CblasColMajor,
		    (t & 1) ? CblasTrans : CblasNoTrans,
		    (t & 2) ? CblasConjTrans : CblasNoTrans, offs[c % 3],
		    msize[s][0], msize[s][1], msize[s][2], alphas[c % 4],
		    (int8_t)(c % 3 ? -3 : 0), (int8_t)(c % 2 ? 0 : 5), betas[c % 3],
		    0);
	    }
	}
    }
    /*  int32 sums would wrap: k = 80000, 127*255 */
    mkl(CblasColMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset, 2, 3, 80000,
        0.5f, 0, 0, 0.0f, 1);
    mkl(CblasRowMajor, CblasTrans, CblasNoTrans, CblasRowOffset, 3, 2, 70000,
        1.0f, 1, -1, 1.0f, 1);
    cases += 2;

    printf("%d cases, %d failures\n", cases, failures);
    return failures != 0;
}
