/*
 *  cblas_sbgemm and sbgemm_ (A, B bfloat16; alpha, beta, C float) against
 *  a reference in double computed from the same bfloat16 values: both
 *  orders, all transpositions, leading dimensions with padding, several
 *  alpha and beta (beta = 0 must not read C), sizes around the block
 *  sizes.  Error bound per element of C:
 *      4 (k + 2) eps_float (|alpha| sum_l |a_il b_lj| + |beta| |c_ij|)
 *  Elements outside the m x n block must not change.  Exit status 1 on
 *  failure.  Used by check-shared.sh.
 */
#include <cblas.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void
sbgemm_(const char *, const char *, const ULMBLAS_INDEX_TYPE *,
        const ULMBLAS_INDEX_TYPE *, const ULMBLAS_INDEX_TYPE *, const float *,
        const uint16_t *, const ULMBLAS_INDEX_TYPE *, const uint16_t *,
        const ULMBLAS_INDEX_TYPE *, const float *, float *,
        const ULMBLAS_INDEX_TYPE *);

static float
bf2f(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float x;
    memcpy(&x, &u, 4);
    return x;
}

static uint16_t
f2bf(float x)
{
    uint32_t u;
    memcpy(&u, &x, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

static double
urand(void)
{
    return (double)rand() / (double)RAND_MAX - 0.5;
}

//  element (i, j) of op(X) for X stored column major with leading dim ld
static double
opget(const uint16_t *X, int ld, int trans, int i, int j)
{
    return bf2f(trans ? X[j + (size_t)i * ld] : X[i + (size_t)j * ld]);
}

static int failures = 0, checks = 0;

static void
check(int rowmajor, int tA, int tB, int m, int n, int k, float alpha,
      float beta, int pad, int f77)
{
    //  column-major view: for row major, C^T = op(B)^T op(A)^T
    int ra = tA ? k : m, ca = tA ? m : k;
    int rb = tB ? n : k, cb = tB ? k : n;
    if (rowmajor) {
	int t = ra;
	ra = ca;
	ca = t;
	t = rb;
	rb = cb;
	cb = t;
    }
    int ldA = ra + pad, ldB = rb + pad;
    int ldC = (rowmajor ? n : m) + pad;
    if (ldA < 1)
	ldA = 1;
    if (ldB < 1)
	ldB = 1;
    if (ldC < 1)
	ldC = 1;
    size_t sA = (size_t)ldA * (ca > 0 ? ca : 1),
           sB = (size_t)ldB * (cb > 0 ? cb : 1);
    size_t sC = (size_t)ldC * ((rowmajor ? m : n) > 0 ? (rowmajor ? m : n) : 1);
    uint16_t *A = malloc(sA * 2), *B = malloc(sB * 2);
    float *C = malloc(sC * 4), *C0 = malloc(sC * 4);
    for (size_t i = 0; i < sA; ++i)
	A[i] = f2bf((float)urand());
    for (size_t i = 0; i < sB; ++i)
	B[i] = f2bf((float)urand());
    //  beta = 0: C is NaN, which must not reach the result
    for (size_t i = 0; i < sC; ++i)
	C0[i] = C[i] = (beta == 0.0f) ? NAN : (float)urand();

    if (f77) {
	ULMBLAS_INDEX_TYPE m_ = m, n_ = n, k_ = k, lA = ldA, lB = ldB, lC = ldC;
	sbgemm_(tA ? "T" : "N", tB ? "T" : "N", &m_, &n_, &k_, &alpha, A, &lA,
	        B, &lB, &beta, C, &lC);
    } else {
	cblas_sbgemm(rowmajor ? CblasRowMajor : CblasColMajor,
	             tA ? CblasTrans : CblasNoTrans,
	             tB ? CblasTrans : CblasNoTrans, m, n, k, alpha, A, ldA, B,
	             ldB, beta, C, ldC);
    }
    double maxq = 0;
    int bad = 0;
    for (int i = 0; i < m; ++i) {
	for (int j = 0; j < n; ++j) {
	    double s = 0, sa = 0;
	    for (int l = 0; l < k; ++l) {
		double a, b;
		if (!rowmajor) {
		    a = opget(A, ldA, tA, i, l);
		    b = opget(B, ldB, tB, l, j);
		} else {
		    //  row major A(i,l) = colmajor A^T: stored at (l,i)
		    a = tA ? bf2f(A[i + (size_t)l * ldA])
		           : bf2f(A[l + (size_t)i * ldA]);
		    b = tB ? bf2f(B[l + (size_t)j * ldB])
		           : bf2f(B[j + (size_t)l * ldB]);
		}
		s += a * b;
		sa += fabs(a * b);
	    }
	    size_t ic = rowmajor ? j + (size_t)i * ldC : i + (size_t)j * ldC;
	    double c0 = C0[ic];
	    double ref = alpha * s + (beta == 0 ? 0 : beta * c0);
	    double bc = (beta == 0) ? 0 : fabs(beta) * fabs(c0);
	    double bound =
	        4.0 * (k + 2) * 1.2e-7 * (fabs(alpha) * sa + bc) + 1e-30;
	    double q = fabs(C[ic] - ref) / bound;
	    if (!(q <= 1))
		bad = 1;
	    if (q > maxq || q != q)
		maxq = q;
	}
    }
    //  elements outside the m x n block unchanged
    for (size_t idx = 0; idx < sC; ++idx) {
	size_t r = idx % ldC, c = idx / ldC;
	int inside = rowmajor ? (r < (size_t)n && c < (size_t)m)
	                      : (r < (size_t)m && c < (size_t)n);
	if (!inside && memcmp(&C[idx], &C0[idx], 4) != 0)
	    bad = 2;
    }
    ++checks;
    if (bad) {
	++failures;
	printf("FAIL %s %s tA=%d tB=%d m=%d n=%d k=%d alpha=%g beta=%g pad=%d "
	       "err/bound=%g%s\n",
	       f77 ? "f77" : "cblas", rowmajor ? "row" : "col", tA, tB, m, n, k,
	       alpha, beta, pad, maxq, bad == 2 ? " (outside changed)" : "");
    }
    free(A);
    free(B);
    free(C);
    free(C0);
}

int
main(void)
{
    int sizes[] = {0, 1, 2, 7, 16, 31, 32, 33, 63, 65, 100, 257, 513};
    float ab[][2] = {{1, 0}, {1, 1}, {-0.5f, 2}, {0, 0.5f}, {2.5f, 0}};
    int ns = sizeof sizes / sizeof sizes[0];
    for (int f77 = 0; f77 < 2; ++f77)
	for (int ord = 0; ord < (f77 ? 1 : 2); ++ord)
	    for (int tA = 0; tA < 2; ++tA)
		for (int tB = 0; tB < 2; ++tB)
		    for (int s = 0; s < 40; ++s) {
			int m = sizes[rand() % ns], n = sizes[rand() % ns],
			    k = sizes[rand() % ns];
			int a = rand() % 5;
			check(ord, tA, tB, m, n, k, ab[a][0], ab[a][1],
			      rand() % 3, f77);
		    }
    /*  larger, crosses the blocks KC, MC, NC of float on all kernels */
    check(0, 0, 0, 700, 2100, 1300, 1, 1, 3, 0);
    check(1, 1, 0, 1031, 517, 2049, -1, 0.5f, 0, 0);
    printf("%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
