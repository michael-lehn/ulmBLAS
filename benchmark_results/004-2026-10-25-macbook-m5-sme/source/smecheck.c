/*
 *  smecheck: randomized correctness test of cblas_dgemm against a naive
 *  product (long double accumulation).
 *
 *      smecheck [SEED [COUNT]]
 *
 *  Each case: order (column or row major), transA, transB, m, n, k, alpha,
 *  beta and leading dimensions with padding are drawn at random; sizes are
 *  drawn from classes (tiny 0..8, around MR/NR multiples, around the block
 *  sizes 384, 608, 1024, 2240, 3072 (those of the SME2 configuration and of
 *  BLIS), and up to 700) so that partial
 *  tiles and multiple blocks of all loops occur.  The flop count of a case
 *  is limited to keep the naive product cheap.
 *
 *  Checks per element: |C - C_ref| <= 2*(k+2)*eps*(|alpha|*(|A||B|) +
 *  |beta*C0|) (elementwise bound).  With beta = 0, C is filled with NaN
 *  before the call (C must be overwritten).  Additional cases with NaN and
 *  Inf in A or B check that they propagate as in the reference (an element
 *  of C is NaN in the result iff it is NaN in the reference).  Elements
 *  outside of the m x n part of C (padding) must not change.
 *
 *  Output: one line per failure (at most 20) and a summary; exit status 1
 *  on failure.
 */
#include <cblas.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long state = 1;

static unsigned
rnd(void)
{
    state = state*6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned)(state >> 33);
}

static double
urand(void)
{
    return (double)rnd()/2147483648.0 - 0.5;
}

static int
size(int cls)
{
    static const int edges[] = {32, 16, 64, 96, 384, 608, 1024, 2240, 3072};
    switch (cls) {
    case 0:  return rnd()%9;
    case 1:  { int e = edges[rnd()%4]; return e - 2 + (int)(rnd()%5); }
    case 2:  return 1 + rnd()%700;
    default: { int e = edges[4 + rnd()%5]; return e - 1 + (int)(rnd()%3); }
    }
}

static const double scalars[] = {0.0, 1.0, -1.0, 0.5, 2.0, -0.75};

int
main(int argc, char **argv)
{
    unsigned long long seed = argc>1 ? strtoull(argv[1], 0, 10) : 1;
    long count = argc>2 ? atol(argv[2]) : 2000;
    long failures = 0, cases = 0, nanCases = 0;
    state = seed;

    for (long c=0; c<count; ++c) {
        int order = rnd()%2 ? CblasColMajor : CblasRowMajor;
        int ta    = rnd()%2 ? CblasTrans : CblasNoTrans;
        int tb    = rnd()%2 ? CblasTrans : CblasNoTrans;
        int m, n, k;
        do {
            m = size(rnd()%4); n = size(rnd()%4); k = size(rnd()%4);
        } while ((double)m*n*k > 6e7 || (double)m*n > 4e6
                 || (double)m*k > 8e6 || (double)n*k > 8e6);
        double alpha = scalars[rnd()%6], beta = scalars[rnd()%6];
        int special = (c%10)==9;            /* NaN/Inf in A or B */

        /* op(A) is m x k, op(B) k x n; stored dimensions */
        int col = order==CblasColMajor;
        int ar = (ta==CblasNoTrans) ? m : k, ac = (ta==CblasNoTrans) ? k : m;
        int br = (tb==CblasNoTrans) ? k : n, bc = (tb==CblasNoTrans) ? n : k;
        int lda = (col ? ar : ac) + (int)(rnd()%3);
        int ldb = (col ? br : bc) + (int)(rnd()%3);
        int ldc = (col ? m : n) + (int)(rnd()%3);
        if (lda<1) lda = 1;
        if (ldb<1) ldb = 1;
        if (ldc<1) ldc = 1;
        long sa = (long)lda*(col ? ac : ar), sb = (long)ldb*(col ? bc : br),
             sc = (long)ldc*(col ? n : m);
        if (sa<1) sa = 1;
        if (sb<1) sb = 1;
        if (sc<1) sc = 1;

        double *A = malloc(sizeof(double)*sa), *B = malloc(sizeof(double)*sb);
        double *C = malloc(sizeof(double)*sc), *C0 = malloc(sizeof(double)*sc);
        long double *R = malloc(sizeof(long double)*((long)m*n+1));
        double *E = malloc(sizeof(double)*((long)m*n+1));
        for (long i=0; i<sa; ++i) A[i] = urand();
        for (long i=0; i<sb; ++i) B[i] = urand();
        for (long i=0; i<sc; ++i) C0[i] = urand();
        if (special && m>0 && n>0 && k>0) {
            ++nanCases;
            A[rnd()%sa] = NAN;
            B[rnd()%sb] = INFINITY;
            A[rnd()%sa] = -INFINITY;
        }

#define AE(i,l) ((ta==CblasNoTrans) == col ? A[(long)(l)*lda+(i)] \
                                           : A[(long)(i)*lda+(l)])
#define BE(l,j) ((tb==CblasNoTrans) == col ? B[(long)(j)*ldb+(l)] \
                                           : B[(long)(l)*ldb+(j)])
#define CI(i,j) (col ? (long)(j)*ldc+(i) : (long)(i)*ldc+(j))

        /* reference with the conventions of the reference BLAS: beta = 0
           does not read C, alpha = 0 does not read A and B */
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                long double s = 0, b = 0;
                if (alpha!=0.0) {
                    for (int l=0; l<k; ++l) {
                        s += (long double)AE(i,l)*BE(l,j);
                        b += fabsl((long double)AE(i,l)*BE(l,j));
                    }
                }
                long double r = (long double)alpha*s;
                long double e = fabsl((long double)alpha)*b;
                if (beta!=0.0) {
                    r += (long double)beta*C0[CI(i,j)];
                    e += fabsl((long double)beta*C0[CI(i,j)]);
                }
                R[(long)i*n+j] = r;
                E[(long)i*n+j] = (double)(2.0L*(k+2)*DBL_EPSILON*e);
            }
        }
        memcpy(C, C0, sizeof(double)*sc);
        if (beta==0.0) {
            for (int i=0; i<m; ++i)
                for (int j=0; j<n; ++j)
                    C[CI(i,j)] = NAN;
        }
        cblas_dgemm(order, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta,
                    C, ldc);
        ++cases;

        int bad = 0;
        for (int i=0; i<m && !bad; ++i) {
            for (int j=0; j<n && !bad; ++j) {
                double x = C[CI(i,j)];
                long double r = R[(long)i*n+j];
                if (isnan(r) || isnan(x)) {
                    bad = isnan(r) != isnan(x);
                } else if (isinf(r) || isinf(x)) {
                    bad = (long double)x != r;
                } else {
                    bad = fabsl((long double)x - r) > E[(long)i*n+j];
                }
                if (bad && failures<20) {
                    printf("FAIL case %ld: %s %c%c m=%d n=%d k=%d lda=%d "
                           "ldb=%d ldc=%d alpha=%g beta=%g: C(%d,%d) = %.17g, "
                           "ref %.17Lg\n", c,
                           col ? "col" : "row", ta==CblasTrans ? 'T' : 'N',
                           tb==CblasTrans ? 'T' : 'N', m, n, k, lda, ldb, ldc,
                           alpha, beta, i, j, x, r);
                }
            }
        }
        /* padding of C unchanged */
        for (long p=0; p<sc && !bad; ++p) {
            long i = col ? p%ldc : p/ldc, j = col ? p/ldc : p%ldc;
            int inside = col ? (i<m && j<n) : (j<n && i<m);
            if (!col) { inside = (i<m && j<n); }
            if (!inside && memcmp(&C[p], &C0[p], sizeof(double))) {
                bad = 1;
                if (failures<20) {
                    printf("FAIL case %ld: padding of C changed at %ld\n",
                           c, p);
                }
            }
        }
        failures += bad;
        free(A); free(B); free(C); free(C0); free(R); free(E);
    }
    printf("seed %llu: %ld cases (%ld with NaN/Inf), %ld failures\n",
           seed, cases, nanCases, failures);
    return failures!=0;
}
