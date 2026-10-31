/*
 * Issue #0023 of ulmBLAS: Inf/NaN in B and the structural zeros of the
 * triangular matrix in cblas_dtrmm.  A standalone program for comparing
 * BLAS libraries (not part of make check), e.g.
 *
 *     cc -O2 trmm_inf.c -L... -lcblas_of_the_library && ./a.out NAME
 *
 *
 * 1. Reproducer: dtrmm(L, L, N, N, m=2, n=1, A=[1 *; 1 1], B=(1, Inf));
 *    reference BLAS gives (1, Inf).
 * 2. Systematic matrix: storage order x side x uplo x trans x diag x
 *    Inf/NaN; for every size s and every position p one Inf or NaN in B
 *    (row p for the left side, column p for the right side).  The result
 *    is compared with a loop over the referenced triangle only (as DTRMM
 *    without its zero test); the unreferenced triangle of A holds NaN, so
 *    reading it is detected as well.  Counted are entries that are NaN in
 *    the library but not in the reference ("spurious NaN") and all other
 *    differences.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BLASINT
#define BLASINT int
#endif
void cblas_dtrmm(int order, int side, int uplo, int trans, int diag,
                 BLASINT m, BLASINT n, double alpha, const double *A,
                 BLASINT lda, double *B, BLASINT ldb);

static int at(int row, int i, int j, int ld) { return row ? i*ld + j : i + j*ld; }

/* op(A)(i,k): referenced?  value */
static int
opA(int row, int lower, int trans, int unit, const double *A, int lda,
    int i, int k, double *v)
{
    int r = trans ? k : i, c = trans ? i : k;
    if (r == c) { *v = unit ? 1.0 : A[at(row, r, c, lda)]; return 1; }
    if (lower ? (r > c) : (r < c)) { *v = A[at(row, r, c, lda)]; return 1; }
    return 0;
}

int
main(int argc, char **argv)
{
    const char *lib = argc > 1 ? argv[1] : "?";
    const int sizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 33, 40, 100, 300};
    const int nsizes = sizeof(sizes)/sizeof(sizes[0]), other = 3;
    long tot_spur = 0, tot_other = 0, tot_cases = 0, tot_cases_spur = 0;

    {   /* reproducer */
        double A[4] = { 1, 1, NAN, 1 }, B[2] = { 1, INFINITY };
        cblas_dtrmm(102, 141, 122, 111, 131, 2, 1, 1.0, A, 2, B, 2);
        printf("%s reproducer dtrmm(L,L,N,N,m=2,n=1,B=(1,Inf)): (%g, %g)"
               "   reference: (1, inf)\n", lib, B[0], B[1]);
    }
    srand(1);
    for (int row = 0; row < 2; ++row)
    for (int special = 0; special < 2; ++special)
    for (int side = 0; side < 2; ++side)
    for (int lower = 0; lower < 2; ++lower)
    for (int trans = 0; trans < 2; ++trans)
    for (int unit = 0; unit < 2; ++unit) {
        double sv = special ? NAN : INFINITY;
        long spur = 0, cases = 0, cases_spur = 0, other_mis = 0;
        int maxblock = 0;
        for (int is = 0; is < nsizes; ++is) {
            int s = sizes[is];
            int m = side == 0 ? s : other, n = side == 0 ? other : s;
            int lda = s, ldb = row ? n : m;
            double *A = malloc(sizeof(double)*s*s), *B = malloc(sizeof(double)*m*n),
                   *R = malloc(sizeof(double)*m*n), *B0 = malloc(sizeof(double)*m*n);
            for (int i = 0; i < s; ++i)
                for (int j = 0; j < s; ++j)
                    A[at(row, i, j, lda)] = (lower ? i >= j : i <= j)
                                          ? 0.5 + rand()/(double)RAND_MAX : NAN;
            for (int p = 0; p < s; ++p) {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j)
                        B0[at(row, i, j, ldb)] = 0.5 + rand()/(double)RAND_MAX;
                if (side == 0) B0[at(row, p, 0, ldb)] = sv;
                else           B0[at(row, 0, p, ldb)] = sv;
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        double sum = 0, a;
                        for (int k = 0; k < s; ++k) {
                            if (side == 0 && opA(row, lower, trans, unit, A, lda, i, k, &a))
                                sum += a*B0[at(row, k, j, ldb)];
                            if (side == 1 && opA(row, lower, trans, unit, A, lda, k, j, &a))
                                sum += B0[at(row, i, k, ldb)]*a;
                        }
                        R[at(row, i, j, ldb)] = sum;
                    }
                memcpy(B, B0, sizeof(double)*m*n);
                cblas_dtrmm(row ? 101 : 102, side ? 142 : 141, lower ? 122 : 121,
                            trans ? 112 : 111, unit ? 132 : 131,
                            m, n, 1.0, A, lda, B, ldb);
                int bad = 0;
                for (int i = 0; i < m*n; ++i) {
                    if (isnan(B[i]) && !isnan(R[i])) {
                        ++bad;
                        /* distance (in the triangular dimension) between the
                           special value and the spurious NaN */
                        int ii = row ? i / ldb : i % ldb, jj = row ? i % ldb : i / ldb;
                        int d = abs((side == 0 ? ii : jj) - p) + 1;
                        if (d > maxblock) maxblock = d;
                    } else if (isnan(R[i]) != isnan(B[i]) || isinf(R[i]) != isinf(B[i])
                               || (isfinite(R[i]) && fabs(R[i]-B[i]) > 1e-12*fabs(R[i])*s))
                        ++other_mis;
                }
                ++cases;
                if (bad) { ++cases_spur; spur += bad; }
            }
            free(A); free(B); free(R); free(B0);
        }
        printf("%s %s %s side=%c uplo=%c trans=%c diag=%c: %4ld/%4ld placements "
               "with spurious NaN (%5ld entries, max distance %2d), other "
               "differences %ld\n", lib, row ? "row" : "col", special ? "NaN" : "Inf",
               side ? 'R' : 'L', lower ? 'L' : 'U', trans ? 'T' : 'N',
               unit ? 'U' : 'N', cases_spur, cases, spur, maxblock, other_mis);
        tot_spur += spur; tot_other += other_mis; tot_cases += cases;
        tot_cases_spur += cases_spur;
    }
    printf("%s total: %ld/%ld placements with spurious NaN (%ld entries), "
           "other differences %ld\n", lib, tot_cases_spur, tot_cases, tot_spur,
           tot_other);
    return 0;
}
