/*
 *  flexiblas-test.c: dgemm and dgesv through FlexiBLAS (linked with
 *  -lflexiblas), run with FLEXIBLAS=ULMBLAS: the results must be right
 *  and ulmBLAS (on x86-64 a backend of the dispatching library) must be
 *  loaded (/proc/self/maps).  Part of the packaging dry run (Fedora).
 */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

void
dgemm_(const char *transA, const char *transB, const int *m, const int *n,
       const int *k, const double *alpha, const double *A, const int *ldA,
       const double *B, const int *ldB, const double *beta, double *C,
       const int *ldC, size_t lenTransA, size_t lenTransB);
void
dgesv_(const int *n, const int *nrhs, double *A, const int *ldA, int *ipiv,
       double *B, const int *ldB, int *info);

enum
{
    N = 200
};
static double A[N * N], B[N * N], C[N * N], M[N * N], x[N];
static int ipiv[N];

int
main(void)
{
    for (ptrdiff_t i = 0; i < N * N; ++i) {
	A[i] = (double)(i % 7) - 3;
	B[i] = (double)(i % 5) - 2;
    }
    int n = N, one = 1, info = -1;
    double alpha = 1, beta = 0;
    dgemm_("N", "N", &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n, 1, 1);
    double err = 0;
    for (ptrdiff_t j = 0; j < N; ++j) {
	for (ptrdiff_t i = 0; i < N; ++i) {
	    double s = 0;
	    for (ptrdiff_t l = 0; l < N; ++l) {
		s += A[i + l * N] * B[l + j * N];
	    }
	    double e = s - C[i + j * N];
	    err = e > err ? e : (-e > err ? -e : err);
	}
    }
    printf("dgemm: max error %g\n", err);

    /*  diagonally dominant M, x = M * ones */
    for (ptrdiff_t j = 0; j < N; ++j) {
	x[j] = 0;
	for (ptrdiff_t i = 0; i < N; ++i) {
	    M[i + j * N] = (i == j) ? 2.0 * N : (double)((i + 2 * j) % 3) - 1;
	}
    }
    for (ptrdiff_t j = 0; j < N; ++j) {
	for (ptrdiff_t i = 0; i < N; ++i) {
	    x[i] += M[i + j * N];
	}
    }
    dgesv_(&n, &one, M, &n, ipiv, x, &n, &info);
    double xerr = 0;
    for (ptrdiff_t i = 0; i < N; ++i) {
	double e = x[i] - 1;
	xerr = e > xerr ? e : (-e > xerr ? -e : xerr);
    }
    printf("dgesv: info %d, max error %g\n", info, xerr);

    int loaded = 0;
    FILE *maps = fopen("/proc/self/maps", "r");
    char line[4096];
    while (maps && fgets(line, sizeof(line), maps)) {
	if (strstr(line, "ulmblas")) {
	    const char *path = strchr(line, '/');
#if defined(__x86_64__) || defined(_M_X64)
	    //  the dispatching libulmblas.so loads a backend libulmblas_ISA.so
	    if (strstr(line, "libulmblas_")) {
		loaded = 1;
	    }
#else
	    //  one library with the kernels (no dispatch)
	    loaded = 1;
#endif
	    printf("    %s", path ? path : line);
	}
    }
    if (maps) {
	fclose(maps);
    }
    printf("ulmBLAS backend loaded: %s\n", loaded ? "yes" : "no");
    return err == 0 && info == 0 && xerr < 1e-12 && loaded ? 0 : 1;
}
