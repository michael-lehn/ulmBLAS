/*
 *  msvc-test.c: an MSVC program (cl.exe) linked with the import libraries
 *  ulmblas.lib (and ulmlapack.lib with -DLAPACK): CBLAS, the Fortran 77
 *  names dgemm_, dgemm and DGEMM, and dgesv_.  Part of the packaging dry
 *  run (Windows).
 */
#include <stddef.h>
#include <stdio.h>

typedef void
Dgemm(const char *transA, const char *transB, const int *m, const int *n,
      const int *k, const double *alpha, const double *A, const int *ldA,
      const double *B, const int *ldB, const double *beta, double *C,
      const int *ldC, size_t lenTransA, size_t lenTransB);
Dgemm dgemm_, dgemm, DGEMM;
const char *
ulmblas_backend(void);
void
cblas_dgemm(int layout, int transA, int transB, int m, int n, int k,
            double alpha, const double *A, int ldA, const double *B, int ldB,
            double beta, double *C, int ldC);
#if defined(LAPACK)
void
dgesv_(const int *n, const int *nrhs, double *A, const int *ldA, int *ipiv,
       double *B, const int *ldB, int *info);
#endif

static int
check(const char *name, const double *C)
{
    int ok = C[0] == 1 && C[1] == 2 && C[2] == 3 && C[3] == 4;
    printf("%-12s %g %g %g %g %s\n", name, C[0], C[1], C[2], C[3],
           ok ? "ok" : "WRONG");
    return ok;
}

int
main(void)
{
    double A[4] = {1, 2, 3, 4}, B[4] = {1, 0, 0, 1};
    double C[4] = {0}, alpha = 1, beta = 0;
    int two = 2, ok = 1;
    Dgemm *f[3] = {dgemm_, dgemm, DGEMM};
    const char *names[3] = {"dgemm_", "dgemm", "DGEMM"};

    printf("backend %s\n", ulmblas_backend());
    cblas_dgemm(102, 111, 111, 2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);
    ok &= check("cblas_dgemm", C);
    for (ptrdiff_t i = 0; i < 3; ++i) {
	C[0] = C[1] = C[2] = C[3] = 0;
	f[i]("N", "N", &two, &two, &two, &alpha, A, &two, B, &two, &beta, C,
	     &two, 1, 1);
	ok &= check(names[i], C);
    }
#if defined(LAPACK)
    {
	double M[4] = {4, 1, 1, 3}, x[2] = {5, 4};
	int n = 2, one = 1, ipiv[2], info = -1;
	dgesv_(&n, &one, M, &n, ipiv, x, &n, &info);
	printf("dgesv_       info %d, x = %g %g\n", info, x[0], x[1]);
	ok &= info == 0 && x[0] > 0.999 && x[0] < 1.001;
    }
#endif
    return ok ? 0 : 1;
}
