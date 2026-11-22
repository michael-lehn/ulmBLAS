// A x = b solved twice: with the LU factors (getrf, getrs) and with the
// inverse (getrf, getri, x = inv(A) b).  For an ill-conditioned matrix the
// solution through the inverse has a much larger residual.
#include "../common/util.h"
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/getri.h>
#include <ulmblas/lapack/getrs.h>
#include <vector>

// ||b - A x|| / (||A|| ||x||) in the maximum norm
double
residual(long n, const double *A, const double *x, const double *b)
{
    double r = 0, nx = 0;
    for (long i = 0; i < n; ++i) {
	double s = b[i];
	for (long j = 0; j < n; ++j) {
	    s -= A[i + j * n] * x[j];
	}
	r = std::max(r, std::abs(s));
	nx = std::max(nx, std::abs(x[i]));
    }
    return r / (normMax(n, n, A, n) * nx);
}

int
main()
{
    const long n = 8;
    std::vector<double> A(n * n), LU(n * n), x(n), b(n), x1(n), x2(n);
    std::vector<long> piv(n);
    for (long j = 0; j < n; ++j) { // the Hilbert matrix, A(i,j) = 1/(i+j+1)
	for (long i = 0; i < n; ++i) {
	    A[i + j * n] = 1.0 / double(i + j + 1);
	}
    }
    for (long i = 0; i < n; ++i) { // b = A x with x = (1, ..., 1)
	x[i] = 1;
	b[i] = 0;
	for (long j = 0; j < n; ++j) {
	    b[i] += A[i + j * n];
	}
    }

    // 1. with the factors
    LU = A;
    x1 = b;
    ulmBLAS::getrf(n, n, LU.data(), 1L, n, piv.data(), 1L);
    ulmBLAS::getrs(false, false, n, 1L, LU.data(), 1L, n, piv.data(), 1L,
                   x1.data(), 1L, n);

    // 2. with the inverse
    ulmBLAS::getri(n, LU.data(), 1L, n, piv.data(), 1L);
    ulmBLAS::gemv(n, n, 1.0, false, LU.data(), 1L, n, b.data(), 1L, 0.0,
                  x2.data(), 1L);

    double e1 = 0, e2 = 0;
    for (long i = 0; i < n; ++i) {
	e1 = std::max(e1, std::abs(x1[i] - 1));
	e2 = std::max(e2, std::abs(x2[i] - 1));
    }
    const double eps = std::numeric_limits<double>::epsilon();
    std::printf("Hilbert matrix, n = %ld\n", n);
    std::printf("LU factors: residual = %5.1f eps, error = %.1e\n",
                residual(n, A.data(), x1.data(), b.data()) / eps, e1);
    std::printf("inverse:    residual = %5.1f eps, error = %.1e\n",
                residual(n, A.data(), x2.data(), b.data()) / eps, e2);
}
