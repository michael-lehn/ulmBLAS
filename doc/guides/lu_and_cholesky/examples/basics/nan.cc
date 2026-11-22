// The test for a positive diagonal element in a Cholesky factorization:
// with NaN every comparison is false.  "a <= 0" lets a NaN through,
// "!(a > 0)" stops it.
#include <cmath>
#include <cstdio>
#include <limits>

// unblocked, right-looking; returns info as LAPACK (0 or j + 1)
template <bool Careful>
long
cholesky(long n, double *A)
{
    for (long j = 0; j < n; ++j) {
	const double a = A[j + j * n];
	if (Careful ? !(a > 0) : a <= 0) {
	    return j + 1;
	}
	A[j + j * n] = std::sqrt(a);
	for (long i = j + 1; i < n; ++i) {
	    A[i + j * n] /= A[j + j * n];
	}
	for (long l = j + 1; l < n; ++l) {
	    for (long i = l; i < n; ++i) {
		A[i + l * n] -= A[i + j * n] * A[l + j * n];
	    }
	}
    }
    return 0;
}

int
main()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // lower triangle of a 3 x 3 matrix with a NaN below the diagonal
    double A[9] = {4, nan, 2, 0, 5, 1, 0, 0, 6}, B[9];
    for (int i = 0; i < 9; ++i) {
	B[i] = A[i];
    }
    const long info1 = cholesky<false>(3, A);
    const long info2 = cholesky<true>(3, B);
    std::printf("a <= 0:    info = %ld, L(2,2) = %g\n", info1, A[8]);
    std::printf("!(a > 0):  info = %ld\n", info2);
}
