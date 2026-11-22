// The three unblocked variants of the Cholesky factorization A = L L^T,
// column major, lower triangle.  They differ in the order of the
// operations, not in the result: all three compute the same L, up to the
// order in which the terms of the sums are added.
#include "../common/util.h"
#include <vector>

// variant 1, bordered (Cholesky-Banachiewicz): row j of L from the rows
// above, by dot products
long
cholVar1(long n, double *A)
{
    for (long j = 0; j < n; ++j) {
	for (long k = 0; k < j; ++k) { // L(j,k), k < j: trsv
	    double s = A[j + k * n];
	    for (long l = 0; l < k; ++l) {
		s -= A[j + l * n] * A[k + l * n];
	    }
	    A[j + k * n] = s / A[k + k * n];
	}
	double d = A[j + j * n]; // L(j,j): dot
	for (long l = 0; l < j; ++l) {
	    d -= A[j + l * n] * A[j + l * n];
	}
	if (!(d > 0)) {
	    return j + 1;
	}
	A[j + j * n] = std::sqrt(d);
    }
    return 0;
}

// variant 2, left-looking (Cholesky-Crout): column j updated with all
// columns to its left, then scaled
long
cholVar2(long n, double *A)
{
    for (long j = 0; j < n; ++j) {
	double *c = &A[j * n];
	for (long k = 0; k < j; ++k) { // gemv, as one axpy per column
	    const double *a = &A[k * n];
	    const double b = a[j];
	    for (long i = j; i < n; ++i) {
		c[i] -= a[i] * b;
	    }
	}
	if (!(c[j] > 0)) {
	    return j + 1;
	}
	c[j] = std::sqrt(c[j]);
	for (long i = j + 1; i < n; ++i) { // scal
	    c[i] /= c[j];
	}
    }
    return 0;
}

// variant 3, right-looking: column j scaled, then the trailing matrix
// updated at once
long
cholVar3(long n, double *A)
{
    for (long j = 0; j < n; ++j) {
	double *a = &A[j * n];
	if (!(a[j] > 0)) {
	    return j + 1;
	}
	a[j] = std::sqrt(a[j]);
	for (long i = j + 1; i < n; ++i) { // scal
	    a[i] /= a[j];
	}
	for (long l = j + 1; l < n; ++l) { // syr
	    double *c = &A[l * n];
	    const double b = a[l];
	    for (long i = l; i < n; ++i) {
		c[i] -= a[i] * b;
	    }
	}
    }
    return 0;
}

int
main()
{
    const long n = 1500;
    std::vector<double> A0(n * n), W(n * n), A(n * n), L1(n * n);
    spdMatrix(n, A0.data(), W.data());

    long (*variant[3])(long, double *) = {cholVar1, cholVar2, cholVar3};
    std::printf("n = %ld, column major\n", n);
    for (int v = 0; v < 3; ++v) {
	A = A0;
	lap();
	const long info = variant[v](n, A.data());
	const double t = lap();
	if (v == 0) {
	    L1 = A;
	}
	double diff = 0; // largest difference to variant 1, lower triangle
	for (long j = 0; j < n; ++j) {
	    for (long i = j; i < n; ++i) {
		diff = std::max(diff, std::abs(A[i + j * n] - L1[i + j * n]));
	    }
	}
	std::printf("variant %d: info = %ld, %6.3f s, %5.2f GFLOPS, "
	            "max |L - L1| = %.1e\n",
	            v + 1, info, t, 1e-9 * n * n * n / 3 / t, diff);
    }
}
