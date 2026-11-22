// The index of the element of largest absolute value (as iamax of the
// BLAS (Basic Linear Algebra Subprograms)): first the maximum with
// reduction(max : ...), then the first index with that value, searched in
// blocks of 16 with reduction(| : ...).
#include <cmath>
#include <cstdio>
#include <vector>

long
iamax(long n, const double *x)
{
    if (n <= 0) {
	return -1;
    }
    double max = 0;
#pragma omp simd reduction(max : max)
    for (long i = 0; i < n; ++i) {
	max = std::fmax(max, std::fabs(x[i]));
    }
    long j = 0;
    for (; j + 16 <= n; j += 16) {
	int hit = 0;
#pragma omp simd reduction(| : hit)
	for (int k = 0; k < 16; ++k) {
	    hit |= std::fabs(x[j + k]) == max;
	}
	if (hit) {
	    break; // the first index is in this block
	}
    }
    while (std::fabs(x[j]) != max) {
	++j;
    }
    return j;
}

int
main()
{
    const long n = 1000;
    std::vector<double> x(n);
    for (long i = 0; i < n; ++i) {
	x[i] = std::sin(double(i));
    }
    x[123] = -5;
    x[700] = 5;
    std::printf("iamax = %ld\n", iamax(n, x.data()));
}
