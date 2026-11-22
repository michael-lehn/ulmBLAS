// Dividing by the pivot or multiplying with its reciprocal: for a tiny
// pivot 1/pivot overflows to infinity, although every quotient is
// representable.  getf2 of ulmBLAS (and LAPACK) multiplies only if
// |pivot| >= sfmin, the smallest number whose reciprocal does not
// overflow, and divides otherwise.
#include <cmath>
#include <cstdio>
#include <limits>

int
main()
{
    const double pivot = 1e-310; // subnormal
    const double x[3] = {2e-310, -5e-311, 1e-312};

    const double r = 1 / pivot;
    std::printf("1 / pivot = %g\n", r);
    for (int i = 0; i < 3; ++i) {
	std::printf("x / pivot = %-10g  x * (1 / pivot) = %g\n", x[i] / pivot,
	            x[i] * r);
    }

    const double sfmin = std::numeric_limits<double>::min(); // 2^-1022
    std::printf("sfmin = %g, 1 / sfmin = %g\n", sfmin, 1 / sfmin);
}
