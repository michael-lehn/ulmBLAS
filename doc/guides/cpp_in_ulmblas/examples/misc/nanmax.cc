// std::max and NaN: std::max(a, b) is (a < b) ? b : a, and every comparison
// with NaN is false.  So the result depends on the order of the arguments,
// and in the usual loop a NaN is lost.  std::fmax drops NaN on purpose.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

// NaN if a or b is NaN, otherwise the larger one
template <typename T>
T
maxNan(T a, T b)
{
    return (std::isnan(a) || a > b) ? a : b;
}

int
main()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();

    std::printf("std::max(nan, 1)  = %g\n", std::max(nan, 1.0));
    std::printf("std::max(1, nan)  = %g\n", std::max(1.0, nan));
    std::printf("std::fmax(nan, 1) = %g\n", std::fmax(nan, 1.0));
    std::printf("std::fmax(1, nan) = %g\n", std::fmax(1.0, nan));
    std::printf("maxNan(nan, 1)    = %g\n", maxNan(nan, 1.0));
    std::printf("maxNan(1, nan)    = %g\n", maxNan(1.0, nan));

    const double x[4] = {1, nan, 3, 2};
    double m1 = x[0], m2 = x[0];
    for (int i = 1; i < 4; ++i) {
	m1 = std::max(m1, x[i]);
	m2 = maxNan(m2, x[i]);
    }
    std::printf("maximum of 1, nan, 3, 2: std::max %g, maxNan %g\n", m1, m2);
}
