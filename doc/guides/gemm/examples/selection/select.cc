// The run-time selection of ulmBLAS: the detected platform, and for some
// shapes of dgemm (column major) the implementation that the selection
// table chooses and the speed of ulmBLAS::gemm.  The environment variable
// ULMBLAS_PLATFORM=generic selects the default table of the build instead
// of the table measured for the platform.
#include "../common/util.h"
#include <ulmblas/ulmblas.h>
#include <vector>

int
main()
{
    std::printf("platform: %s\n", ulmBLAS::platformName(ulmBLAS::platform()));
    struct
    {
	long m, n, k;
    } shapes[] = {{8, 8, 8},          {64, 64, 64},     {256, 256, 256},
                  {2000, 2000, 2000}, {2000, 2000, 16}, {2000, 16, 2000},
                  {16, 2000, 2000}};
    std::printf("%5s %5s %5s  %-32s %8s\n", "m", "n", "k", "selected",
                "GFLOPS");
    for (const auto &s : shapes) {
	const long m = s.m, n = s.n, k = s.k;
	std::vector<double> A(m * k), B(k * n), C(m * n);
	randomMatrix(m, k, A.data(), 1, m, 1);
	randomMatrix(k, n, B.data(), 1, k, 2);
	const ulmBLAS::GemmProblem<long> problem = {m, n, k, 1,     m,     1,
	                                            k, 1, m, false, false, 1};
	const auto plan =
	    ulmBLAS::gemmSelect<long, double, double, double, double, double>(
	        problem);
	const long reps = std::max(1L, long(1e9 / (double(m) * n * k)));
	double best = INFINITY;
	for (int run = 0; run < 3; ++run) {
	    lap();
	    for (long r = 0; r < reps; ++r) {
		ulmBLAS::gemm(m, n, k, 1.0, false, A.data(), 1L, m, false,
		              B.data(), 1L, k, 0.0, C.data(), 1L, m);
	    }
	    best = std::min(best, lap());
	}
	std::printf("%5ld %5ld %5ld  %-32s %8.1f\n", m, n, k, plan.name,
	            2e-9 * m * n * k * reps / best);
    }
}
