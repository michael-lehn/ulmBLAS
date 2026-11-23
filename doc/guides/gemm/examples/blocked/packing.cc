// What packing costs: the blocked gemm with the SIMD kernel, and the same
// frame with a micro kernel that computes nothing.  The second takes the
// time of packing A and B, of the loops and of the update of C ("rest").
// For square matrices, and for a small k, where every element of A and B
// is used only a few times.  One thread.
#include "../common/util.h"
#include "gemm.h"
#include "kernels.h"
#include <vector>

#if defined(HAVE_SIMD_KERNEL)
// a micro kernel with the shape of SimdKernel that only writes zeros
struct NullKernel
{
    static const long MR = blocked::SimdKernel::MR;
    static const long NR = blocked::SimdKernel::NR;

    static void
    run(long, double, const double *, const double *, double *AB)
    {
	for (long i = 0; i < MR * NR; ++i) {
	    AB[i] = 0;
	}
    }
};

template <typename UK>
double
seconds(long m, long n, long k, const double *A, const double *B, double *C)
{
    double best = 1e30;
    for (int run = 0; run < 3; ++run) {
	lap();
	blocked::gemm<UK, 96, 256, 4104>(m, n, k, 1.0, A, 1, m, B, 1, k, 0.0, C,
	                                 1, m);
	best = std::min(best, lap());
    }
    return best;
}

int
main()
{
    std::printf("%6s %6s %6s %10s %10s %8s\n", "m", "n", "k", "gemm (s)",
                "rest", "share");
    const long sizes[][3] = {{512, 512, 512},
                             {1024, 1024, 1024},
                             {2048, 2048, 2048},
                             {2048, 2048, 64},
                             {4096, 4096, 32}};
    for (const auto &s : sizes) {
	const long m = s[0], n = s[1], k = s[2];
	std::vector<double> A(m * k), B(k * n), C(m * n);
	randomMatrix(m, k, A.data(), 1, m, 1);
	randomMatrix(k, n, B.data(), 1, k, 2);
	const double t =
	    seconds<blocked::SimdKernel>(m, n, k, A.data(), B.data(), C.data());
	const double p =
	    seconds<NullKernel>(m, n, k, A.data(), B.data(), C.data());
	std::printf("%6ld %6ld %6ld %10.4f %10.4f %7.1f%%\n", m, n, k, t, p,
	            100 * p / t);
    }
}
#else
int
main()
{
    std::printf("no SIMD kernel for this processor\n");
}
#endif
