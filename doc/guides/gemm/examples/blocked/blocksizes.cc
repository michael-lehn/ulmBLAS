// The blocked gemm with the NEON kernel (8 x 6) for different block sizes
// MC and KC, n = 2048, one thread.  (AArch64 only.)
#include "../common/util.h"
#include "gemm.h"
#include "kernels.h"
#include <vector>

#if defined(__aarch64__)
typedef void (*Gemm)(long, long, long, double, const double *, long, long,
                     const double *, long, long, double, double *, long, long);

#define G(MC, KC) blocked::gemm<blocked::NeonKernel, MC, KC, 4104>
const long mcs[] = {48, 96, 240, 480, 960};
const long kcs[] = {64, 128, 256, 512, 1024};
Gemm gemms[5][5] = {
    {G(48, 64), G(48, 128), G(48, 256), G(48, 512), G(48, 1024)},
    {G(96, 64), G(96, 128), G(96, 256), G(96, 512), G(96, 1024)},
    {G(240, 64), G(240, 128), G(240, 256), G(240, 512), G(240, 1024)},
    {G(480, 64), G(480, 128), G(480, 256), G(480, 512), G(480, 1024)},
    {G(960, 64), G(960, 128), G(960, 256), G(960, 512), G(960, 1024)}};

int
main()
{
    const long n = 2048;
    std::vector<double> A(n * n), B(n * n), C(n * n);
    randomMatrix(n, n, A.data(), 1, n, 1);
    randomMatrix(n, n, B.data(), 1, n, 2);
    std::printf("n = %ld, GFLOPS\n%8s", n, "MC \\ KC");
    for (long kc : kcs) {
	std::printf(" %7ld", kc);
    }
    std::printf("\n");
    for (int i = 0; i < 5; ++i) {
	std::printf("%8ld", mcs[i]);
	for (int j = 0; j < 5; ++j) {
	    double best = 0;
	    for (int run = 0; run < 2; ++run) {
		lap();
		gemms[i][j](n, n, n, 1.0, A.data(), 1, n, B.data(), 1, n, 0.0,
		            C.data(), 1, n);
		best = std::max(best, 2e-9 * n * n * n / lap());
	    }
	    std::printf(" %7.1f", best);
	}
	std::printf("\n");
    }
}
#else
int
main()
{
    std::printf("AArch64 only\n");
}
#endif
