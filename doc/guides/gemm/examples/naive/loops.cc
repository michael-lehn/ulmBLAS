// C = C + A B with three nested loops, in all six orders, for column-major
// matrices.  The operations are the same; only the order of the memory
// accesses differs.
#include "../common/util.h"
#include <vector>

#define LOOP(name, a, b, c)                                                    \
    void name(long n, const double *A, const double *B, double *C)             \
    {                                                                          \
	for (long a = 0; a < n; ++a) {                                         \
	    for (long b = 0; b < n; ++b) {                                     \
		for (long c = 0; c < n; ++c) {                                 \
		    C[i + j * n] += A[i + l * n] * B[l + j * n];               \
		}                                                              \
	    }                                                                  \
	}                                                                      \
    }

LOOP(ijl, i, j, l)
LOOP(ilj, i, l, j)
LOOP(jil, j, i, l)
LOOP(jli, j, l, i)
LOOP(lij, l, i, j)
LOOP(lji, l, j, i)

int
main()
{
    void (*f[6])(long, const double *, const double *,
                 double *) = {ijl, ilj, jil, jli, lij, lji};
    const char *name[6] = {"ijl", "ilj", "jil", "jli", "lij", "lji"};

    std::printf("%6s", "n");
    for (int v = 0; v < 6; ++v) {
	std::printf(" %7s", name[v]);
    }
    std::printf("   (GFLOPS, innermost loop last)\n");
    for (long n : {100L, 200L, 400L, 800L, 1600L}) {
	std::vector<double> A(n * n), B(n * n), C(n * n);
	randomMatrix(n, n, A.data(), 1, n, 1);
	randomMatrix(n, n, B.data(), 1, n, 2);
	std::printf("%6ld", n);
	for (int v = 0; v < 6; ++v) {
	    std::fill(C.begin(), C.end(), 0.0);
	    lap();
	    f[v](n, A.data(), B.data(), C.data());
	    std::printf(" %7.2f", 2e-9 * n * n * n / lap());
	}
	std::printf("\n");
    }
}
