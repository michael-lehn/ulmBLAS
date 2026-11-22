// A function that takes "what to do" as a parameter: withUnitStride copies
// a strided matrix into a contiguous buffer, calls f on the copy and copies
// it back, as in ulmblas/lapack/unitstride.tcc.  The caller passes a lambda.
#include <cstdio>
#include <cstdlib>
#include <vector>

template <typename T, typename F>
long
withUnitStride(long m, long n, T *A, long incRowA, long incColA, F f)
{
    std::vector<T> B(m * n); // column major, leading dimension m
    for (long j = 0; j < n; ++j) {
	for (long i = 0; i < m; ++i) {
	    B[i + j * m] = A[i * incRowA + j * incColA];
	}
    }
    const long info = f(B.data(), 1L, m);
    for (long j = 0; j < n; ++j) {
	for (long i = 0; i < m; ++i) {
	    A[i * incRowA + j * incColA] = B[i + j * m];
	}
    }
    return info;
}

// a function evaluated once, at the first call, also if several threads
// call it at the same time (C++11: initialization of a local static)
long
blockSize()
{
    static const long nb = []() -> long {
	const char *s = std::getenv("BLOCK_SIZE");
	std::printf("  (blockSize: initialized once)\n");
	return s ? std::atol(s) : 64L;
    }();
    return nb;
}

int
main()
{
    // a 2 x 3 matrix stored row by row: incRowA = 3, incColA = 1
    double A[6] = {1, 2, 3, 4, 5, 6};
    const double alpha = 10;
    long calls = 0;

    // [alpha] captures a copy, [&calls] a reference
    const long info = withUnitStride(
        2L, 3L, A, 3L, 1L,
        [alpha, &calls](double *B, long incRow, long incCol) -> long {
	    ++calls;
	    for (long j = 0; j < 3; ++j) {
	        for (long i = 0; i < 2; ++i) {
		    B[i * incRow + j * incCol] *= alpha;
	        }
	    }
	    return 0L;
        });
    std::printf("info = %ld, calls = %ld, A = %g %g %g / %g %g %g\n", info,
                calls, A[0], A[1], A[2], A[3], A[4], A[5]);
    std::printf("block size %ld\n", blockSize());
    std::printf("block size %ld\n", blockSize());
}
