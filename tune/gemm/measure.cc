//
//  Measurement program of the GEMM tuning (make tune, see tune/README.md).
//
//      measure set                  the candidate set of the build
//      measure platform             the detected platform
//      measure list                 the shapes of the grid (index m n k)
//      measure run FIRST COUNT ROUNDS
//                                   measures the shapes FIRST to
//                                   FIRST+COUNT-1 of the grid
//
//  For each shape every candidate (candidates.h, generated from
//  tune/gemm/candidates.py) computes C <- C + A B with column major A, B
//  and C, ROUNDS times, interleaved in random order; each sample is the
//  time of enough calls for at least 10 ms, after one call that is not
//  timed (the candidates have their own buffers).  Output, one line per shape
//  and candidate:
//
//      m  n  k  candidate  GFLOP/s (median)  GFLOP/s (min)  GFLOP/s (max)
//
//  Before the measurement the results of all candidates are compared with
//  those of the first one (exit status 1 on a mismatch).
//
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <ulmblas/ulmblas.h>
#include <ulmblas/config/platform.h>
#include <ulmblas/level3/gemm/frame.h>

#include "candidates.h"

#if !defined(ULMBLAS_TUNE_SET)

int
main()
{
    std::fprintf(stderr, "measure: no candidates for this build "
                         "(ARCHFLAGS), see tune/gemm/candidates.py\n");
    return 1;
}

#else

namespace {

typedef std::ptrdiff_t Index;

//
//  Random elements in [-1, 1) (complex: real and imaginary part), the real
//  type and the flops of a multiply-add.
//
template <typename T> struct Random
{
    typedef T Real;
    static constexpr double flopsPerMultiplyAdd = 2;

    template <typename Dist, typename Gen>
    static T
    value(Dist &dist, Gen &gen)
    {
	return T(dist(gen));
    }
};

template <typename T> struct Random<std::complex<T>>
{
    typedef T Real;
    static constexpr double flopsPerMultiplyAdd = 8;

    template <typename Dist, typename Gen>
    static std::complex<T>
    value(Dist &dist, Gen &gen)
    {
	const T re = T(dist(gen));
	return std::complex<T>(re, T(dist(gen)));
    }
};

//
//  The grid: m, n, k from the grid values, all combinations, ordered by
//  the number of operations.
//
struct Shape
{
    Index m, n, k;
};

std::vector<Shape>
grid()
{
    //  1100 and 2300: just above the block sizes KC = 1024 and 2240, with
    //  a small rest (see the maximum block sizes, GemmBlocks).  The
    //  environment variable ULMBLAS_TUNE_GRID (comma separated) replaces
    //  the values (e.g. a smaller grid for slower machines).
    std::vector<Index> values = {32, 64, 128, 256, 512, 1100, 2300};
    if (const char *env = std::getenv("ULMBLAS_TUNE_GRID")) {
	values.clear();
	for (const char *p = env; *p;) {
	    char *end;
	    long v = std::strtol(p, &end, 10);
	    if (end == p) {
		break;
	    }
	    values.push_back(Index(v));
	    p = *end == ',' ? end + 1 : end;
	}
    }
    std::vector<Shape> shapes;
    for (Index m : values) {
	for (Index n : values) {
	    for (Index k : values) {
		shapes.push_back({m, n, k});
	    }
	}
    }
    std::stable_sort(shapes.begin(), shapes.end(),
                     [](const Shape &a, const Shape &b) {
	                 return a.m * a.n * a.k < b.m * b.n * b.k;
                     });
    return shapes;
}

double
seconds()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void
call(const TuneCandidate &candidate, const Shape &s, const TuneT *A,
     const TuneT *B, TuneT *C, const TuneT &beta)
{
    candidate.run(s.m, s.n, s.k, TuneT(1), false, A, Index(1), s.m, false, B,
                  Index(1), s.k, beta, C, Index(1), s.m);
}

int
measure(Index first, Index count, Index rounds)
{
    const std::vector<Shape> shapes = grid();
    const std::size_t numCandidates =
        sizeof(candidates) / sizeof(candidates[0]);

    std::mt19937 gen(std::uint32_t(first + 1));
    std::uniform_real_distribution<double> dist(-1, 1);

    for (Index q = first; q < first + count && q < Index(shapes.size()); ++q) {
	const Shape &s = shapes[q];
	std::vector<TuneT> A(s.m * s.k), B(s.k * s.n), C(s.m * s.n),
	    R(s.m * s.n), X(s.m * s.n);
	for (TuneT &x : A)
	    x = Random<TuneT>::value(dist, gen);
	for (TuneT &x : B)
	    x = Random<TuneT>::value(dist, gen);
	for (TuneT &x : C)
	    x = Random<TuneT>::value(dist, gen);

	//  the same result as the first candidate
	for (std::size_t c = 0; c < numCandidates; ++c) {
	    X = C;
	    call(candidates[c], s, A.data(), B.data(), X.data(), TuneT(0.5));
	    if (c == 0) {
		R = X;
		continue;
	    }
	    double err = 0, size = 0;
	    for (std::size_t i = 0; i < X.size(); ++i) {
		err = std::max(err, double(std::abs(X[i] - R[i])));
		size = std::max(size, double(std::abs(R[i])));
	    }
	    const double eps = double(
	        std::numeric_limits<typename Random<TuneT>::Real>::epsilon());
	    if (err > 16 * eps * double(s.k) * (1 + size)) {
		std::fprintf(stderr,
		             "measure: %s differs for m=%td n=%td "
		             "k=%td: %g\n",
		             candidates[c].name, s.m, s.n, s.k, err);
		return 1;
	    }
	}

	//  complex: a multiply-add is 4 real ones
	const double flops = Random<TuneT>::flopsPerMultiplyAdd * double(s.m) *
	                     double(s.n) * double(s.k);

	//  calls per sample: at least 10 ms (estimated with the first
	//  candidate)
	double t0 = seconds();
	call(candidates[0], s, A.data(), B.data(), C.data(), TuneT(1));
	double once = seconds() - t0;
	Index reps = std::max(Index(1), Index(10e-3 / std::max(once, 1e-7)));

	std::vector<std::vector<double>> samples(numCandidates);
	std::vector<std::size_t> order(numCandidates);
	for (std::size_t c = 0; c < numCandidates; ++c) {
	    order[c] = c;
	}
	for (Index r = 0; r < rounds; ++r) {
	    std::shuffle(order.begin(), order.end(), gen);
	    for (std::size_t c : order) {
		//  one call before the sample: the buffers and operands of
		//  this candidate in the caches, as after a previous call
		//  (not for calls of 50 ms or more: then it hardly matters)
		if (once < 50e-3) {
		    call(candidates[c], s, A.data(), B.data(), C.data(),
		         TuneT(1));
		}
		double t = seconds();
		for (Index p = 0; p < reps; ++p) {
		    call(candidates[c], s, A.data(), B.data(), C.data(),
		         TuneT(1));
		}
		t = (seconds() - t) / double(reps);
		samples[c].push_back(flops / t * 1e-9);
	    }
	}
	for (std::size_t c = 0; c < numCandidates; ++c) {
	    std::vector<double> &x = samples[c];
	    std::sort(x.begin(), x.end());
	    std::printf("%td\t%td\t%td\t%s\t%.2f\t%.2f\t%.2f\n", s.m, s.n, s.k,
	                candidates[c].name, x[x.size() / 2], x.front(),
	                x.back());
	}
	std::fflush(stdout);
    }
    return 0;
}

} // namespace

int
main(int argc, char **argv)
{
    if (argc == 2 && std::strcmp(argv[1], "set") == 0) {
	std::printf("%s\n", ULMBLAS_TUNE_SET);
	return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "platform") == 0) {
	std::printf("%s\n", ulmBLAS::platformName(ulmBLAS::platform()));
	return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "list") == 0) {
	const std::vector<Shape> shapes = grid();
	for (std::size_t q = 0; q < shapes.size(); ++q) {
	    std::printf("%zu\t%td\t%td\t%td\n", q, shapes[q].m, shapes[q].n,
	                shapes[q].k);
	}
	return 0;
    }
    if (argc == 5 && std::strcmp(argv[1], "run") == 0) {
	return measure(std::atol(argv[2]), std::atol(argv[3]),
	               std::atol(argv[4]));
    }
    std::fprintf(stderr, "usage: measure set | platform | list | run FIRST "
                         "COUNT ROUNDS\n");
    return 2;
}

#endif
