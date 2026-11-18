#ifndef ULMBLAS_CONFIG_TUNED_H
#define ULMBLAS_CONFIG_TUNED_H 1

#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ulmblas/config/platform.h>

//
//  The parameters of ulmBLAS that depend on the platform (besides the
//  selection tables of gemm, ulmblas/level3/gemm/select.h): thresholds for
//  the number of threads and for variants, block sizes.  tuned() returns
//  them for the platform that ulmBLAS runs on (ulmblas/config/platform.h):
//
//  - the values written by hand (tunedDefaults, below), the same on every
//    platform except where noted;
//  - then the values that make tune measured on the platform, if there are
//    any (ulmblas/config/tuned/PLATFORM.h, generated, tune/README.md).
//
//  The environment variables of the places that use a parameter (e.g.
//  ULMBLAS_TRSM_SHARED, see ulmblas/level3/gemm/trshared.h) still override
//  it, for experiments.  make tune changes the parameters of tuned() in the
//  measurement program to measure candidates.
//
//  Parameters per type are arrays indexed by tunedS, tunedD, tunedC,
//  tunedZ (float, double, complex float, complex double; TunedIndex<T>).
//  A decision by size is a TunedSteps: e.g. {1, 460800, 0} is 1 for
//  x < 460800 and 0 from there on.
//
namespace ulmBLAS {

constexpr std::ptrdiff_t tunedS = 0, tunedD = 1, tunedC = 2, tunedZ = 3;
constexpr std::ptrdiff_t tunedTypes = 4;

//  the index of type T: complex types by the size of their parts, other
//  types by their size (e.g. the packed blocks of bfloat16 gemm are float)
template <typename T> struct TunedIndex
{
    static constexpr std::ptrdiff_t value = sizeof(T) <= 4 ? tunedS : tunedD;
};

template <typename T> struct TunedIndex<std::complex<T>>
{
    static constexpr std::ptrdiff_t value = sizeof(T) <= 4 ? tunedC : tunedZ;
};

//
//  A decision by size with at most MaxSteps bounds: {v0, b0, v1, b1, ...,
//  vn} gives v0 for x < b0, v1 for b0 <= x < b1, ..., vn for x >= the last
//  bound (increasing bounds; more than MaxSteps are ignored).
//
struct TunedSteps
{
    static constexpr std::ptrdiff_t MaxSteps = 8;

    std::int64_t bound[MaxSteps];
    std::int64_t value[MaxSteps + 1];
    std::ptrdiff_t count;

    TunedSteps() : count(0)
    {
	value[0] = 0;
    }

    //  one value for all sizes
    TunedSteps(std::int64_t v) : count(0)
    {
	value[0] = v;
    }

    TunedSteps(std::initializer_list<std::int64_t> list) : count(0)
    {
	value[0] = 0;
	std::ptrdiff_t i = 0;
	for (std::int64_t x : list) {
	    if (i % 2 == 0) {
		value[i / 2] = x;
	    } else if (i / 2 < MaxSteps) {
		bound[i / 2] = x;
		count = i / 2 + 1;
	    } else {
		break;
	    }
	    ++i;
	}
	//  a list ending with a bound: the last value repeated
	if (count > 0 && i % 2 == 0) {
	    value[count] = value[count - 1];
	}
    }

    std::int64_t
    operator()(std::int64_t x) const
    {
	std::ptrdiff_t i = 0;
	while (i < count && x >= bound[i]) {
	    ++i;
	}
	return value[i];
    }
};

//
//  The parameters.  Where they are used:
//
//  gemm and the level 3 operations on its frame (ulmblas/level3/gemm/
//  threads.h):
//
//      gemmThreads    the threads by the work m n k (multiply-adds); 0:
//                     all threads of the pool, -1: by gemmMinWork
//      gemmMinWork    multiply-adds per thread at least (complex: real
//                     multiply-adds / 4)
//      gemmRatioM,    the partition of the threads as BLIS (BLAS-like
//      gemmRatioN,    Library Instantiation Software; BLAS: Basic Linear
//      gemmMaxJR      Algebra Subprograms): ratio of the work of rows and
//                     columns, at most gemmMaxJR threads in the loop over
//                     the panels of B
//
//  trsm and trmm with A on the left (ulmblas/level3/gemm/frametrsm.h,
//  frametrmm.h, trshared.h; A on the right is mapped to the left):
//
//      kb             rows of a diagonal block by m (rounded up to whole
//                     panels of MR rows, at most KCmax)
//      sharedBlocks,  on more than one thread the threads share the packed
//      sharedPanels   blocks of A if m > sharedBlocks kb and there are fewer
//                     than sharedPanels panels (NR columns) of B per thread,
//                     otherwise each thread solves its columns of B
//      minPanels      ... with at least minPanels panels per thread
//      trsmInvert     the diagonal tiles inverted (1) or solved by
//                     substitution (0); -1: inverted for MR >= 16 on Ice
//                     Lake
//
//  level 2 (ulmblas/level2/threads.h): the number of threads by m n, 0 for
//  all threads of the pool (at most the pool):
//
//      gemvNThreads   gemv with A column major (BLAS N)
//      gemvTThreads   gemv with A row major (BLAS T, C)
//      gerThreads     ger
//
//  LAPACK (Linear Algebra PACKage) with OpenMP (Open Multi-Processing;
//  OPENMP=1, ulmblas/lapack/tasks/param.h): potrf (and trtri, lauum, getri)
//  as tasks for n > potrfSerialN (inverses: inverseSerialN) on at most
//  n / potrfRowsPerThread threads (at least 4), tiles of potrfNb(n) rows.
//
struct Tuned
{
    TunedSteps gemmThreads[tunedTypes];
    std::int64_t gemmMinWork[tunedTypes];
    std::int64_t gemmRatioM, gemmRatioN;
    std::ptrdiff_t gemmMaxJR;

    struct Tr
    {
	TunedSteps kb[tunedTypes];
	std::ptrdiff_t sharedBlocks[tunedTypes];
	std::ptrdiff_t sharedPanels[tunedTypes];
	std::ptrdiff_t minPanels[tunedTypes];
    };

    Tr trsm, trmm;
    std::ptrdiff_t trsmInvert[tunedTypes];

    TunedSteps gemvNThreads[tunedTypes];
    TunedSteps gemvTThreads[tunedTypes];
    TunedSteps gerThreads[tunedTypes];

    TunedSteps potrfNb;
    std::ptrdiff_t potrfSerialN, inverseSerialN, potrfRowsPerThread;
};

//
//  The values written by hand (before make tune measured them):
//
//  - gemm threads by gemmMinWork only, as OpenBLAS 0.3.34 (interface/
//    gemm.c): 65536 * 4 multiply-adds per thread, 8192 * 4 for complex
//    types; the partition as BLIS 2.0 with the constants of its
//    configuration for the platform:
//    skx (Ice Lake) 3:2 and 4, zen3 (Zen 4) 1:1 and 1, otherwise 1:1 and 4.
//  - trsm, trmm: diagonal blocks of 256 rows (as OpenBLAS's GEMM_Q);
//    shared packing above 4 diagonal blocks (Zen 4: 12; complex types 4
//    times that) with fewer than 16 panels per thread, otherwise at least
//    one panel per thread (measurements in ulmblas/level3/gemm/trshared.h);
//    inverted tiles for MR >= 16 on Ice Lake (frametrsm.h).
//  - level 2 threads as OpenBLAS 0.3.34 (interface/gemv.c, ger.c, zgemv.c,
//    zger.c with GEMM_MULTITHREAD_THRESHOLD 4): s/dgemv all threads from
//    m n = 115200 * 4, c/zgemv from 1024 * 4; s/dger 3 threads above
//    2048 * 4, all above 20480 * 4; c/zger all above 36 s^2 * 4 (s the
//    size of a real part in bytes).
//  - potrf tasks (tuned for double on a Xeon Platinum 8358, see
//    experiments/lapack/README.md): tiles of 32 (n < 250), 48 (< 600), 64
//    (< 900), 96 (< 2200) or 128 rows; tasks for n > 100 (inverses 128),
//    one thread per 64 rows.
//
inline Tuned
tunedDefaults(Platform platform)
{
    Tuned p;
    for (std::ptrdiff_t t = 0; t < tunedTypes; ++t) {
	const bool complexType = t == tunedC || t == tunedZ;
	p.gemmThreads[t] = {-1};
	p.gemmMinWork[t] = complexType ? 8192 * 4 : 65536 * 4;

	const std::ptrdiff_t blocks =
	    (platform == Platform::AmdZen4 ? 12 : 4) * (complexType ? 4 : 1);
	Tuned::Tr *ops[] = {&p.trsm, &p.trmm};
	for (Tuned::Tr *op : ops) {
	    op->kb[t] = {256};
	    op->sharedBlocks[t] = blocks;
	    op->sharedPanels[t] = 16;
	    op->minPanels[t] = 1;
	}
	p.trsmInvert[t] = -1;
    }
    p.gemmRatioM = platform == Platform::IntelIceLake ? 3 : 1;
    p.gemmRatioN = platform == Platform::IntelIceLake ? 2 : 1;
    p.gemmMaxJR = platform == Platform::AmdZen4 ? 1 : 4;

    for (std::ptrdiff_t t = 0; t < tunedTypes; ++t) {
	const bool complexType = t == tunedC || t == tunedZ;
	p.gemvNThreads[t] = {1, complexType ? 1024 * 4 : 115200 * 4, 0};
	p.gemvTThreads[t] = p.gemvNThreads[t];
    }
    p.gerThreads[tunedS] =
        p.gerThreads[tunedD] = {1, 2048 * 4 + 1, 3, 20480 * 4 + 1, 0};
    p.gerThreads[tunedC] = {1, 36 * 4 * 4 * 4 + 1, 0};
    p.gerThreads[tunedZ] = {1, 36 * 8 * 8 * 4 + 1, 0};

    p.potrfNb = {32, 250, 48, 600, 64, 900, 96, 2200, 128};
    p.potrfSerialN = 100;
    p.inverseSerialN = 128;
    p.potrfRowsPerThread = 64;
    return p;
}

} // namespace ulmBLAS

//  the values measured by make tune (tunedForPlatform)
#include <ulmblas/config/tuned/tuned.h>

namespace ulmBLAS {

//
//  The parameters for the platform (determined at the first call,
//  thread-safe).  Only the measurement program of make tune changes them.
//
inline Tuned &
tuned()
{
    static Tuned value = [] {
	Tuned p = tunedDefaults(platform());
#if !defined(USE_TESTPARAM)
	tunedForPlatform(platform(), p);
#endif
	return p;
    }();
    return value;
}

} // namespace ulmBLAS

#endif // ULMBLAS_CONFIG_TUNED_H
