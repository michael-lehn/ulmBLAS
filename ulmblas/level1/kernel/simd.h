#ifndef ULMBLAS_LEVEL1_KERNEL_SIMD_H
#define ULMBLAS_LEVEL1_KERNEL_SIMD_H 1

//
//  OpenMP SIMD directives for the level 1 loops.  The build compiles with
//  -fopenmp-simd (make.inc, SIMDFLAGS): the SIMD directives only, no
//  OpenMP runtime.  Without it the directives are ignored.
//
#define ULMBLAS_PRAGMA(x) _Pragma(#x)
#define ULMBLAS_SIMD ULMBLAS_PRAGMA(omp simd)

//
//  clang (21, Apple M5): the simple loops of before (one omp simd reduction,
//  one loop for scal, swap, rot) are vectorized and interleaved well; the
//  shapes below that GCC 12 needs (32 partial sums of float in a block, scal
//  in blocks, rot and swap in blocks) gave slow code: sasum n = 1000 3.1
//  instead of 24, sscal 11 instead of 32, sswap n = 100000 2.8 instead of
//  9.7 elements per ns.
//
//  For the sums over double the partial sums of GCC are faster with clang
//  too (dasum n = 1000 16.8 against 13.4, dnrm2 2.5 against 1.3), for
//  float elements they are not (SimpleSum).
//
#if defined(__clang__)
#define ULMBLAS_LEVEL1_SIMPLE_LOOPS 1
#endif

struct SimpleLoops
{
    enum
    {
#if defined(ULMBLAS_LEVEL1_SIMPLE_LOOPS)
	value = 1
#else
	value = 0
#endif
    };
};

template <typename T> struct SimpleSum
{
    enum
    {
#if defined(ULMBLAS_LEVEL1_SIMPLE_LOOPS)
	value = sizeof(T) < 8
#else
	value = 0
#endif
    };
};

//  Loop bodies that must be inlined into the loop (clang 21 called the block
//  of partial sums as a function: dnrm2 n = 1000 on the M5 2.5 instead of
//  6.8 elements per ns; with GCC 12 always_inline made sdot n = 1000 on heim
//  slower, 20.7 against 25.4)
#if defined(__clang__)
#define ULMBLAS_LEVEL1_INLINE inline __attribute__((always_inline))
#else
#define ULMBLAS_LEVEL1_INLINE inline
#endif

#include <cstdint>

namespace ulmBLAS {

//
//  Loops that store a contiguous vector y: the first elements up to a
//  64-byte boundary of y are done separately (for n >= 256), so that the
//  vectorized loop stores whole cache lines (stores across cache lines
//  halve the speed: dcopy n = 1000 on heim 9.5 against 16.4 elements per
//  ns; for n = 100 the extra loop costs more than it gains).
//
template <typename IndexType, typename T>
IndexType
storePeel(IndexType n, const T *y)
{
#if defined(ULMBLAS_LEVEL1_SIMPLE_LOOPS)
    //  clang: no gain on the M5 (scopy n = 100000 21.6 without, 16.1 with)
    return 0;
#else
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(y);
    if (n < IndexType(256) || a % sizeof(T) != 0) {
	return 0;
    }
    return IndexType((64 - a % 64) % 64 / sizeof(T));
#endif
}

//
//  Two vectors updated in place (rot, rotm, swap), contiguous and long:
//  blocks of 32 elements, all loads of a block before its stores.  Large
//  vectors from malloc usually have the same offset in a page, and loads
//  of x after stores to y at the same offset modulo 4 KB wait for the
//  stores (4K aliasing): drot n = 100000 on heim 0.84 elements per ns with
//  one vectorized loop, 1.32 in blocks, OpenBLAS 1.26.  For short vectors
//  one loop is faster (n = 1000: 5.9 against 3.8).
//
struct InPlaceBlocks
{
    enum
    {
	size = 32,
#if defined(ULMBLAS_LEVEL1_SIMPLE_LOOPS)
	minN = 0x7fffffff
#else
	minN = 4096
#endif
    };
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_KERNEL_SIMD_H
