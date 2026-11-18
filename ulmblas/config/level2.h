#ifndef ULMBLAS_CONFIG_LEVEL2_H
#define ULMBLAS_CONFIG_LEVEL2_H 1

#include <complex>
#include <cstdint>

//
//  Parameters of the level 2 kernels (gemv, ger, trsv), all in one place;
//  each can be set with -D.  They are compile-time constants (the loops of
//  the kernels are built for them); the number of threads is a parameter
//  of the platform (ulmblas/config/tuned.h, measured by make tune).
//
//  gemv, all operands of one type, A column major (BLAS N) or row major
//  (BLAS T), see ulmblas/level2/kernel/gemvf.h:
//
//      FuseN    columns of A per pass over a block of y (A column major)
//      BlockN   rows of y per block (A column major), in the L1 cache
//      FuseT    rows of A per pass over a block of x (A row major)
//      LanesT   partial sums per row of A (A row major): a multiple of the
//               vector length, FuseT LanesT accumulators
//      BlockT   elements of x per block (A row major)
//
//  ger (float, double): FuseGer columns (A column major) or rows (row
//  major) of A per pass over x (y), see ulmblas/level2/kernel/gemvf.h.
//
//  trsv: TrsvBlock columns per diagonal block, the rest of the column
//  panel with one gemv (OpenBLAS: DTB_ENTRIES = 64).
//

#ifndef ULMBLAS_L2_DFUSE_N
#define ULMBLAS_L2_DFUSE_N 8
#endif
#ifndef ULMBLAS_L2_DFUSE_T
#define ULMBLAS_L2_DFUSE_T 4
#endif
#ifndef ULMBLAS_L2_DLANES_T
#define ULMBLAS_L2_DLANES_T 8
#endif
//  ger, real numbers: columns (A column major) or rows (row major) per pass
//  over x (y).  1: with more, columns at a distance of about a multiple of
//  4 KB (ldA = 500, 1000) are updated in the same pass, and loads of one
//  wait for stores to another (4K aliasing): dger n = 500 on heim 4.8
//  (4 columns), 4.1 (2) against 7.6 GFLOP/s (1; OpenBLAS 7.3)
#ifndef ULMBLAS_L2_FUSE_GER
#define ULMBLAS_L2_FUSE_GER 1
#endif
#ifndef ULMBLAS_L2_ZFUSE_N
#define ULMBLAS_L2_ZFUSE_N 4
#endif
#ifndef ULMBLAS_L2_ZFUSE_T
#define ULMBLAS_L2_ZFUSE_T 4
#endif
#ifndef ULMBLAS_L2_ZLANES_T
#define ULMBLAS_L2_ZLANES_T 4
#endif
//  bytes of a block of y (BLAS N) or x (BLAS T)
#ifndef ULMBLAS_L2_BLOCK_BYTES_N
#define ULMBLAS_L2_BLOCK_BYTES_N 16384
#endif
#ifndef ULMBLAS_L2_BLOCK_BYTES_T
#define ULMBLAS_L2_BLOCK_BYTES_T 16384
#endif
//  1: the calling thread is pinned in the parallel regions of level 2 (see
//  ThreadPool::run)
#ifndef ULMBLAS_L2_PIN_CALLER
#define ULMBLAS_L2_PIN_CALLER 0
#endif
#ifndef ULMBLAS_L2_TRSV_BLOCK
#define ULMBLAS_L2_TRSV_BLOCK 64
#endif

namespace ulmBLAS {

//  real types (float, double): the double values, float with twice the
//  lanes (same vector registers)
template <typename T> struct Level2Param
{
    static constexpr int FuseN = ULMBLAS_L2_DFUSE_N;
    static constexpr int FuseT = ULMBLAS_L2_DFUSE_T;
    static constexpr int LanesT = ULMBLAS_L2_DLANES_T * 8 / int(sizeof(T));
    static constexpr int FuseGer = ULMBLAS_L2_FUSE_GER;
    static constexpr std::int64_t BlockN = ULMBLAS_L2_BLOCK_BYTES_N / sizeof(T);
    static constexpr std::int64_t BlockT = ULMBLAS_L2_BLOCK_BYTES_T / sizeof(T);
    static constexpr std::int64_t TrsvBlock = ULMBLAS_L2_TRSV_BLOCK;
};

//  complex types: lanes are real numbers (LanesT real and imaginary parts
//  per row)
template <typename T> struct Level2Param<std::complex<T>>
{
    static constexpr int FuseN = ULMBLAS_L2_ZFUSE_N;
    static constexpr int FuseT = ULMBLAS_L2_ZFUSE_T;
    static constexpr int LanesT = ULMBLAS_L2_ZLANES_T * 8 / int(sizeof(T));
    static constexpr std::int64_t BlockN =
        ULMBLAS_L2_BLOCK_BYTES_N / (2 * sizeof(T));
    static constexpr std::int64_t BlockT =
        ULMBLAS_L2_BLOCK_BYTES_T / (2 * sizeof(T));
    static constexpr std::int64_t TrsvBlock = ULMBLAS_L2_TRSV_BLOCK;
};

} // namespace ulmBLAS

#endif // ULMBLAS_CONFIG_LEVEL2_H
