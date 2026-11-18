#ifndef ULMBLAS_LEVEL3_GEMM_DEFAULT_H
#define ULMBLAS_LEVEL3_GEMM_DEFAULT_H 1

#include <cstddef>
#include <type_traits>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/gemm/frame.h>
#include <ulmblas/level3/ukernel/backend.h>

//
//  With the OpenBLAS-derived kernel (ULMBLAS_BLIS_DGEMM_PANEL, see
//  ulmblas/level3/ukernel/blis/ugemm.h) DGEMM packs B interleaved with the
//  computation for the first block of A, as OpenBLAS's driver does: each
//  panel of NR columns of B is packed and immediately used while it is in
//  cache.  The packed layout is the same; -DULMBLAS_DGEMM_NO_INTERLEAVE
//  packs the whole block of B first, as for all other kernels.  Kernels
//  with ULMBLAS_BLIS_DGEMM_INTERLEAVE_PANELS (select.h) take that many
//  panels at a time (OpenBLAS's HASWELL kernel: 3, i.e. 24 columns, as
//  OpenBLAS's driver).
//
#if defined(ULMBLAS_BLIS_DGEMM_PANEL) && !defined(ULMBLAS_DGEMM_NO_INTERLEAVE)
#define ULMBLAS_DGEMM_INTERLEAVE 1
#else
#define ULMBLAS_DGEMM_INTERLEAVE 0
#endif
#if defined(ULMBLAS_BLIS_DGEMM_INTERLEAVE_PANELS)
#define ULMBLAS_DGEMM_INTERLEAVE_PANELS ULMBLAS_BLIS_DGEMM_INTERLEAVE_PANELS
#else
#define ULMBLAS_DGEMM_INTERLEAVE_PANELS 1
#endif

namespace ulmBLAS {

//
//  The frame algorithm that the build selects for gemm with packed blocks
//  of type T: the default backend (micro kernel of the build), the block
//  sizes of ulmblas/config/blocksize.h and interleaved packing of B for
//  DGEMM with the OpenBLAS-derived kernels.
//
template <typename T> struct DefaultGemm
{
    static constexpr std::ptrdiff_t Interleave =
        (ULMBLAS_DGEMM_INTERLEAVE && std::is_same<T, double>::value)
            ? ULMBLAS_DGEMM_INTERLEAVE_PANELS
            : 0;

    typedef GemmBlocks<BlockSize<T>::MC, BlockSize<T>::KC, BlockSize<T>::NC>
        Blocks;
    typedef GemmFrame<DefaultBackend<T>, Blocks, Interleave> Frame;
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_DEFAULT_H
