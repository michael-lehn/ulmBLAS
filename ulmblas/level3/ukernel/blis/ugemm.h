#ifndef ULMBLAS_LEVEL3_UKERNEL_BLIS_UGEMM_H
#define ULMBLAS_LEVEL3_UKERNEL_BLIS_UGEMM_H 1

#include <cstdint>
#include <type_traits>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/ukernel/blis/select.h>
#include <ulmblas/level3/ukernel/blis/xugemm.h>
#if defined(USE_BLIS_DGEMM)
#   include ULMBLAS_BLIS_DGEMM_HEADER
#endif

//
//  ulmBLAS micro kernel interface for a BLIS-derived DGEMM micro kernel
//  (selected with -DULMBLAS_DGEMM_UKERNEL_blis_NAME, see select.h and
//  README.md).  For double the vendored kernel kernel/NAME.h is called.
//  Types with a vendored kernel of their own (ZGEMM: select_z.h) use the
//  adapter in xugemm.h; all other types use the micro kernels of
//  ULMBLAS_BASE_UGEMM_KERNEL, the selection without a vendored kernel (see
//  ugemm.h one directory up).  Each of these selections works without the
//  others.
//
//  The adapter:
//
//  - passes alpha, beta and the panels unchanged and nextA, nextB as the
//    addresses of the next panels in BLIS's auxinfo_t (the same meaning);
//  - calls kernels with the interface of BLIS 1.0 with m = MR, n = NR
//    (ulmBLAS passes full tiles only, partial tiles are computed by the
//    buffered variant in ulmblas/level3/ukernel/ugemm.tcc); such kernels
//    handle the storage of C themselves (GEMM_UKR_SETUP_CT*, compat.h);
//  - for kernels with the interface of BLIS 0.x, computes alpha*A*B into a
//    local tile (column major, or row major for kernels that support row
//    major C only) if the storage of C is not supported by the assembly
//    code;
//  - for OpenBLAS kernels (C <- C + alpha*A*B, column-major C) scales the
//    tile of C by beta first (sets it to zero for beta = 0) and uses a local
//    tile for other storage of C; for kernels with a limit KMAX on k it
//    splits longer k loops into calls with at most KMAX;
//  - for kernels that store C with aligned instructions (storage class 4)
//    uses a 64-byte aligned local tile unless C is column major with
//    32-byte aligned columns;
//  - computes alpha*A*B into a local tile and updates C in C++ if beta is
//    NaN: the kernels test beta == 0 with vcomisd and je, which is also
//    taken for NaN, so they would not read C.  With a local tile
//    C <- beta*C + alpha*A*B propagates NaN as the other ulmBLAS kernels.
//
namespace ulmBLAS { namespace blisk {

template <typename T, bool = Kernel<T>::selected>
struct BlockSizeUGemm_ : ULMBLAS_BASE_UGEMM_KERNEL::BlockSizeUGemm<T>
{
};

template <typename T>
struct BlockSizeUGemm_<T, true> : BlockSizeKernel<T>
{
};

template <typename T>
struct BlockSizeUGemm : BlockSizeUGemm_<T>
{
};

#if defined(USE_BLIS_DGEMM)
template <>
struct BlockSizeUGemm<double>
{
    static constexpr int MR = ULMBLAS_BLIS_DGEMM_MR;
    static constexpr int NR = ULMBLAS_BLIS_DGEMM_NR;

    //  The block sizes MC, KC, NC (and the sizes of the packing buffers)
    //  are chosen for these MR, NR.
    static_assert(MR==BlockSize<double>::MR && NR==BlockSize<double>::NR,
                  "BLIS DGEMM micro kernel: MR and NR do not match BlockSize");
};

#endif

#if defined(USE_BLIS_DGEMM)
#   define ULMBLAS_BLIS_DGEMM_SELECTED_ true
#else
#   define ULMBLAS_BLIS_DGEMM_SELECTED_ false
#endif

//  types without a vendored kernel
template <typename IndexType, typename T>
static typename std::enable_if<!Kernel<T>::selected
                            && !(std::is_same<T,double>::value
                                 && ULMBLAS_BLIS_DGEMM_SELECTED_),
void>::type
ugemm(IndexType   kc,
      const T     &alpha,
      const T     *A,
      const T     *B,
      const T     &beta,
      T           *C,
      IndexType   incRowC,
      IndexType   incColC,
      const T     *nextA,
      const T     *nextB)
{
    ULMBLAS_BASE_UGEMM_KERNEL::ugemm(kc, alpha, A, B, beta, C,
                                     incRowC, incColC, nextA, nextB);
}

#undef ULMBLAS_BLIS_DGEMM_SELECTED_

//  vendored kernels for types other than double (xugemm.h)
template <typename IndexType, typename T>
static typename std::enable_if<Kernel<T>::selected, void>::type
ugemm(IndexType   kc,
      const T     &alpha,
      const T     *A,
      const T     *B,
      const T     &beta,
      T           *C,
      IndexType   incRowC,
      IndexType   incColC,
      const T     *nextA,
      const T     *nextB)
{
    xugemm(kc, alpha, A, B, beta, C, incRowC, incColC, nextA, nextB);
}

#if defined(USE_BLIS_DGEMM)

template <typename IndexType>
static typename std::enable_if<std::is_convertible<IndexType,long>::value,
void>::type
ugemm(IndexType      kc,
      const double   &alpha,
      const double   *A,
      const double   *B,
      const double   &beta,
      double         *C,
      IndexType      incRowC,
      IndexType      incColC,
      const double   *nextA,
      const double   *nextB)
{
    constexpr IndexType MR = ULMBLAS_BLIS_DGEMM_MR;
    constexpr IndexType NR = ULMBLAS_BLIS_DGEMM_NR;

#if ULMBLAS_BLIS_DGEMM_API==0
#   if ULMBLAS_BLIS_DGEMM_STORAGE==0
    const bool storageOk = incRowC==IndexType(1);
#   elif ULMBLAS_BLIS_DGEMM_STORAGE==1
    const bool storageOk = incRowC==IndexType(1) || incColC==IndexType(1);
#   elif ULMBLAS_BLIS_DGEMM_STORAGE==3
    const bool storageOk = incColC==IndexType(1);
#   else
    const bool storageOk = true;
#   endif
#elif ULMBLAS_BLIS_DGEMM_API==2
    const bool storageOk = incRowC==IndexType(1);
#else
    const bool storageOk = true;
#endif
#if ULMBLAS_BLIS_DGEMM_STORAGE==4
    //  column major with 32-byte aligned columns (aligned stores to C)
    const bool alignedOk = incRowC==IndexType(1)
                        && reinterpret_cast<std::uintptr_t>(C)%32==0
                        && (incColC*sizeof(double))%32==0;
#else
    const bool alignedOk = true;
#endif

    if (beta!=beta || !storageOk || !alignedOk) {
        //  local tile in a storage the kernel supports
#if ULMBLAS_BLIS_DGEMM_API==0 && ULMBLAS_BLIS_DGEMM_STORAGE==3
        constexpr IndexType rs = NR, cs = 1;
#else
        constexpr IndexType rs = 1, cs = MR;
#endif
        alignas(64) double C_[MR*NR];

        ugemm(kc, alpha, A, B, 0.0, C_, rs, cs, nextA, nextB);
        if (beta==0.0) {
            for (IndexType j=0; j<NR; ++j) {
                for (IndexType i=0; i<MR; ++i) {
                    C[i*incRowC+j*incColC] = C_[i*rs+j*cs];
                }
            }
        } else {
            for (IndexType j=0; j<NR; ++j) {
                for (IndexType i=0; i<MR; ++i) {
                    C[i*incRowC+j*incColC] = beta*C[i*incRowC+j*incColC]
                                           + C_[i*rs+j*cs];
                }
            }
        }
        return;
    }

    namespace K = ULMBLAS_BLIS_DGEMM_NAMESPACE;

#if ULMBLAS_BLIS_DGEMM_API==2
    //  OpenBLAS: C <- C + alpha*A*B for column-major C; beta first (C is not
    //  read for beta = 0)
    if (beta==0.0) {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                C[i+j*incColC] = 0.0;
            }
        }
    } else if (beta!=1.0) {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                C[i+j*incColC] *= beta;
            }
        }
    }
    (void)nextA; (void)nextB;
#   if defined(ULMBLAS_BLIS_DGEMM_KMAX)
    //  the packed panels are k-major: the part l, l+1, ... of the k loop
    //  starts at A + l*MR and B + l*NR
    constexpr IndexType KMAX = ULMBLAS_BLIS_DGEMM_KMAX;
    for (IndexType l=0; l<kc; l+=KMAX) {
        const IndexType k = (kc-l<KMAX) ? kc-l : KMAX;
        K::ULMBLAS_BLIS_DGEMM_FUNCTION(long(MR), long(NR), long(k), alpha,
                                       const_cast<double *>(A+l*MR),
                                       const_cast<double *>(B+l*NR),
                                       C, long(incColC));
    }
#   else
    K::ULMBLAS_BLIS_DGEMM_FUNCTION(long(MR), long(NR), long(kc), alpha,
                                   const_cast<double *>(A),
                                   const_cast<double *>(B),
                                   C, long(incColC));
#   endif
#else
    using namespace blis::compat;
    auxinfo_t data = { nextA, nextB };

#if ULMBLAS_BLIS_DGEMM_API==1
    K::ULMBLAS_BLIS_DGEMM_FUNCTION(dim_t(MR), dim_t(NR), dim_t(kc),
                                   &alpha, A, B, &beta,
                                   C, inc_t(incRowC), inc_t(incColC),
                                   &data, nullptr);
#else
    K::ULMBLAS_BLIS_DGEMM_FUNCTION(dim_t(kc),
                                   const_cast<double *>(&alpha),
                                   const_cast<double *>(A),
                                   const_cast<double *>(B),
                                   const_cast<double *>(&beta),
                                   C, inc_t(incRowC), inc_t(incColC),
                                   &data, nullptr);
#endif
#endif
}

#if ULMBLAS_BLIS_DGEMM_API==2 && !defined(ULMBLAS_DGEMM_NO_PANEL)
//
//  Panel variant for OpenBLAS kernels (used by the macro kernel mgemm,
//  -DULMBLAS_DGEMM_NO_PANEL switches it off): C <- beta*C + alpha*A*B for
//  mp full MR x NR tiles of one panel of B, i.e. A holds mp consecutive
//  packed panels of MR rows and C is column major (incColC = ldc).  OpenBLAS
//  kernels compute a whole packed block per call, as OpenBLAS's driver calls
//  them; one call per tile costs 2-3% on an Intel Xeon Platinum 8358
//  (benchmark_results/016-2026-10-27-uc3-icelake-openblas-kernel-prof).  The
//  kernel is called with m = mp*MR, n = NR only (the path its B addressing
//  is adapted for).
//
#   define ULMBLAS_BLIS_DGEMM_PANEL

template <typename IndexType>
static void
ugemm_panel(IndexType      mp,
            IndexType      kc,
            const double   &alpha,
            const double   *A,
            const double   *B,
            const double   &beta,
            double         *C,
            IndexType      incColC)
{
    constexpr IndexType MR = ULMBLAS_BLIS_DGEMM_MR;
    constexpr IndexType NR = ULMBLAS_BLIS_DGEMM_NR;
    namespace K = ULMBLAS_BLIS_DGEMM_NAMESPACE;

#   if defined(ULMBLAS_BLIS_DGEMM_KMAX)
    //  the k loop is too long for one call: tile by tile (the per-tile
    //  adapter splits it)
    if (kc>IndexType(ULMBLAS_BLIS_DGEMM_KMAX)) {
        for (IndexType i=0; i<mp; ++i) {
            ugemm(kc, alpha, A+i*kc*MR, B, beta, C+i*MR,
                  IndexType(1), incColC, A, B);
        }
        return;
    }
#   endif

    const IndexType m = mp*MR;
    if (beta==0.0) {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<m; ++i) {
                C[i+j*incColC] = 0.0;
            }
        }
    } else if (beta!=1.0) {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<m; ++i) {
                C[i+j*incColC] *= beta;
            }
        }
    }
    K::ULMBLAS_BLIS_DGEMM_FUNCTION(long(m), long(NR), long(kc), alpha,
                                   const_cast<double *>(A),
                                   const_cast<double *>(B),
                                   C, long(incColC));
}

#   if defined(ULMBLAS_BLIS_DGEMM_BLOCKCALL)
//
//  Block variant for OpenBLAS kernels that address B correctly for any
//  number of full panels (ULMBLAS_BLIS_DGEMM_BLOCKCALL): C <- beta*C +
//  alpha*A*B for the mp x np full MR x NR tiles of a packed block (A: mp
//  panels of MR rows, B: np panels of NR columns, column-major C), with one
//  call of the kernel, as OpenBLAS's driver calls it.  Returns false (and
//  computes nothing) if kc exceeds the limit of the kernel.
//
#       define ULMBLAS_BLIS_DGEMM_BLOCK

template <typename IndexType>
static bool
ugemm_block(IndexType      mp,
            IndexType      np,
            IndexType      kc,
            const double   &alpha,
            const double   *A,
            const double   *B,
            const double   &beta,
            double         *C,
            IndexType      incColC)
{
    constexpr IndexType MR = ULMBLAS_BLIS_DGEMM_MR;
    constexpr IndexType NR = ULMBLAS_BLIS_DGEMM_NR;
    namespace K = ULMBLAS_BLIS_DGEMM_NAMESPACE;

#       if defined(ULMBLAS_BLIS_DGEMM_KMAX)
    if (kc>IndexType(ULMBLAS_BLIS_DGEMM_KMAX)) {
        return false;
    }
#       endif

    const IndexType m = mp*MR;
    const IndexType n = np*NR;
    if (beta==0.0) {
        for (IndexType j=0; j<n; ++j) {
            for (IndexType i=0; i<m; ++i) {
                C[i+j*incColC] = 0.0;
            }
        }
    } else if (beta!=1.0) {
        for (IndexType j=0; j<n; ++j) {
            for (IndexType i=0; i<m; ++i) {
                C[i+j*incColC] *= beta;
            }
        }
    }
    K::ULMBLAS_BLIS_DGEMM_FUNCTION(long(m), long(n), long(kc), alpha,
                                   const_cast<double *>(A),
                                   const_cast<double *>(B),
                                   C, long(incColC));
    return true;
}
#   endif
#endif

#endif // USE_BLIS_DGEMM

} } // namespace blisk, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_BLIS_UGEMM_H
