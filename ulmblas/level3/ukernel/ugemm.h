#ifndef ULMBLAS_LEVEL3_UKERNEL_UGEMM_H
#define ULMBLAS_LEVEL3_UKERNEL_UGEMM_H 1

#include <ulmblas/config/simd.h>

//
//  Selected optimized micro kernel
//
#if defined(USE_AVX) && !defined(USE_BLIS_DGEMM)
#   define  ULMBLAS_BASE_UGEMM_KERNEL   avx
#   include <ulmblas/level3/ukernel/avx/ugemm.h>
#elif defined(USE_AVX)
//  The AVX kernel set has an optimized kernel for double only (replaced
//  by the BLIS kernel) and uses the reference kernels for the other types.
#   define  ULMBLAS_BASE_UGEMM_KERNEL   ref
#   include <ulmblas/level3/ukernel/ref/ugemm.h>
#elif defined(USE_SME)
#   define  ULMBLAS_BASE_UGEMM_KERNEL   sme
#   include <ulmblas/level3/ukernel/sme/ugemm.h>
#elif defined(USE_NEON)
#   define  ULMBLAS_BASE_UGEMM_KERNEL   neon
#   include <ulmblas/level3/ukernel/neon/ugemm.h>
#elif defined(USE_SSE)
#   define  ULMBLAS_BASE_UGEMM_KERNEL   sse
#   include <ulmblas/level3/ukernel/sse/ugemm.h>
#else
#   define  ULMBLAS_BASE_UGEMM_KERNEL   ref
#   include <ulmblas/level3/ukernel/ref/ugemm.h>
#endif

//
//  A BLIS-derived micro kernel for double (USE_BLIS_DGEMM, see
//  ulmblas/level3/ukernel/blis/select.h) replaces the kernel for double,
//  vendored kernels for other types (USE_BLIS_XGEMM, select_z.h) the kernels
//  of their types; the remaining types use ULMBLAS_BASE_UGEMM_KERNEL.
//
#if defined(USE_BLIS_DGEMM) || defined(USE_BLIS_XGEMM)
#   define  SELECT_UGEMM_KERNEL     blisk
#   include <ulmblas/level3/ukernel/blis/ugemm.h>
#else
#   define  SELECT_UGEMM_KERNEL     ULMBLAS_BASE_UGEMM_KERNEL
#endif

namespace ulmBLAS {

template <typename T>
struct BlockSizeUGemm
{
    static constexpr int MR = SELECT_UGEMM_KERNEL::BlockSizeUGemm<T>::MR;
    static constexpr int NR = SELECT_UGEMM_KERNEL::BlockSizeUGemm<T>::NR;
};

//
//  Whether the micro kernel for T prefers row-major C (vendored BLIS
//  kernels whose assembly stores only row-major tiles, see
//  ulmblas/level3/ukernel/blis/select_z.h); GEMM then computes C^T = B^T A^T
//  for column-major C.
//
template <typename T>
struct UGemmRowPreference
{
#if defined(USE_BLIS_XGEMM)
    static constexpr bool value = blisk::Kernel<T>::rowpref;
#else
    static constexpr bool value = false;
#endif
};

//
//  Buffered variant.  Used for zero padded panels.
//
template <typename IndexType, typename T, typename Beta, typename TC>
    void
    ugemm(IndexType    mr,
          IndexType    nr,
          IndexType    kc,
          const T      &alpha,
          const T      *A,
          const T      *B,
          const Beta   &beta,
          TC           *C,
          IndexType    incRowC,
          IndexType    incColC,
          const T      *nextA,
          const T      *nextB);

//
//  Buffered variant.  Used if the result alpha*A*B needs to be upcasted for
//  computing C <- beta*C + (alpha*A*B)
//
template <typename IndexType, typename T, typename Beta, typename TC>
    void
    ugemm(IndexType   kc,
          const T     &alpha,
          const T     *A,
          const T     *B,
          const Beta  &beta,
          TC          *C,
          IndexType   incRowC,
          IndexType   incColC,
          const T     *nextA,
          const T     *nextB);

//
//  Unbuffered variant.
//
template <typename IndexType, typename T>
    void
    ugemm(IndexType   kc,
          const T     &alpha,
          const T     *A,
          const T     *B,
          const T     &beta,
          T           *C,
          IndexType   incRowC,
          IndexType   incColC,
          const T     *nextA,
          const T     *nextB);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_UGEMM_H

#include <ulmblas/level3/ukernel/ugemm.tcc>
