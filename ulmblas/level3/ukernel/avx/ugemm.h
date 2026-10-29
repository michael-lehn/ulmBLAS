#ifndef ULMBLAS_LEVEL3_UKERNEL_AVX_UGEMM_H
#define ULMBLAS_LEVEL3_UKERNEL_AVX_UGEMM_H 1

#include <type_traits>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/ukernel/ref/ugemm.h>

//
//  AVX/FMA micro kernel for double (6 x 8), derived from BLIS: see
//  ugemm.tcc for the license and the modifications.  Other types use the
//  reference micro kernel.
//
namespace ulmBLAS { namespace avx {

template <typename T>
struct BlockSizeUGemm
{
    static constexpr int MR = (std::is_same<T,double>::value) ? 6
                            : ref::BlockSizeUGemm<T>::MR;
    static constexpr int NR = (std::is_same<T,double>::value) ? 8
                            : ref::BlockSizeUGemm<T>::NR;

    static_assert(MR>0 && NR>0, "Invalid block size.");

    //  The block sizes MC, KC, NC (and the sizes of the packing buffers)
    //  are chosen for these MR, NR.
    static_assert(MR==BlockSize<T>::MR && NR==BlockSize<T>::NR,
                  "AVX micro kernel requires MR = 6 and NR = 8 for double");
};

using ref::ugemm;

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
          const double   *nextB);

} } // namespace avx, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_AVX_UGEMM_H

#include <ulmblas/level3/ukernel/avx/ugemm.tcc>
