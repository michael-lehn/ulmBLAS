#ifndef ULMBLAS_LEVEL3_UKERNEL_NEON_UGEMM_H
#define ULMBLAS_LEVEL3_UKERNEL_NEON_UGEMM_H 1

#include <complex>
#include <type_traits>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/ukernel/ref/ugemm.h>

//
//  AArch64 NEON micro kernels: for double (6 x 8), derived from BLIS (see
//  ugemm.tcc for the license and the modifications), and for float (8 x
//  12, sgemm.tcc), complex float (4 x 6, cgemm.tcc) and complex double (4 x
//  3, zgemm.tcc), written for ulmBLAS.  Other types use the reference micro
//  kernel.
//
namespace ulmBLAS { namespace neon {

template <typename T>
struct BlockSizeUGemm
{
    static constexpr int MR = (std::is_same<T,double>::value) ? 6
                            : (std::is_same<T,float>::value) ? 8
                            : (std::is_same<T,std::complex<float> >::value) ? 4
                            : (std::is_same<T,std::complex<double> >::value) ? 4
                            : ref::BlockSizeUGemm<T>::MR;
    static constexpr int NR = (std::is_same<T,double>::value) ? 8
                            : (std::is_same<T,float>::value) ? 12
                            : (std::is_same<T,std::complex<float> >::value) ? 6
                            : (std::is_same<T,std::complex<double> >::value) ? 3
                            : ref::BlockSizeUGemm<T>::NR;

    static_assert(MR>0 && NR>0, "Invalid block size.");

    //  The block sizes MC, KC, NC (and the sizes of the packing buffers)
    //  are chosen for these MR, NR.
    static_assert(MR==BlockSize<T>::MR && NR==BlockSize<T>::NR,
                  "NEON micro kernels require MR x NR = 6 x 8 for double, "
                  "8 x 12 for float, 4 x 6 for complex float, 4 x 3 for "
                  "complex double");
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

} } // namespace neon, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_NEON_UGEMM_H

#include <ulmblas/level3/ukernel/neon/ugemm.tcc>
#include <ulmblas/level3/ukernel/neon/sgemm.tcc>
#include <ulmblas/level3/ukernel/neon/cgemm.tcc>
#include <ulmblas/level3/ukernel/neon/zgemm.tcc>
