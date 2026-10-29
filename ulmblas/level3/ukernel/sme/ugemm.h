#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_UGEMM_H
#define ULMBLAS_LEVEL3_UKERNEL_SME_UGEMM_H 1

#include <complex>
#include <type_traits>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/ukernel/ref/ugemm.h>

//
//  Experimental SME2 micro kernels: for double (32 x 16), derived from
//  BLIS (see ugemm.tcc for the license and the modifications), and for
//  float (32 x 32, sgemm.tcc), complex float (32 x 16, cgemm.tcc) and
//  complex double (16 x 16, zgemm.tcc), written for ulmBLAS.  Other types
//  use the reference micro kernel.  The panels are packed by the routines
//  of sme/pack.h.
//
namespace ulmBLAS { namespace sme {

template <typename T>
struct BlockSizeUGemm
{
    static constexpr int MR = (std::is_same<T,double>::value) ? 32
                            : (std::is_same<T,float>::value) ? 32
                            : (std::is_same<T,std::complex<float> >::value) ? 32
                            : (std::is_same<T,std::complex<double> >::value) ? 16
                            : ref::BlockSizeUGemm<T>::MR;
    static constexpr int NR = (std::is_same<T,double>::value) ? 16
                            : (std::is_same<T,float>::value) ? 32
                            : (std::is_same<T,std::complex<float> >::value) ? 16
                            : (std::is_same<T,std::complex<double> >::value) ? 16
                            : ref::BlockSizeUGemm<T>::NR;

    static_assert(MR>0 && NR>0, "Invalid block size.");

    //  The block sizes MC, KC, NC (and the sizes of the packing buffers)
    //  are chosen for these MR, NR.
    static_assert(MR==BlockSize<T>::MR && NR==BlockSize<T>::NR,
                  "SME micro kernels require MR x NR = 32 x 16 for double, "
                  "32 x 32 for float, 32 x 16 for complex float, 16 x 16 "
                  "for complex double");
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

} // namespace sme

//
//  Buffered variant of ulmblas/level3/ukernel/ugemm.h (used by the macro
//  kernels for partial tiles) for double with the SME2 kernel: for C with
//  unit row stride the kernel updates the mr x nr part of the tile directly
//  (predicated loads and stores).  This overload is more specialized than
//  the portable template and therefore chosen; it is only declared with
//  the SME2 kernel (USE_SME).
//
template <typename IndexType>
    void
    ugemm(IndexType      mr,
          IndexType      nr,
          IndexType      kc,
          const double   &alpha,
          const double   *A,
          const double   *B,
          const double   &beta,
          double         *C,
          IndexType      incRowC,
          IndexType      incColC,
          const double   *nextA,
          const double   *nextB);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_UGEMM_H

#include <ulmblas/level3/ukernel/sme/ugemm.tcc>
#include <ulmblas/level3/ukernel/sme/sgemm.tcc>
#include <ulmblas/level3/ukernel/sme/cgemm.tcc>
#include <ulmblas/level3/ukernel/sme/zgemm.tcc>
#include <ulmblas/level3/ukernel/sme/pack.h>
