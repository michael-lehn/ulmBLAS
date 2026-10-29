#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_PACK_H
#define ULMBLAS_LEVEL3_UKERNEL_SME_PACK_H 1

//
//  Packing of a single panel for the SME2 micro kernels (double, MR = 32,
//  NR = 16; float, MR = NR = 32; complex float, MR = 32, NR = 16; complex
//  double, MR = NR = 16): the packing layout of the portable routines
//  pack_MRxk and pack_kxNR of ulmblas/level3/pack/gepack.tcc, packed in
//  streaming mode with ZA as transpose buffer where needed (see pack.tcc,
//  spack.tcc, cpack.tcc and zpack.tcc).
//
//  gepack_A and gepack_B organize the packing of a block into panels and
//  call pack_MRxk, pack_kxNR, pack_MRxk_partial and pack_kxNR_partial
//  unqualified; these overloads for float, double, complex float and
//  complex double are more specialized than the portable templates and
//  therefore chosen.  They are only declared with
//  the SME2 kernel (USE_SME), so the packing of the other configurations
//  is unchanged.
//
#include <complex>

namespace ulmBLAS {

//  The portable routines (defined in ulmblas/level3/pack/gepack.tcc).
template <int P, typename IndexType, typename TA, typename Buffer>
    static void
    pack_MRxk(IndexType   k,
              bool        conj,
              const TA    *A,
              IndexType   incRowA,
              IndexType   incColA,
              Buffer      *buffer);

template <typename IndexType, typename TB, typename Buffer>
    static void
    pack_kxNR(IndexType   k,
              bool        conj,
              const TB    *B,
              IndexType   incRowB,
              IndexType   incColB,
              Buffer      *buffer);

//  The routines for double with the SME2 kernel.
template <int P, typename IndexType>
    static void
    pack_MRxk(IndexType      k,
              bool           conj,
              const double   *A,
              IndexType      incRowA,
              IndexType      incColA,
              double         *buffer);

template <typename IndexType>
    static void
    pack_kxNR(IndexType      k,
              bool           conj,
              const double   *B,
              IndexType      incRowB,
              IndexType      incColB,
              double         *buffer);

//  The last (partial) panels, see gepack_A and gepack_B: packed in
//  streaming mode as well, since data written in ordinary mode and read
//  by the kernel in streaming mode (or vice versa) cost about 0.7 us per
//  2 KiB on an Apple M5 (much more than packing small matrices).
template <typename IndexType>
    static bool
    pack_MRxk_partial(IndexType      mr,
                      IndexType      k,
                      bool           conj,
                      const double   *A,
                      IndexType      incRowA,
                      IndexType      incColA,
                      double         *buffer);

template <typename IndexType>
    static bool
    pack_kxNR_partial(IndexType      nr,
                      IndexType      k,
                      bool           conj,
                      const double   *B,
                      IndexType      incRowB,
                      IndexType      incColB,
                      double         *buffer);

//  The routines for float with the SME2 kernel (spack.tcc).
template <int P, typename IndexType>
    static void
    pack_MRxk(IndexType                     k,
              bool                          conj,
              const float                   *A,
              IndexType                     incRowA,
              IndexType                     incColA,
              float                         *buffer);

template <typename IndexType>
    static void
    pack_kxNR(IndexType                     k,
              bool                          conj,
              const float                   *B,
              IndexType                     incRowB,
              IndexType                     incColB,
              float                         *buffer);

template <typename IndexType>
    static bool
    pack_MRxk_partial(IndexType                     mr,
                      IndexType                     k,
                      bool                          conj,
                      const float                   *A,
                      IndexType                     incRowA,
                      IndexType                     incColA,
                      float                         *buffer);

template <typename IndexType>
    static bool
    pack_kxNR_partial(IndexType                     nr,
                      IndexType                     k,
                      bool                          conj,
                      const float                   *B,
                      IndexType                     incRowB,
                      IndexType                     incColB,
                      float                         *buffer);

//  The routines for complex float with the SME2 kernel (cpack.tcc).
template <int P, typename IndexType>
    static void
    pack_MRxk(IndexType                     k,
              bool                          conj,
              const std::complex<float>     *A,
              IndexType                     incRowA,
              IndexType                     incColA,
              std::complex<float>           *buffer);

template <typename IndexType>
    static void
    pack_kxNR(IndexType                     k,
              bool                          conj,
              const std::complex<float>     *B,
              IndexType                     incRowB,
              IndexType                     incColB,
              std::complex<float>           *buffer);

template <typename IndexType>
    static bool
    pack_MRxk_partial(IndexType                     mr,
                      IndexType                     k,
                      bool                          conj,
                      const std::complex<float>     *A,
                      IndexType                     incRowA,
                      IndexType                     incColA,
                      std::complex<float>           *buffer);

template <typename IndexType>
    static bool
    pack_kxNR_partial(IndexType                     nr,
                      IndexType                     k,
                      bool                          conj,
                      const std::complex<float>     *B,
                      IndexType                     incRowB,
                      IndexType                     incColB,
                      std::complex<float>           *buffer);

//  The routines for complex double with the SME2 kernel (zpack.tcc).
template <int P, typename IndexType>
    static void
    pack_MRxk(IndexType                     k,
              bool                          conj,
              const std::complex<double>    *A,
              IndexType                     incRowA,
              IndexType                     incColA,
              std::complex<double>          *buffer);

template <typename IndexType>
    static void
    pack_kxNR(IndexType                     k,
              bool                          conj,
              const std::complex<double>    *B,
              IndexType                     incRowB,
              IndexType                     incColB,
              std::complex<double>          *buffer);

template <typename IndexType>
    static bool
    pack_MRxk_partial(IndexType                     mr,
                      IndexType                     k,
                      bool                          conj,
                      const std::complex<double>    *A,
                      IndexType                     incRowA,
                      IndexType                     incColA,
                      std::complex<double>          *buffer);

template <typename IndexType>
    static bool
    pack_kxNR_partial(IndexType                     nr,
                      IndexType                     k,
                      bool                          conj,
                      const std::complex<double>    *B,
                      IndexType                     incRowB,
                      IndexType                     incColB,
                      std::complex<double>          *buffer);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_PACK_H

#include <ulmblas/level3/ukernel/sme/pack.tcc>
#include <ulmblas/level3/ukernel/sme/spack.tcc>
#include <ulmblas/level3/ukernel/sme/cpack.tcc>
#include <ulmblas/level3/ukernel/sme/zpack.tcc>
