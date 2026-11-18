#ifndef ULMBLAS_LEVEL3_UKERNEL_BACKEND_H
#define ULMBLAS_LEVEL3_UKERNEL_BACKEND_H 1

#include <cstddef>
#include <ulmblas/config/inline.h>
#include <ulmblas/level3/pack/panel.h>
#include <ulmblas/level3/ukernel/ugemm.h>

//
//  A backend of gemm: the micro kernel for elements of type T and the
//  packing of single panels into its format.  The frame algorithm, the
//  macro kernel and the organization of the packing (ulmblas/level3/gemm/)
//  are templates on the backend, so that its micro kernel and its packing
//  are part of their instances.
//
//      T              the type of the packed panels (compute type)
//      MR, NR         the size of a tile of C
//      ugemm          C <- beta C + alpha A B for a tile: full (MR x NR)
//                     or partial (mr x nr)
//      packA, packB   P panels of MR rows of A, a panel of NR columns of B
//                     (converting to T)
//      packAPartial,  a partial panel (mr < MR, nr < NR): true if packed,
//      packBPartial   false: the frame zero pads it
//
//  DefaultBackend<T> is the backend that the build selects for T: the
//  micro kernel of ulmblas/level3/ukernel/ugemm.h (SELECT_UGEMM_KERNEL) and
//  the packing of ulmblas/level3/pack/panel.h with the overloads of the
//  kernel (e.g. ulmblas/level3/ukernel/sme/pack.h).
//
namespace ulmBLAS {

template <typename T_> struct DefaultBackend
{
    typedef T_ T;

    static constexpr std::ptrdiff_t MR = BlockSizeUGemm<T>::MR;
    static constexpr std::ptrdiff_t NR = BlockSizeUGemm<T>::NR;
    //  the wrapper of gemm (ulmblas/level3/gemm.tcc) already computes C^T for
    //  kernels that prefer row-major C, the frame need not
    static constexpr bool rowPreference = false;

    template <typename IndexType, typename Beta, typename TC>
    static ULMBLAS_ALWAYS_INLINE void
    ugemm(IndexType kc, const T &alpha, const T *A, const T *B,
          const Beta &beta, TC *C, IndexType incRowC, IndexType incColC,
          const T *nextA, const T *nextB)
    {
	ulmBLAS::ugemm(kc, alpha, A, B, beta, C, incRowC, incColC, nextA,
	               nextB);
    }

    template <typename IndexType, typename Beta, typename TC>
    static ULMBLAS_ALWAYS_INLINE void
    ugemm(IndexType mr, IndexType nr, IndexType kc, const T &alpha, const T *A,
          const T *B, const Beta &beta, TC *C, IndexType incRowC,
          IndexType incColC, const T *nextA, const T *nextB)
    {
	ulmBLAS::ugemm(mr, nr, kc, alpha, A, B, beta, C, incRowC, incColC,
	               nextA, nextB);
    }

    template <int P, typename IndexType, typename TA>
    static ULMBLAS_ALWAYS_INLINE void
    packA(IndexType k, bool conj, const TA *A, IndexType incRowA,
          IndexType incColA, T *buffer)
    {
	pack_MRxk<P>(k, conj, A, incRowA, incColA, buffer);
    }

    template <typename IndexType, typename TA>
    static ULMBLAS_ALWAYS_INLINE bool
    packAPartial(IndexType mr, IndexType k, bool conj, const TA *A,
                 IndexType incRowA, IndexType incColA, T *buffer)
    {
	return pack_MRxk_partial(mr, k, conj, A, incRowA, incColA, buffer);
    }

    template <typename IndexType, typename TB>
    static ULMBLAS_ALWAYS_INLINE void
    packB(IndexType k, bool conj, const TB *B, IndexType incRowB,
          IndexType incColB, T *buffer)
    {
	pack_kxNR(k, conj, B, incRowB, incColB, buffer);
    }

    template <typename IndexType, typename TB>
    static ULMBLAS_ALWAYS_INLINE bool
    packBPartial(IndexType nr, IndexType k, bool conj, const TB *B,
                 IndexType incRowB, IndexType incColB, T *buffer)
    {
	return pack_kxNR_partial(nr, k, conj, B, incRowB, incColB, buffer);
    }
};

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_BACKEND_H
