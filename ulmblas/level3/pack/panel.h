#ifndef ULMBLAS_LEVEL3_PACK_PANEL_H
#define ULMBLAS_LEVEL3_PACK_PANEL_H 1

#include <cstddef>
#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/ukernel/ugemm.h>

//
//  Packing of single panels for the micro kernel of gemm, portable: a
//  panel of MR rows of A (packPanelA, P panels at once) or NR columns of B
//  (packPanelB), MR and NR as template parameters; pack_MRxk and pack_kxNR
//  with MR and NR of the kernel the build selects for the type.  The partial
//  panels (mr < MR rows, nr < NR columns) are zero padded by the frame
//  (ulmblas/level3/gemm/pack.h) unless a backend packs them (pack_MRxk_partial,
//  pack_kxNR_partial return true).  Backends provide overloads for their types
//  (ulmblas/level3/ukernel/sme/pack.h); they are declared before these
//  templates are used (ulmblas/level3/ukernel/backend.h).
//
namespace ulmBLAS {

//
//  Packs P consecutive panels of MR rows.  The panels are packed together
//  column by column, so that P*MR elements of each column of A are read at
//  once (whole cache lines for incRowA == 1) instead of MR elements per
//  column and panel.
//
template <std::ptrdiff_t MR_, int P, typename IndexType, typename TA,
          typename Buffer>
static void
packPanelA(IndexType k, bool conj, const TA *A, IndexType incRowA,
           IndexType incColA, Buffer *buffer)
{
    constexpr IndexType MR = MR_;
    const IndexType ld = k * MR;

    if (!conj && incRowA == 1) {
	for (IndexType j = 0; j < k; ++j) {
	    for (IndexType p = 0; p < P; ++p) {
		for (IndexType i = 0; i < MR; ++i) {
		    buffer[p * ld + i] = A[p * MR + i];
		}
	    }
	    buffer += MR;
	    A += incColA;
	}
    } else if (!conj) {
	for (IndexType j = 0; j < k; ++j) {
	    for (IndexType p = 0; p < P; ++p) {
		for (IndexType i = 0; i < MR; ++i) {
		    buffer[p * ld + i] = A[(p * MR + i) * incRowA];
		}
	    }
	    buffer += MR;
	    A += incColA;
	}
    } else {
	for (IndexType j = 0; j < k; ++j) {
	    for (IndexType p = 0; p < P; ++p) {
		for (IndexType i = 0; i < MR; ++i) {
		    buffer[p * ld + i] = conjugate(A[(p * MR + i) * incRowA]);
		}
	    }
	    buffer += MR;
	    A += incColA;
	}
    }
}

//  the same with MR of the build's kernel for Buffer
template <int P, typename IndexType, typename TA, typename Buffer>
static void
pack_MRxk(IndexType k, bool conj, const TA *A, IndexType incRowA,
          IndexType incColA, Buffer *buffer)
{
    packPanelA<BlockSizeUGemm<Buffer>::MR, P>(k, conj, A, incRowA, incColA,
                                              buffer);
}

//
//  The last panel of A, with mr < MR rows, is zero padded to MR rows by
//  gepack_A.  A backend of the micro kernel can pack it instead (overloads
//  for its types, e.g. ulmblas/level3/ukernel/sme/pack.h); the result tells
//  whether the panel was packed.
//
template <typename IndexType, typename TA, typename Buffer>
static bool
pack_MRxk_partial(IndexType mr, IndexType k, bool conj, const TA *A,
                  IndexType incRowA, IndexType incColA, Buffer *buffer)
{
    return false;
}

template <std::ptrdiff_t NR_, typename IndexType, typename TB, typename Buffer>
static void
packPanelB(IndexType k, bool conj, const TB *B, IndexType incRowB,
           IndexType incColB, Buffer *buffer)
{
    constexpr IndexType NR = NR_;

    if (!conj) {
	for (IndexType i = 0; i < k; ++i) {
	    for (IndexType j = 0; j < NR; ++j) {
		buffer[j] = B[j * incColB];
	    }
	    buffer += NR;
	    B += incRowB;
	}
    } else {
	for (IndexType i = 0; i < k; ++i) {
	    for (IndexType j = 0; j < NR; ++j) {
		buffer[j] = conjugate(B[j * incColB]);
	    }
	    buffer += NR;
	    B += incRowB;
	}
    }
}

//  the same with NR of the build's kernel for Buffer
template <typename IndexType, typename TB, typename Buffer>
static void
pack_kxNR(IndexType k, bool conj, const TB *B, IndexType incRowB,
          IndexType incColB, Buffer *buffer)
{
    packPanelB<BlockSizeUGemm<Buffer>::NR>(k, conj, B, incRowB, incColB,
                                           buffer);
}

//
//  The same for the last panel of B, with nr < NR columns.
//
template <typename IndexType, typename TB, typename Buffer>
static bool
pack_kxNR_partial(IndexType nr, IndexType k, bool conj, const TB *B,
                  IndexType incRowB, IndexType incColB, Buffer *buffer)
{
    return false;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_PACK_PANEL_H
