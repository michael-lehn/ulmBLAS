#ifndef ULMBLAS_LEVEL3_GEMM_PACK_H
#define ULMBLAS_LEVEL3_GEMM_PACK_H 1

#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/ukernel/backend.h>

//
//  Packing of blocks for the macro kernel: PackA<Backend> packs an mc x kc
//  block of A into panels of MR rows, PackB<Backend> a kc x nc block of B
//  into panels of NR columns, with the packing of single panels of the
//  backend (ulmblas/level3/ukernel/backend.h).  The last panel of a block
//  is zero padded unless the backend packs partial panels.
//
namespace ulmBLAS {

template <typename Backend> struct PackA
{
    typedef typename Backend::T Buffer;

    template <typename IndexType, typename TA>
    static void
    run(IndexType mc, IndexType kc, bool conj, const TA *A, IndexType incRowA,
        IndexType incColA, Buffer *buffer);
};

template <typename Backend>
template <typename IndexType, typename TA>
void
PackA<Backend>::run(IndexType mc, IndexType kc, bool conj, const TA *A,
                    IndexType incRowA, IndexType incColA,
                    typename Backend::T *buffer)
{
    constexpr IndexType MR = Backend::MR;
    const IndexType mp = mc / MR;
    const IndexType mr_ = mc % MR;

    // P panels are packed together if the columns of A are at least D bytes
    // apart (parameters of the configuration).  More panels than ways of
    // the L1 cache would conflict: the panels are k*MR elements apart, for
    // large k a multiple of the size of a way.  Grouping pays off only if
    // the columns of A are far apart (measured on an Apple M5 and an Intel
    // Core i5-12500 with 8 KiB: up to 12% faster for m >= 1024 and small n,
    // but up to 7% slower for small matrices that are in the cache).
    constexpr int P = BlockSize<Buffer>::PACK_A_PANELS;
    constexpr int D = BlockSize<Buffer>::PACK_A_GROUP_DISTANCE;
    const IndexType dist = (incColA < 0) ? -incColA : incColA;
    const bool group = dist * IndexType(sizeof(TA)) >= D;

    IndexType p = 0;
    for (; group && p + P <= mp; p += P) {
	Backend::template packA<P>(kc, conj, A, incRowA, incColA, buffer);
	buffer += P * kc * MR;
	A += P * MR * incRowA;
    }
    for (; p < mp; ++p) {
	Backend::template packA<1>(kc, conj, A, incRowA, incColA, buffer);
	buffer += kc * MR;
	A += MR * incRowA;
    }
    if (mr_ > 0 &&
        !Backend::packAPartial(mr_, kc, conj, A, incRowA, incColA, buffer)) {
	for (IndexType j = 0; j < kc; ++j) {
	    for (IndexType i = 0; i < mr_; ++i) {
		buffer[i] =
		    (!conj) ? A[i * incRowA] : conjugate(A[i * incRowA]);
	    }
	    for (IndexType i = mr_; i < MR; ++i) {
		buffer[i] = Buffer(0);
	    }
	    buffer += MR;
	    A += incColA;
	}
    }
}

template <typename Backend> struct PackB
{
    typedef typename Backend::T Buffer;

    template <typename IndexType, typename TB>
    static void
    run(IndexType kc, IndexType nc, bool conj, const TB *B, IndexType incRowB,
        IndexType incColB, Buffer *buffer);
};

template <typename Backend>
template <typename IndexType, typename TB>
void
PackB<Backend>::run(IndexType kc, IndexType nc, bool conj, const TB *B,
                    IndexType incRowB, IndexType incColB,
                    typename Backend::T *buffer)
{
    constexpr IndexType NR = Backend::NR;
    const IndexType np = nc / NR;
    const IndexType nr_ = nc % NR;

    for (IndexType j = 0; j < np; ++j) {
	Backend::packB(kc, conj, B, incRowB, incColB, buffer);
	buffer += kc * NR;
	B += NR * incColB;
    }
    if (nr_ > 0 &&
        !Backend::packBPartial(nr_, kc, conj, B, incRowB, incColB, buffer)) {
	for (IndexType i = 0; i < kc; ++i) {
	    for (IndexType j = 0; j < nr_; ++j) {
		buffer[j] =
		    (!conj) ? B[j * incColB] : conjugate(B[j * incColB]);
	    }
	    for (IndexType j = nr_; j < NR; ++j) {
		buffer[j] = Buffer(0);
	    }
	    buffer += NR;
	    B += incRowB;
	}
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_GEMM_PACK_H
