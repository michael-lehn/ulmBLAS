#ifndef ULMBLAS_LEVEL3_PACK_GEPACK_TCC
#define ULMBLAS_LEVEL3_PACK_GEPACK_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/config/blocksize.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/ukernel/ugemm.h>

namespace ulmBLAS {

//
//  Packs P consecutive panels of MR rows.  The panels are packed together
//  column by column, so that P*MR elements of each column of A are read at
//  once (whole cache lines for incRowA == 1) instead of MR elements per
//  column and panel.
//
template <int P, typename IndexType, typename TA, typename Buffer>
static void
pack_MRxk(IndexType   k,
          bool        conj,
          const TA    *A,
          IndexType   incRowA,
          IndexType   incColA,
          Buffer      *buffer)
{
    constexpr IndexType MR = BlockSizeUGemm<Buffer>::MR;
    const IndexType ld = k*MR;

    if (!conj && incRowA==1) {
        for (IndexType j=0; j<k; ++j) {
            for (IndexType p=0; p<P; ++p) {
                for (IndexType i=0; i<MR; ++i) {
                    buffer[p*ld+i] = A[p*MR+i];
                }
            }
            buffer += MR;
            A      += incColA;
        }
    } else if (!conj) {
        for (IndexType j=0; j<k; ++j) {
            for (IndexType p=0; p<P; ++p) {
                for (IndexType i=0; i<MR; ++i) {
                    buffer[p*ld+i] = A[(p*MR+i)*incRowA];
                }
            }
            buffer += MR;
            A      += incColA;
        }
    } else {
        for (IndexType j=0; j<k; ++j) {
            for (IndexType p=0; p<P; ++p) {
                for (IndexType i=0; i<MR; ++i) {
                    buffer[p*ld+i] = conjugate(A[(p*MR+i)*incRowA]);
                }
            }
            buffer += MR;
            A      += incColA;
        }
    }
}

//
//  The last panel of A, with mr < MR rows, is zero padded to MR rows by
//  gepack_A.  A backend of the micro kernel can pack it instead (overloads
//  for its types, e.g. ulmblas/level3/ukernel/sme/pack.h); the result tells
//  whether the panel was packed.
//
template <typename IndexType, typename TA, typename Buffer>
static bool
pack_MRxk_partial(IndexType, IndexType, bool, const TA *, IndexType,
                  IndexType, Buffer *)
{
    return false;
}

template <typename IndexType, typename TA, typename Buffer>
void
gepack_A(IndexType   mc,
         IndexType   kc,
         bool        conj,
         const TA    *A,
         IndexType   incRowA,
         IndexType   incColA,
         Buffer      *buffer)
{
    constexpr IndexType MR  = BlockSizeUGemm<Buffer>::MR;
    const IndexType mp  = mc / MR;
    const IndexType mr_ = mc % MR;

    // P panels are packed together if the columns of A are at least D bytes
    // apart (parameters of the configuration).  More panels than ways of
    // the L1 cache would conflict: the panels are k*MR elements apart, for
    // large k a multiple of the size of a way.  Grouping pays off only if
    // the columns of A are far apart (measured on an Apple M5 and an Intel
    // Core i5-12500 with 8 KiB: up to 12% faster for m >= 1024 and small n,
    // but up to 7% slower for small matrices that are in the cache).
    constexpr int   P     = BlockSize<Buffer>::PACK_A_PANELS;
    constexpr int   D     = BlockSize<Buffer>::PACK_A_GROUP_DISTANCE;
    const IndexType dist  = (incColA<0) ? -incColA : incColA;
    const bool      group = dist*IndexType(sizeof(TA)) >= D;

    IndexType p=0;
    for (; group && p+P<=mp; p+=P) {
        pack_MRxk<P>(kc, conj, A, incRowA, incColA, buffer);
        buffer += P*kc*MR;
        A      += P*MR*incRowA;
    }
    for (; p<mp; ++p) {
        pack_MRxk<1>(kc, conj, A, incRowA, incColA, buffer);
        buffer += kc*MR;
        A      += MR*incRowA;
    }
    if (mr_>0 && !pack_MRxk_partial(mr_, kc, conj, A, incRowA, incColA,
                                    buffer))
    {
        for (IndexType j=0; j<kc; ++j) {
            for (IndexType i=0; i<mr_; ++i) {
                buffer[i] = (!conj) ? A[i*incRowA] : conjugate(A[i*incRowA]);
            }
            for (IndexType i=mr_; i<MR; ++i) {
                buffer[i] = Buffer(0);
            }
            buffer += MR;
            A      += incColA;
        }
    }
}

template <typename IndexType, typename TB, typename Buffer>
static void
pack_kxNR(IndexType   k,
          bool        conj,
          const TB    *B,
          IndexType   incRowB,
          IndexType   incColB,
          Buffer      *buffer)
{
    constexpr IndexType NR = BlockSizeUGemm<Buffer>::NR;

    if (!conj) {
        for (IndexType i=0; i<k; ++i) {
            for (IndexType j=0; j<NR; ++j) {
                buffer[j] = B[j*incColB];
            }
            buffer += NR;
            B      += incRowB;
        }
    } else {
        for (IndexType i=0; i<k; ++i) {
            for (IndexType j=0; j<NR; ++j) {
                buffer[j] = conjugate(B[j*incColB]);
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}

//
//  The same for the last panel of B, with nr < NR columns.
//
template <typename IndexType, typename TB, typename Buffer>
static bool
pack_kxNR_partial(IndexType, IndexType, bool, const TB *, IndexType,
                  IndexType, Buffer *)
{
    return false;
}

template <typename IndexType, typename TB, typename Buffer>
void
gepack_B(IndexType   kc,
         IndexType   nc,
         bool        conj,
         const TB    *B,
         IndexType   incRowB,
         IndexType   incColB,
         Buffer      *buffer)
{
    constexpr IndexType NR  = BlockSizeUGemm<Buffer>::NR;
    const IndexType np  = nc / NR;
    const IndexType nr_ = nc % NR;

    for (IndexType j=0; j<np; ++j) {
        pack_kxNR(kc, conj, B, incRowB, incColB, buffer);
        buffer += kc*NR;
        B      += NR*incColB;
    }
    if (nr_>0 && !pack_kxNR_partial(nr_, kc, conj, B, incRowB, incColB,
                                    buffer))
    {
        for (IndexType i=0; i<kc; ++i) {
            for (IndexType j=0; j<nr_; ++j) {
                buffer[j] = (!conj) ? B[j*incColB] : conjugate(B[j*incColB]);
            }
            for (IndexType j=nr_; j<NR; ++j) {
                buffer[j] = Buffer(0);
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_PACK_GEPACK_TCC
