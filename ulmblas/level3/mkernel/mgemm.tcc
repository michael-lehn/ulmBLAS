#ifndef ULMBLAS_LEVEL3_MKERNEL_MGEMM_TCC
#define ULMBLAS_LEVEL3_MKERNEL_MGEMM_TCC 1

#include <complex>
#include <ulmblas/level3/mkernel/mgemm.h>
#include <ulmblas/level3/ukernel/ugemm.h>

namespace ulmBLAS {

//
//  Kernels that can compute several tiles of one panel of B per call (the
//  OpenBLAS-derived kernels, ULMBLAS_BLIS_DGEMM_PANEL, see
//  ulmblas/level3/ukernel/blis/ugemm.h): MgemmPanel::run computes the
//  full tiles of the panel and returns their number, otherwise 0.
//
template <typename IndexType, typename T, typename Beta, typename TC>
struct MgemmPanel
{
    static IndexType
    run(IndexType, IndexType, const T &, const T *, const T *, const Beta &,
        TC *, IndexType, IndexType)
    {
        return 0;
    }
};

#if defined(ULMBLAS_BLIS_DGEMM_PANEL)
template <typename IndexType>
struct MgemmPanel<IndexType, double, double, double>
{
    static IndexType
    run(IndexType mp, IndexType kc, const double &alpha, const double *A,
        const double *B, const double &beta, double *C, IndexType incRowC,
        IndexType incColC)
    {
        if (mp==0 || incRowC!=IndexType(1)) {
            return 0;
        }
        blisk::ugemm_panel(mp, kc, alpha, A, B, beta, C, incColC);
        return mp;
    }
};
#endif

//
//  Kernels that can compute all full tiles of a block with one call
//  (ULMBLAS_BLIS_DGEMM_BLOCK, see ulmblas/level3/ukernel/blis/ugemm.h):
//  MgemmBlock::run computes them and returns the number of panels of B it
//  computed (all full ones), otherwise 0.
//
template <typename IndexType, typename T, typename Beta, typename TC>
struct MgemmBlock
{
    static IndexType
    run(IndexType, IndexType, IndexType, const T &, const T *, const T *,
        const Beta &, TC *, IndexType, IndexType)
    {
        return 0;
    }
};

#if defined(ULMBLAS_BLIS_DGEMM_BLOCK)
template <typename IndexType>
struct MgemmBlock<IndexType, double, double, double>
{
    static IndexType
    run(IndexType mp, IndexType np, IndexType kc, const double &alpha,
        const double *A, const double *B, const double &beta, double *C,
        IndexType incRowC, IndexType incColC)
    {
        if (mp==0 || np==0 || incRowC!=IndexType(1)) {
            return 0;
        }
        return blisk::ugemm_block(mp, np, kc, alpha, A, B, beta, C, incColC)
             ? np : 0;
    }
};
#endif

//
//  The same for the vendored kernels of the other types with block calls
//  (blisk::Kernel<T>::blockcall, ulmblas/level3/ukernel/blis/xugemm.h);
//  -DULMBLAS_SGEMM_NO_PANEL, -DULMBLAS_ZGEMM_NO_PANEL switch them off (one
//  call per tile).
//
#if defined(USE_BLIS_XGEMM)
template <typename IndexType, typename T>
struct MgemmBlockKernel
{
    static IndexType
    run(IndexType mp, IndexType np, IndexType kc, const T &alpha,
        const T *A, const T *B, const T &beta, T *C, IndexType incRowC,
        IndexType incColC)
    {
        if (mp==0 || np==0 || incRowC!=IndexType(1)) {
            return 0;
        }
        return blisk::xugemm_block(mp, np, kc, alpha, A, B, beta, C, incColC)
             ? np : 0;
    }
};

#if defined(USE_BLIS_ZGEMM) && ULMBLAS_BLIS_ZGEMM_BLOCKCALL \
 && !defined(ULMBLAS_ZGEMM_NO_PANEL)
template <typename IndexType>
struct MgemmBlock<IndexType, std::complex<double>, std::complex<double>,
                  std::complex<double> >
    : MgemmBlockKernel<IndexType, std::complex<double> >
{
};
#endif

#if defined(USE_BLIS_SGEMM) && ULMBLAS_BLIS_SGEMM_BLOCKCALL \
 && !defined(ULMBLAS_SGEMM_NO_PANEL)
template <typename IndexType>
struct MgemmBlock<IndexType, float, float, float>
    : MgemmBlockKernel<IndexType, float>
{
};
#endif

#if defined(USE_BLIS_CGEMM) && ULMBLAS_BLIS_CGEMM_BLOCKCALL \
 && !defined(ULMBLAS_CGEMM_NO_PANEL)
template <typename IndexType>
struct MgemmBlock<IndexType, std::complex<float>, std::complex<float>,
                  std::complex<float> >
    : MgemmBlockKernel<IndexType, std::complex<float> >
{
};
#endif
#endif // USE_BLIS_XGEMM

template <typename IndexType, typename T, typename Beta, typename TC>
void
mgemm(IndexType     mc,
      IndexType     nc,
      IndexType     kc,
      const T       &alpha,
      const T       *A_,
      const T       *B_,
      const Beta    &beta,
      TC            *C,
      IndexType     incRowC,
      IndexType     incColC)
{
    constexpr IndexType MR = BlockSizeUGemm<T>::MR;
    constexpr IndexType NR = BlockSizeUGemm<T>::NR;

    const IndexType mp = (mc+MR-1) / MR;
    const IndexType np = (nc+NR-1) / NR;

    const IndexType mr_ = mc % MR;
    const IndexType nr_ = nc % NR;

    IndexType mr, nr;

    const T *nextA;
    const T *nextB;

    //  all full tiles with one call (OpenBLAS kernels that support it)
    const IndexType jb = MgemmBlock<IndexType, T, Beta, TC>::run(mc/MR, nc/NR,
                                                                 kc, alpha,
                                                                 A_, B_, beta,
                                                                 C, incRowC,
                                                                 incColC);

    for (IndexType j=0; j<np; ++j) {
        nr    = (j!=np-1 || nr_==0) ? NR : nr_;
        nextB = &B_[j*kc*NR];

        IndexType i0 = 0;
        if (j<jb) {
            i0 = mc/MR;
        } else if (nr==NR) {
            i0 = MgemmPanel<IndexType, T, Beta, TC>::run(mc/MR, kc, alpha,
                                                         A_, &B_[j*kc*NR],
                                                         beta,
                                                         &C[j*NR*incColC],
                                                         incRowC, incColC);
        }

        for (IndexType i=i0; i<mp; ++i) {
            mr    = (i!=mp-1 || mr_==0) ? MR : mr_;
            nextA = &A_[(i+1)*kc*MR];

            if (i==mp-1) {
                nextA = A_;
                nextB = &B_[(j+1)*kc*NR];
                if (j==np-1) {
                    nextB = B_;
                }
            }

            if (mr==MR && nr==NR) {
                ugemm(kc,
                      alpha, &A_[i*kc*MR], &B_[j*kc*NR],
                      beta,
                      &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC,
                      nextA, nextB);
            } else {
                // Call the buffered micro kernel
                ugemm(mr, nr, kc,
                      alpha, &A_[i*kc*MR], &B_[j*kc*NR],
                      beta,
                      &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC,
                      nextA, nextB);
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_MKERNEL_MGEMM_TCC
