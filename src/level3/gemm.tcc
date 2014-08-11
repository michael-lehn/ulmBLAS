#ifndef ULMBLAS_SRC_LEVEL3_GEMM_TCC
#define ULMBLAS_SRC_LEVEL3_GEMM_TCC 1

#include <stdio.h>

#include <src/config/blocksize.h>
#include <src/auxiliary/memorypool.h>
#include <src/auxiliary/memorypool.tcc>
#include <src/level1extensions/geaxpy.h>
#include <src/level1extensions/geaxpy.tcc>
#include <src/level1extensions/gescal.h>
#include <src/level1extensions/gescal.tcc>

#if defined(__SSE3__)
#include <src/level3/kernel/sse.h>
#include <src/level3/kernel/sse.tcc>
#endif

namespace ulmBLAS {

template <typename IndexType, typename TA, typename Buffer>
static void
pack_MRxk(IndexType k, const TA *A, IndexType incRowA, IndexType incColA,
          Buffer *buffer)
{
    const IndexType MR = BlockSize<Buffer>::MR;

    for (IndexType j=0; j<k; ++j) {
        for (IndexType i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}

template <typename IndexType, typename TA, typename Buffer>
static void
pack_A(IndexType mc, IndexType kc,
       const TA *A, IndexType incRowA, IndexType incColA,
       Buffer *buffer)
{
    const IndexType MR = BlockSize<Buffer>::MR;

    IndexType mp  = mc / MR;
    IndexType _mr = mc % MR;

    for (IndexType i=0; i<mp; ++i) {
        pack_MRxk(kc, A, incRowA, incColA, buffer);
        buffer += kc*MR;
        A      += MR*incRowA;
    }
    if (_mr>0) {
        for (IndexType j=0; j<kc; ++j) {
            for (IndexType i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (IndexType i=_mr; i<MR; ++i) {
                buffer[i] = 0.0;
            }
            buffer += MR;
            A      += incColA;
        }
    }
}

template <typename IndexType, typename TB, typename Buffer>
static void
pack_kxNR(IndexType k, const TB *B, IndexType incRowB, IndexType incColB,
          Buffer *buffer)
{
    const IndexType NR = BlockSize<Buffer>::NR;

    for (IndexType i=0; i<k; ++i) {
        for (IndexType j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
        B      += incRowB;
    }
}

template <typename IndexType, typename TB, typename Buffer>
static void
pack_B(IndexType kc, IndexType nc,
       const TB *B, IndexType incRowB, IndexType incColB,
       Buffer *buffer)
{
    const IndexType NR = BlockSize<Buffer>::NR;

    IndexType np  = nc / NR;
    IndexType _nr = nc % NR;

    for (IndexType j=0; j<np; ++j) {
        pack_kxNR(kc, B, incRowB, incColB, buffer);
        buffer += kc*NR;
        B      += NR*incColB;
    }
    if (_nr>0) {
        for (IndexType i=0; i<kc; ++i) {
            for (IndexType j=0; j<_nr; ++j) {
                buffer[j] = B[j*incColB];
            }
            for (IndexType j=_nr; j<NR; ++j) {
                buffer[j] = 0.0;
            }
            buffer += NR;
            B      += incRowB;
        }
    }
}

template <typename IndexType, typename T, typename TC>
static void
gemm_micro_kernel(IndexType kc,
                  const T &alpha, const T *A, const T *B,
                  const TC &beta, TC *C, IndexType incRowC, IndexType incColC,
                  const T *nextA, const T *nextB)
{
    const IndexType MR = BlockSize<T>::MR;
    const IndexType NR = BlockSize<T>::NR;

    T AB[MR*NR];

    for (IndexType i=0; i<MR*NR; ++i) {
        AB[i] = T(0);
    }

    for (IndexType l=0; l<kc; ++l) {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                AB[i+j*MR] += A[i]*B[j];
            }
        }
        A += MR;
        B += NR;
    }

    if (beta==0.0) {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] = TC(0);
            }
        }
    } else {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] *= beta;
            }
        }
    }

    if (alpha==1.0) {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += AB[i+j*MR];
            }
        }
    } else {
        for (IndexType j=0; j<NR; ++j) {
            for (IndexType i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += alpha*AB[i+j*MR];
            }
        }
    }
}

template <typename IndexType, typename T, typename Beta, typename TC>
static void
buffered_kernel(IndexType mr, IndexType nr, IndexType kc,
                const T &alpha, const T *A, const T *B,
                const Beta &beta,
                TC *C, IndexType incRowC, IndexType incColC,
                const T *nextA, const T *nextB)
{
    const IndexType MR = BlockSize<T>::MR;
    const IndexType NR = BlockSize<T>::NR;

    T   _C[MR*NR];

    gemm_micro_kernel(kc, alpha, A, B,
                      T(0), _C, IndexType(1), MR,
                      nextA, nextB);
    gescal(mr, nr, beta, C, incRowC, incColC);
    geaxpy(mr, nr, Beta(1), _C, IndexType(1), MR, C, incRowC, incColC);
}

template <typename IndexType, typename Alpha, typename T, typename Beta,
          typename TC>
static void
gemm_macro_kernel(IndexType     mc,
                  IndexType     nc,
                  IndexType     kc,
                  const Alpha   &alpha,
                  T             *_A,
                  T             *_B,
                  const Beta    &beta,
                  TC            *C,
                  IndexType     incRowC,
                  IndexType     incColC)
{
    const IndexType MR = BlockSize<T>::MR;
    const IndexType NR = BlockSize<T>::NR;

    IndexType mp = (mc+MR-1) / MR;
    IndexType np = (nc+NR-1) / NR;

    IndexType _mr = mc % MR;
    IndexType _nr = nc % NR;

    IndexType mr, nr;

    const T *nextA;
    const T *nextB;

    for (IndexType j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;
        nextB = &_B[j*kc*NR];

        for (IndexType i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;
            nextA = &_A[(i+1)*kc*MR];

            if (i==mp-1) {
                nextA = _A;
                nextB = &_B[(j+1)*kc*NR];
                if (j==np-1) {
                    nextB = _B;
                }
            }

            if (mr==MR && nr==NR) {
                gemm_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                                  beta,
                                  &C[i*MR*incRowC+j*NR*incColC],
                                  incRowC, incColC,
                                  nextA, nextB);
            } else {
                buffered_kernel(mr, nr, kc,
                                alpha, &_A[i*kc*MR], &_B[j*kc*NR],
                                beta,
                                &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC,
                                nextA, nextB);
            }
        }
    }
}

//double _A[384*384];
//double _B[384*4096];

template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
void
gemm(IndexType    m,
     IndexType    n,
     IndexType    k,
     const Alpha  &alpha,
     const TA     *A,
     IndexType    incRowA,
     IndexType    incColA,
     const TB     *B,
     IndexType    incRowB,
     IndexType    incColB,
     const Beta   &beta,
     TC           *C,
     IndexType    incRowC,
     IndexType    incColC)
{
    typedef decltype(Alpha(0)*TA(0)*TB(0))  T;

    const IndexType MC = BlockSize<T>::MC;
    const IndexType NC = BlockSize<T>::NC;
    const IndexType KC = BlockSize<T>::KC;

    const IndexType mb = (m+MC-1) / MC;
    const IndexType nb = (n+NC-1) / NC;
    const IndexType kb = (k+KC-1) / KC;

    const IndexType _mc = m % MC;
    const IndexType _nc = n % NC;
    const IndexType _kc = k % KC;

    static MemoryPool<T> memoryPool;

    T  *_A = memoryPool.allocate(MC*KC);
    T  *_B = memoryPool.allocate(KC*NC);

    //T *_A = new T[MC*KC];
    //T *_B = new T[KC*NC];

    if (alpha==0.0 || k==0) {
        gescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    for (IndexType j=0; j<nb; ++j) {
        IndexType nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (IndexType l=0; l<kb; ++l) {
            IndexType kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
            Beta      _beta = (l==0) ? beta : 1.0;

            pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                   _B);

            for (IndexType i=0; i<mb; ++i) {
                IndexType mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_A(mc, kc,
                       &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                       _A);

                gemm_macro_kernel(mc, nc, kc, alpha, _A, _B, _beta,
                                  &C[i*MC*incRowC+j*NC*incColC],
                                  incRowC, incColC);
            }
        }
    }

    //delete [] _B;
    //delete [] _A;
    memoryPool.release(_A);
    memoryPool.release(_B);
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL3_GEMM_TCC
