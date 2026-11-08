#ifndef ULMBLAS_LAPACK_TASKS_INVERSETASKS_TCC
#define ULMBLAS_LAPACK_TASKS_INVERSETASKS_TCC 1

//
//  pztrtri, lower (step k, tiles of A, X = inv(A)):
//
//      TRSM   A(m,k) := -A(m,k) A(k,k)^{-1}    m > k     in (k,k), inout (m,k)
//      GEMM   A(m,n) += A(m,k) A(k,n)          m > k > n in (m,k), (k,n), inout (m,n)
//      TRSM   A(k,n) := A(k,k)^{-1} A(k,n)     n < k     in (k,k), inout (k,n)
//      TRTRI  A(k,k) := A(k,k)^{-1}                      inout (k,k)
//
//  pzlauum, upper (step k, A := U U^H):
//
//      HERK   A(n,n) += A(n,k) A(n,k)^H        n < k     in (n,k), inout (n,n)
//      GEMM   A(n,m) += A(n,k) A(m,k)^H        n < m < k in (n,k), (m,k), inout (n,m)
//      TRMM   A(n,k) := A(n,k) A(k,k)^H        n < k     in (k,k), inout (n,k)
//      LAUUM  A(k,k) := A(k,k) A(k,k)^H                  inout (k,k)
//

#include <algorithm>
#include <ulmblas/lapack/buffer.h>
#include <ulmblas/lapack/lauum.h>
#include <ulmblas/lapack/tasks/inversetasks.h>
#include <ulmblas/lapack/tasks/sync.h>
#include <ulmblas/lapack/trtri.h>
#include <ulmblas/level3/gemm.h>
#include <ulmblas/level3/trlsm.h>
#include <ulmblas/level3/trumm.h>
#include <ulmblas/level3/trusm.h>

namespace ulmBLAS { namespace tasks {

template <typename IndexType, typename T>
void
trtriLowerTasks(bool         unitDiag,
                IndexType    n,
                T            *A,
                IndexType    incRowA,
                IndexType    incColA,
                IndexType    nb,
                int          threads)
{
    const int PRIO_CRIT  = 2;
    const int PRIO_TRSM  = 1;
    const int PRIO_OTHER = 0;
    (void) PRIO_CRIT; (void) PRIO_TRSM; (void) PRIO_OTHER; (void) threads;

    const IndexType rs = incRowA, cs = incColA;
    const IndexType nt = (n+nb-1)/nb;
    auto tile = [=](IndexType i, IndexType j) { return &A[i*nb*rs+j*nb*cs]; };
    auto size = [=](IndexType i) { return std::min(nb, n-i*nb); };

    Buffer<char> tok(std::size_t(nt*nt));
    char         *Tk = tok.data();
    (void) Tk;

    #pragma omp parallel num_threads(threads)
    #pragma omp single
    {
        for (IndexType k=0; k<nt; ++k) {
            const IndexType nk  = size(k);
            T               *Akk = tile(k, k);

            for (IndexType m=k+1; m<nt; ++m) {
                const IndexType nm  = size(m);
                T               *Amk = tile(m, k);
                //  A(m,k) := -A(m,k) A(k,k)^{-1}, i.e. A(k,k)^T X^T = -A(m,k)^T
                #pragma omp task depend(in: Tk[k*nt+k]) depend(inout: Tk[m*nt+k]) \
                                 priority(PRIO_CRIT)
                {
                    taskEnter();
                    trusm(nk, nm, T(-1), false, unitDiag, Akk, cs, rs, Amk, cs, rs);
                    taskLeave();
                }
            }
            for (IndexType m=k+1; m<nt; ++m) {
                const IndexType nm  = size(m);
                const T         *Amk = tile(m, k);
                for (IndexType j=0; j<k; ++j) {
                    const IndexType nj  = size(j);
                    const T         *Akj = tile(k, j);
                    T               *Amj = tile(m, j);
                    #pragma omp task depend(in: Tk[m*nt+k], Tk[k*nt+j]) \
                                     depend(inout: Tk[m*nt+j]) priority(PRIO_OTHER)
                    {
                        taskEnter();
                        gemm(nm, nj, nk, T(1), false, Amk, rs, cs,
                             false, Akj, rs, cs, T(1), Amj, rs, cs);
                        taskLeave();
                    }
                }
            }
            for (IndexType j=0; j<k; ++j) {
                const IndexType nj  = size(j);
                T               *Akj = tile(k, j);
                #pragma omp task depend(in: Tk[k*nt+k]) depend(inout: Tk[k*nt+j]) \
                                 priority(PRIO_TRSM)
                {
                    taskEnter();
                    trlsm(nk, nj, T(1), false, unitDiag, Akk, rs, cs, Akj, rs, cs);
                    taskLeave();
                }
            }
            #pragma omp task depend(inout: Tk[k*nt+k]) priority(PRIO_CRIT)
            {
                taskEnter();
                trtri(true, unitDiag, nk, Akk, rs, cs);
                taskLeave();
            }
        }
    }
}

template <typename IndexType, typename T>
void
lauumUpperTasks(IndexType    n,
                T            *A,
                IndexType    incRowA,
                IndexType    incColA,
                IndexType    nb,
                int          threads)
{
    const int PRIO_CRIT  = 1;
    const int PRIO_OTHER = 0;
    (void) PRIO_CRIT; (void) PRIO_OTHER; (void) threads;

    const IndexType rs = incRowA, cs = incColA;
    const IndexType nt = (n+nb-1)/nb;
    auto tile = [=](IndexType i, IndexType j) { return &A[i*nb*rs+j*nb*cs]; };
    auto size = [=](IndexType i) { return std::min(nb, n-i*nb); };

    Buffer<char> tok(std::size_t(nt*nt));
    char         *Tk = tok.data();
    (void) Tk;

    #pragma omp parallel num_threads(threads)
    #pragma omp single
    {
        for (IndexType k=0; k<nt; ++k) {
            const IndexType nk  = size(k);
            const T         *Akk = tile(k, k);

            for (IndexType i=0; i<k; ++i) {
                const IndexType ni  = size(i);
                const T         *Aik = tile(i, k);
                T               *Aii = tile(i, i);
                #pragma omp task depend(in: Tk[i*nt+k]) depend(inout: Tk[i*nt+i]) \
                                 priority(PRIO_OTHER)
                {
                    taskEnter();
                    lauum_impl::herkUpper(ni, nk, Aik, rs, cs, Aii, rs, cs);
                    taskLeave();
                }
                for (IndexType m=i+1; m<k; ++m) {
                    const IndexType nm  = size(m);
                    const T         *Amk = tile(m, k);
                    T               *Aim = tile(i, m);
                    #pragma omp task depend(in: Tk[i*nt+k], Tk[m*nt+k]) \
                                     depend(inout: Tk[i*nt+m]) priority(PRIO_OTHER)
                    {
                        taskEnter();
                        gemm(ni, nm, nk, T(1), false, Aik, rs, cs,
                             true, Amk, cs, rs, T(1), Aim, rs, cs);
                        taskLeave();
                    }
                }
            }
            for (IndexType i=0; i<k; ++i) {
                const IndexType ni  = size(i);
                T               *Aik = tile(i, k);
                //  A(i,k) := A(i,k) A(k,k)^H, i.e. A(i,k)^T := conj(A(k,k)) A(i,k)^T
                #pragma omp task depend(in: Tk[k*nt+k]) depend(inout: Tk[i*nt+k]) \
                                 priority(PRIO_CRIT)
                {
                    taskEnter();
                    trumm(nk, ni, T(1), true, false, Akk, rs, cs, Aik, cs, rs);
                    taskLeave();
                }
            }
            T *Akkw = tile(k, k);
            #pragma omp task depend(inout: Tk[k*nt+k]) priority(PRIO_CRIT)
            {
                taskEnter();
                lauum(false, nk, Akkw, rs, cs);
                taskLeave();
            }
        }
    }
}

} } // namespace tasks, ulmBLAS

#endif // ULMBLAS_LAPACK_TASKS_INVERSETASKS_TCC
