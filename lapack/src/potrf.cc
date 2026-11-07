//
//  DPOTRF: tiled Cholesky factorization with OpenMP task dependencies.
//
//  Right-looking tile algorithm (as PLASMA's pdpotrf), in place on the
//  column-major matrix.  The tiles are nb x nb blocks of A; a byte per tile
//  (tok) stands for the tile in the depend clauses:
//
//      for k = 0, ..., nt-1
//          POTRF  A(k,k) = L(k,k) L(k,k)^T             inout A(k,k)
//          for i > k
//              TRSM   A(i,k) = A(i,k) L(k,k)^{-T}      in A(k,k), inout A(i,k)
//          for i > k
//              SYRK   A(i,i) -= A(i,k) A(i,k)^T        in A(i,k), inout A(i,i)
//              for k < j < i
//                  GEMM  A(i,j) -= A(i,k) A(j,k)^T     in A(i,k), A(j,k),
//                                                      inout A(i,j)
//
//  The tasks on the critical path (POTRF, the TRSMs of column k and the
//  updates of column k+1) have higher priorities: the factorization of the
//  next column can start while the rest of the trailing matrix is updated
//  (lookahead by priority, as in PLASMA).
//
//  The upper variant A = U^T U is the lower variant of the transposed view
//  (row and column stride exchanged): U = L^T.
//

#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <limits>
#include <vector>

#include <omp.h>

#include <ulmlapack.h>
#include <ulmlapack.hpp>
#include "common.h"
#include "kernels.h"
#include "tiles.h"
#include "checkdeps.h"

namespace ulmlapack {

namespace {

const int PRIO_POTRF  = 3;
const int PRIO_CRIT   = 2;
const int PRIO_NEXT   = 1;
const int PRIO_OTHER  = 0;

//
//  W: the tiles the graph works on (in place, a column-major copy or the
//  tile layout).  src: the caller's matrix if W is a copy; then the graph
//  starts with a copy task per tile (lower triangle) and ends with one that
//  copies the tile back, as soon as its last update is done.
//
Index
potrf_tasks(Index n, Tiles W, const Source *src, Index nb, int threads,
            bool empty)
{
    const Index nt = (n+nb-1)/nb;
    const Index rs = W.rs, cs = W.cs;

    std::vector<char>  tok(nt*nt);
    char               *T = tok.data();
    (void) T;           // only used in depend clauses (GCC warns)

    // step of the first failed POTRF (max: none) and its INFO
    std::atomic<Index> failStep(std::numeric_limits<Index>::max());
    // empty tasks: every body is skipped as after a failure at step -1
    const Index skipFrom = empty ? -1 : std::numeric_limits<Index>::max();
    Index              info = 0;

    CHECKDEPS_LOG(log, std::size_t(nt)*nt*nt+5*nt*nt+nt);

    CHECKDEPS_TIME(tStart);
    CHECKDEPS_TIME(tGenEnd);
    #pragma omp parallel num_threads(threads) shared(failStep, info)
    #pragma omp single
    {
        if (src) {
            for (Index j=0; j<nt; ++j) {
                for (Index i=j; i<nt; ++i) {
                    const Index mi = std::min(nb, n-i*nb);
                    const Index nj = std::min(nb, n-j*nb);
                    double      *Wij = W.tile(i, j);
                    const double *Sij = &src->A[i*nb*src->rs+j*nb*src->cs];

                    CHECKDEPS_CREATE(rc, log, "COPYIN", -1, i, j);
                    CHECKDEPS_ACCESS(rc, i*nt+j, true);
                    #pragma omp task depend(out: T[i*nt+j]) \
                                     priority(j<=1 ? PRIO_CRIT : PRIO_OTHER)
                    {
                        taskEnter();
                        CHECKDEPS_START(rc);
                        if (!empty) {
                            copyBlock(mi, nj, Sij, src->rs, src->cs,
                                      Wij, rs, cs, i==j);
                        }
                        CHECKDEPS_END(rc);
                        taskLeave();
                    }
                }
            }
        }
        for (Index k=0; k<nt; ++k) {
            const Index  nk  = std::min(nb, n-k*nb);
            double       *Akk = W.tile(k, k);

            CHECKDEPS_CREATE(rp, log, "POTRF", k, k, k);
            CHECKDEPS_ACCESS(rp, k*nt+k, true);
            #pragma omp task depend(inout: T[k*nt+k]) priority(PRIO_POTRF)
            {
                taskEnter();
                CHECKDEPS_START(rp);
                if (std::min(skipFrom, failStep.load(std::memory_order_relaxed))>k) {
                    Index inf = potrf_lower_packed(nk, Akk, rs, cs);
                    if (inf) {
                        // tasks of later steps depend on this one, so the
                        // first failure is the one with the smallest k
                        info = k*nb+inf;
                        failStep.store(k);
                    }
                }
                CHECKDEPS_END(rp);
                taskLeave();
            }
            for (Index i=k+1; i<nt; ++i) {
                const Index  ni  = std::min(nb, n-i*nb);
                double       *Aik = W.tile(i, k);

                CHECKDEPS_CREATE(rt, log, "TRSM", k, i, k);
                CHECKDEPS_ACCESS(rt, k*nt+k, false);
                CHECKDEPS_ACCESS(rt, i*nt+k, true);
                #pragma omp task depend(in: T[k*nt+k]) \
                                 depend(inout: T[i*nt+k]) \
                                 priority(i==k+1 ? PRIO_POTRF : PRIO_CRIT)
                {
                    taskEnter();
                    CHECKDEPS_START(rt);
                    if (std::min(skipFrom, failStep.load(std::memory_order_relaxed))>k) {
                        // A(i,k) L(k,k)^T = A(i,k)
                        ulmBLAS::trlsm(nk, ni, 1.0, false, false,
                                       Akk, rs, cs, Aik, cs, rs);
                    }
                    CHECKDEPS_END(rt);
                    taskLeave();
                }
            }
            for (Index i=k+1; i<nt; ++i) {
                const Index  ni  = std::min(nb, n-i*nb);
                const double *Aik = W.tile(i, k);
                double       *Aii = W.tile(i, i);

                CHECKDEPS_CREATE(rs_, log, "SYRK", k, i, i);
                CHECKDEPS_ACCESS(rs_, i*nt+k, false);
                CHECKDEPS_ACCESS(rs_, i*nt+i, true);
                #pragma omp task depend(in: T[i*nt+k]) \
                                 depend(inout: T[i*nt+i]) \
                                 priority(i==k+1 ? PRIO_POTRF : PRIO_NEXT)
                {
                    taskEnter();
                    CHECKDEPS_START(rs_);
                    if (std::min(skipFrom, failStep.load(std::memory_order_relaxed))>k) {
                        ulmBLAS::sylrk(ni, nk, -1.0, Aik, rs, cs,
                                       1.0, Aii, rs, cs);
                    }
                    CHECKDEPS_END(rs_);
                    taskLeave();
                }
                for (Index j=k+1; j<i; ++j) {
                    const Index  nj  = std::min(nb, n-j*nb);
                    const double *Ajk = W.tile(j, k);
                    double       *Aij = W.tile(i, j);

                    CHECKDEPS_CREATE(rg, log, "GEMM", k, i, j);
                    CHECKDEPS_ACCESS(rg, i*nt+k, false);
                    CHECKDEPS_ACCESS(rg, j*nt+k, false);
                    CHECKDEPS_ACCESS(rg, i*nt+j, true);
                    #pragma omp task depend(in: T[i*nt+k], T[j*nt+k]) \
                                     depend(inout: T[i*nt+j]) \
                                     priority(j==k+1 ? PRIO_CRIT : PRIO_OTHER)
                    {
                        taskEnter();
                        CHECKDEPS_START(rg);
                        if (std::min(skipFrom, failStep.load(std::memory_order_relaxed))>k) {
                            ulmBLAS::gemm(ni, nj, nk, -1.0,
                                          false, Aik, rs, cs,
                                          false, Ajk, cs, rs,
                                          1.0, Aij, rs, cs);
                        }
                        CHECKDEPS_END(rg);
                        taskLeave();
                    }
                }
            }
        }
        if (src) {
            for (Index j=0; j<nt; ++j) {
                for (Index i=j; i<nt; ++i) {
                    const Index  mi  = std::min(nb, n-i*nb);
                    const Index  nj  = std::min(nb, n-j*nb);
                    const double *Wij = W.tile(i, j);
                    double       *Sij = &src->A[i*nb*src->rs+j*nb*src->cs];

                    CHECKDEPS_CREATE(rb, log, "COPYBACK", nt, i, j);
                    CHECKDEPS_ACCESS(rb, i*nt+j, false);
                    #pragma omp task depend(in: T[i*nt+j]) priority(PRIO_OTHER)
                    {
                        taskEnter();
                        CHECKDEPS_START(rb);
                        if (!empty) {
                            copyBlock(mi, nj, Wij, rs, cs,
                                      Sij, src->rs, src->cs, i==j);
                        }
                        CHECKDEPS_END(rb);
                        taskLeave();
                    }
                }
            }
        }
        CHECKDEPS_TIME_SET(tGenEnd);
    }
    CHECKDEPS_TIME(tEnd);
    CHECKDEPS_PROFILE(log, threads, tStart, tGenEnd, tEnd);
    if (CHECKDEPS_CHECK(log, "dpotrf")) {
        std::abort();
    }
    return info;
}

} // namespace

} // namespace ulmlapack

namespace ulmlapack {

Index
potrf(bool lower, Index n, double *A, Index incRowA, Index incColA,
      const ulmlapack_param *param)
{
    if (n<=0) {
        return 0;
    }
    ulmlapack_param p;
    if (param) {
        p = *param;
    } else {
        ulmlapack_param_init(&p);
    }
    ulmlapack_dpotrf_resolve(int(n), &p);

    // the upper variant is the lower one of the transposed view
    const Index rs = lower ? incRowA : incColA;
    const Index cs = lower ? incColA : incRowA;

    if (n<=p.serial_n || p.threads==1 || n<=p.nb) {
        return potrf_lower_packed(n, A, rs, cs);
    }
    return potrf_tasks(n, stridedTiles(A, rs, cs, p.nb), 0, p.nb, p.threads,
                       p.empty_tasks!=0);
}

namespace {

//  parameters and the transposed view, as in potrf
bool
potrfSetup(bool lower, Index n, Index incRowA, Index incColA,
           const ulmlapack_param *param, ulmlapack_param &p,
           Index &rs, Index &cs)
{
    if (param) {
        p = *param;
    } else {
        ulmlapack_param_init(&p);
    }
    ulmlapack_dpotrf_resolve(int(n), &p);
    rs = lower ? incRowA : incColA;
    cs = lower ? incColA : incRowA;
    return n<=p.serial_n || p.threads==1 || n<=p.nb;   // serial
}

} // namespace

Index
potrf_colmajor(bool lower, Index n, double *A, Index incRowA, Index incColA,
               const ulmlapack_param *param)
{
    if (n<=0) {
        return 0;
    }
    ulmlapack_param p;
    Index           rs, cs;
    if (potrfSetup(lower, n, incRowA, incColA, param, p, rs, cs)) {
        return potrf_lower_packed(n, A, rs, cs);
    }
    double       *B  = workspace(std::size_t(n)*n);
    const Source src = {A, rs, cs};
    return potrf_tasks(n, stridedTiles(B, 1, n, p.nb), &src, p.nb, p.threads,
                       p.empty_tasks!=0);
}

Index
potrf_tiled(bool lower, Index n, double *A, Index incRowA, Index incColA,
            const ulmlapack_param *param)
{
    if (n<=0) {
        return 0;
    }
    ulmlapack_param p;
    Index           rs, cs;
    if (potrfSetup(lower, n, incRowA, incColA, param, p, rs, cs)) {
        return potrf_lower_packed(n, A, rs, cs);
    }
    const Index  nt  = (n+p.nb-1)/p.nb;
    double       *B  = workspace(std::size_t(nt)*nt*p.nb*p.nb);
    const Source src = {A, rs, cs};
    return potrf_tasks(n, tiledTiles(B, nt, p.nb), &src, p.nb, p.threads,
                       p.empty_tasks!=0);
}

Index
potrf(TileMatrix &A, const ulmlapack_param *param)
{
    const Index n = A.n;
    if (n<=0) {
        return 0;
    }
    ulmlapack_param p;
    if (param) {
        p = *param;
    } else {
        ulmlapack_param_init(&p);
    }
    p.nb = int(A.nb);
    ulmlapack_dpotrf_resolve(int(n), &p);
    const Tiles W = tiledTiles(A.tile(0, 0), A.mt, A.nb);
    if (A.nt==1) {
        return potrf_lower_rec(n, W.tile(0, 0), 1, A.nb);
    }
    //  no serial algorithm across tiles: the task graph (also with one
    //  thread)
    return potrf_tasks(n, W, 0, A.nb, p.threads, p.empty_tasks!=0);
}

} // namespace ulmlapack

using ulmlapack::Index;

extern "C" {

void
ulmlapack_param_init(ulmlapack_param *p)
{
    p->nb            = 0;
    p->ib            = 0;
    p->panel_threads = 0;
    p->threads       = 0;
    p->serial_n      = -1;
    p->empty_tasks   = 0;
}

void
ulmlapack_dpotrf_resolve(int n, ulmlapack_param *p)
{
    // Defaults tuned on one socket of a Xeon Platinum 8358 (Ice Lake, 32
    // cores) with sizes off the benchmark grid; see README (Tuning).
    if (p->threads<=0) {
        p->threads = omp_get_max_threads();
    }
    if (p->nb<=0) {
        p->nb = (n<250) ? 32 : (n<600) ? 48 : (n<900) ? 64 : (n<2200) ? 96 : 128;
    }
    if (p->serial_n<0) {
        p->serial_n = 100;
    }
    // about one thread per 64 rows (at least 4): more threads only add
    // idle time and synchronization for small and medium n
    p->threads = std::max(1, std::min(p->threads, std::max(4, n/64)));
    // more threads than tiles of the trailing matrix do not help
    const Index nt = (n+p->nb-1)/p->nb;
    const Index tiles = nt*(nt+1)/2;
    p->threads = (int) std::max<Index>(1, std::min<Index>(p->threads, tiles));
}

int
ulmlapack_dpotrf_param(char uplo, int n, double *A, int ldA,
                       const ulmlapack_param *param)
{
    const bool lower = (uplo=='L' || uplo=='l');
    if (!lower && uplo!='U' && uplo!='u') {
        return -1;
    }
    if (n<0) {
        return -2;
    }
    if (ldA<std::max(1, n)) {
        return -4;
    }
    return (int) ulmlapack::potrf(lower, n, A, Index(1), Index(ldA), param);
}

int
ulmlapack_dpotrf(char uplo, int n, double *A, int ldA)
{
    return ulmlapack_dpotrf_param(uplo, n, A, ldA, 0);
}

} // extern "C"
