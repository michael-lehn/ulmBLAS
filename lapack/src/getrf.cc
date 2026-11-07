//
//  DGETRF: LU factorization with partial pivoting, OpenMP tasks with
//  dependencies.
//
//  In place on the column-major matrix (no tile layout).  The matrix is
//  divided into nb x nb tiles; a byte per tile (tok) stands for the tile in
//  the depend clauses.  Step k (panel = column block k, rows k*nb, ..., m-1):
//
//      PANEL(k)       LU with partial pivoting of the panel      inout tiles (i,k), i >= k
//      SWAP_TRSM(k,j) interchanges of panel k in column block j,
//                     U(k,j) = L(k,k)^{-1} A(k,j)                in (k,k), inout (i,j), i >= k
//      GEMM(k,i,j)    A(i,j) -= L(i,k) U(k,j)                    in (i,k), (k,j), inout (i,j)
//      LSWAP(k,c)     interchanges of panel k in an earlier
//                     column block c (the L part, "pivoting
//                     to the left")                              in (k,k), inout (i,c), i >= k
//
//  The interchanges of a panel act on all rows below k*nb, so the tasks
//  that apply them depend on the whole column of tiles (OpenMP 5.0
//  iterator in the depend clause); the GEMMs depend on single tiles and run
//  in parallel.  PLASMA uses one task per column of the trailing matrix
//  for the interchanges, the TRSM and the GEMMs (nested tasks and a
//  taskwait) and applies the pivoting to the left after the factorization.
//  Here the GEMMs are tasks of their own (no taskwait, a waiting thread
//  never idles) and the pivoting to the left runs as soon as a panel is
//  done, overlapped with the rest.
//
//  Priorities: PANEL > tasks of column block k+1 (lookahead: the next
//  panel can start while the rest of the trailing matrix is updated) >
//  other updates > LSWAP.
//
//  The panel is the critical path.  It is factored either by the serial
//  recursive algorithm (getrf_rec) or by a team of threads (see
//  panel_team): every thread of the team owns a contiguous range of rows
//  of the panel and keeps it in its cache; per column the threads search
//  their rows for the pivot candidate, meet at a barrier, and all compute
//  the pivot; the interchange and the update of the own rows follow
//  without another barrier.  The panel is processed in blocks of ib
//  columns (right-looking inside the panel, the update of the rest of the
//  panel with ulmBLAS's gemm).
//

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <omp.h>

#include <ulmlapack.h>
#include <ulmlapack.hpp>
#include <ulmblas/lapack/safemin.h>
#include "common.h"
#include "kernels.h"
#include "checkdeps.h"
#include "tiles.h"

namespace ulmlapack {

namespace {

const int PRIO_PANEL = 3;
const int PRIO_NEXT  = 2;
const int PRIO_UPD   = 1;
const int PRIO_LSWAP = 0;

//
//  Panel team.
//
//  PLASMA starts a fixed number of tasks for the panel which spin at
//  barriers until all of them run.  If the OpenMP runtime executes such a
//  task immediately in the creating thread (libgomp and LLVM do that when
//  many tasks are queued), or if all other threads are busy, the team
//  waits or deadlocks.  Here the team is formed dynamically: the panel task
//  creates helper tasks and waits at most joinWait seconds for helpers to
//  join; then it closes the team.  Helpers that start later, or that run in
//  the panel's own thread, return at once.  The panel therefore uses the
//  threads that are idle when it starts (with priorities: the next ones to
//  become free), and never waits for a thread that is not running.
//
struct Panel
{
    // panel
    Index        m, n;
    double       *A;
    Index        rs, cs;         // row and column stride
    double       *orig;          // packed: the panel itself (else null),
    Index        ors, ocs;       // A is a column-major copy (rs = 1, cs = m)
    Index        *piv;
    Index        ib;

    // team
    int                 creator;        // omp thread number of the panel task
    int                 excluded;       // thread that must not join (or -1)
    int                 maxSize;
    std::atomic<int>    state;          // members (bits 0..29), closed (bit 30)
    std::atomic<int>    started;        // members that have read the size
    int                 size;           // final size
    Index               chunk;          // rows per member
    SpinBarrier         barrier;

    // per column, double buffered: candidates of the members
    std::vector<double> candVal;        // [2][size]
    std::vector<Index>  candIdx;        // [2][size]
    std::vector<double> candRow;        // [2][size][ib]
    std::vector<double> topRow;         // [2][ib]

    Index               info;           // written by member 0
};

const int CLOSED = 1<<30;

//  diagnostics: panels factored by the task algorithm, by a team, members
std::atomic<long> statPanels(0), statTeams(0), statMembers(0);

//  Try to join; returns the member number or -1.
int
join(Panel &P)
{
    int s = P.state.load(std::memory_order_acquire);
    while (!(s & CLOSED) && s<P.maxSize) {
        if (P.state.compare_exchange_weak(s, s+1,
                                          std::memory_order_acq_rel)) {
            return s;           // members 1, 2, ... (0 is the creator)
        }
    }
    return -1;
}

//
//  One member of a panel team.  UnitRows: row stride 1 (column-major
//  storage, the common case), so that the loops over the rows of a column
//  are contiguous; otherwise any row stride rs.
//
template <bool UnitRows>
void
member(Panel &P, int r)
{
    const Index   m      = P.m;
    const Index   n      = P.n;
    const Index   kmin   = std::min(m, n);
    const Index   ib     = P.ib;
    const Index   rs     = UnitRows ? Index(1) : P.rs;
    const Index   cs     = P.cs;
    const int     size   = P.size;
    const Index   lo     = std::min(m, r*P.chunk);
    const Index   hi     = std::min(m, lo+P.chunk);
    const double  sfmin  = ulmBLAS::safeMin<double>();
    double        *A     = P.A;
    int           sense  = 0;

    if (P.orig) {
        for (Index j=0; j<n; ++j) {
            for (Index i=lo; i<hi; ++i) {
                A[i*rs+j*cs] = P.orig[i*P.ors+j*P.ocs];
            }
        }
    }

    for (Index c=0; c<kmin; c+=ib) {
        const Index kb = std::min(ib, kmin-c);

        for (Index j=c; j<c+kb; ++j) {
            const int par = int(j & 1);

            // pivot candidate among the own rows (first maximum)
            const Index i0   = std::max(lo, j);
            Index       best = -1;
            double      bval = 0;
            if (i0<hi) {
                const double *a = &A[j*cs];
                best = i0;
                bval = std::fabs(a[i0*rs]);
                for (Index i=i0+1; i<hi; ++i) {
                    const double v = std::fabs(a[i*rs]);
                    if (v>bval) {
                        bval = v;
                        best = i;
                    }
                }
                double *cr = &P.candRow[(par*size+r)*ib];
                for (Index l=0; l<kb; ++l) {
                    cr[l] = A[best*rs+(c+l)*cs];
                }
            }
            P.candIdx[par*size+r] = best;
            P.candVal[par*size+r] = bval;
            if (lo<=j && j<hi) {
                double *tr = &P.topRow[par*ib];
                for (Index l=0; l<kb; ++l) {
                    tr[l] = A[j*rs+(c+l)*cs];
                }
            }

            P.barrier.wait(sense);

            // pivot: the first maximum over the members in row order
            int    ws = -1;
            double pv = -1;
            for (int s=0; s<size; ++s) {
                if (P.candIdx[par*size+s]>=0 && P.candVal[par*size+s]>pv) {
                    pv = P.candVal[par*size+s];
                    ws = s;
                }
            }
            const Index  p     = P.candIdx[par*size+ws];
            const double *prow = &P.candRow[(par*size+ws)*ib];
            const double pivot = prow[j-c];

            if (r==0) {
                P.piv[j] = p;
            }
            if (pivot!=0) {
                if (p!=j) {
                    if (lo<=j && j<hi) {
                        for (Index l=0; l<kb; ++l) {
                            A[j*rs+(c+l)*cs] = prow[l];
                        }
                    }
                    if (lo<=p && p<hi) {
                        const double *tr = &P.topRow[par*ib];
                        for (Index l=0; l<kb; ++l) {
                            A[p*rs+(c+l)*cs] = tr[l];
                        }
                    }
                }
                const Index s0 = std::max(lo, j+1);
                if (s0<hi) {
                    double *a = &A[j*cs];
                    if (std::fabs(pivot)>=sfmin) {
                        const double rcp = 1.0/pivot;
                        for (Index i=s0; i<hi; ++i) {
                            a[i*rs] *= rcp;
                        }
                    } else {
                        for (Index i=s0; i<hi; ++i) {
                            a[i*rs] /= pivot;
                        }
                    }
                    for (Index l=j+1; l<c+kb; ++l) {
                        const double u = prow[l-c];
                        if (u!=0) {
                            double *b = &A[l*cs];
                            for (Index i=s0; i<hi; ++i) {
                                b[i*rs] -= a[i*rs]*u;
                            }
                        }
                    }
                }
            } else if (r==0 && P.info==0) {
                P.info = j+1;
            }
        }

        P.barrier.wait(sense);

        // Interchanges of this block in the columns outside of it, and the
        // triangular solve for the columns right of it; columns split
        // among the members.
        const Index nOut = c + (n-c-kb);
        const Index per  = (nOut+size-1)/size;
        const Index q0   = std::min(nOut, r*per);
        const Index q1   = std::min(nOut, q0+per);
        for (Index q=q0; q<q1; ++q) {
            const Index col = (q<c) ? q : q+kb;
            double *a = &A[col*cs];
            for (Index j=c; j<c+kb; ++j) {
                const Index p = P.piv[j];
                if (p!=j) {
                    std::swap(a[j*rs], a[p*rs]);
                }
            }
            if (col>=c+kb) {
                for (Index j=c; j<c+kb; ++j) {
                    const double x = a[j*rs];
                    if (x!=0) {
                        const double *l = &A[j*cs];
                        for (Index i=j+1; i<c+kb; ++i) {
                            a[i*rs] -= l[i*rs]*x;
                        }
                    }
                }
            }
        }

        P.barrier.wait(sense);

        // update of the own rows right of the block
        const Index g0 = std::max(lo, c+kb);
        if (g0<hi && c+kb<n) {
            ulmBLAS::gemm(hi-g0, n-c-kb, kb, -1.0,
                          false, &A[g0*rs+c*cs], rs, cs,
                          false, &A[c*rs+(c+kb)*cs], rs, cs,
                          1.0, &A[g0*rs+(c+kb)*cs], rs, cs);
        }
    }

    if (P.orig) {
        // the swaps of the last block wrote rows of other members
        P.barrier.wait(sense);
        for (Index j=0; j<n; ++j) {
            for (Index i=lo; i<hi; ++i) {
                P.orig[i*P.ors+j*P.ocs] = A[i*rs+j*cs];
            }
        }
    }
}

void
member(Panel &P, int r)
{
    if (P.rs==1) {
        member<true>(P, r);
    } else {
        member<false>(P, r);
    }
}

//
//  Factor the panel (m x n, row stride rs, column stride cs).
//  piv: 0-based, relative to the panel.  Returns 0 or the first zero pivot.
//
Index
panel(Index m, Index n, double *A, Index rs, Index cs, Index *piv,
      Index ib, int maxTeam, Index minRows, double joinWait, int excluded)
{
    int maxSize = (int) std::min<Index>(maxTeam, std::max<Index>(1, m/minRows));
    if (maxSize<=1 || n<=1) {
        statPanels.fetch_add(1, std::memory_order_relaxed);
        return getrf_packed(m, n, A, rs, cs, piv);
    }

    Panel P;
    P.m        = m;
    P.n        = n;
    P.A        = A;
    P.rs       = rs;
    P.cs       = cs;
    P.orig     = 0;
    P.ors      = 0;
    P.ocs      = 0;
    P.piv      = piv;
    P.ib       = std::min(ib, std::min(m, n));
    P.creator  = omp_get_thread_num();
    P.excluded = excluded;
    P.maxSize  = maxSize;
    P.state.store(1);
    P.started.store(0);
    P.size     = 1;
    P.info     = 0;

    Panel *pp = &P;
    for (int h=1; h<maxSize; ++h) {
        #pragma omp task firstprivate(pp) priority(PRIO_PANEL)
        {
            const int t = omp_get_thread_num();
            if (t!=pp->creator && t!=pp->excluded) {
                const int r = join(*pp);
                if (r>0) {
                    // wait until the team is closed and set up
                    while (pp->started.load(std::memory_order_acquire)==0) {
                        cpuRelax();
                    }
                    member(*pp, r);
                }
            }
        }
    }

    // wait for helpers (at most joinWait seconds), then close the team
    const double t0 = seconds();
    while ((P.state.load(std::memory_order_acquire) & ~CLOSED)<maxSize
           && seconds()-t0<joinWait)
    {
        cpuRelax();
    }
    const int size = P.state.fetch_or(CLOSED, std::memory_order_acq_rel)
                   & ~CLOSED;
    statPanels.fetch_add(1, std::memory_order_relaxed);
    if (size>1) {
        statTeams.fetch_add(1, std::memory_order_relaxed);
        statMembers.fetch_add(size, std::memory_order_relaxed);
    }

    //  copy of the panel for a team (see below); it must outlive the
    //  members, which finish before the taskwait
    std::vector<double> packed;
    Index info;
    if (size==1) {
        P.started.store(1, std::memory_order_release);
        info = getrf_packed(m, n, A, rs, cs, piv);
    } else {
        Index chunk = (m+size-1)/size;
        chunk = (chunk+7)/8*8;
        P.size  = size;
        P.chunk = chunk;
        //  neither stride 1: the team works on a column-major copy of the
        //  panel; every member copies its own rows in and back
        if (rs!=1 && cs!=1) {
            packed.resize(std::size_t(m)*n);
            P.orig = A;
            P.ors  = rs;
            P.ocs  = cs;
            P.A    = packed.data();
            P.rs   = 1;
            P.cs   = m;
        }
        P.barrier.reset(size);
        P.candVal.assign(2*size, 0.0);
        P.candIdx.assign(2*size, -1);
        P.candRow.assign(2*size*P.ib, 0.0);
        P.topRow.assign(2*P.ib, 0.0);
        P.started.store(1, std::memory_order_release);

        member(P, 0);
        info = P.info;
    }
    #pragma omp taskwait
    return info;
}

Index
getrf_tasks(Index m, Index n, double *A, Index rs, Index cs, const Source *src,
            Index *piv, Index nb, Index ib, int threads, int panelThreads,
            Index minRows, double joinWait, bool empty)
{
    const Index mt   = (m+nb-1)/nb;
    const Index nt   = (n+nb-1)/nb;
    const Index kmin = std::min(m, n);
    const Index kt   = (kmin+nb-1)/nb;

    std::vector<char>   tok(mt*nt);
    std::vector<Index>  pinfo(kt, 0);
    char                *T  = tok.data();
    (void) T;           // only used in depend clauses (GCC warns)
    Index               *PI = pinfo.data();

    // the generating thread does not join panel teams while it creates tasks
    std::atomic<int>    generator(-1);

    CHECKDEPS_LOG(log, std::size_t(kt)*(kt+nt+mt*nt)+kt+2*mt*nt);

    CHECKDEPS_TIME(tStart);
    CHECKDEPS_TIME(tGenEnd);
    #pragma omp parallel num_threads(threads) shared(generator)
    #pragma omp single
    {
        generator.store(omp_get_thread_num());

        //  working on a copy (src): one copy task per tile first
        if (src) {
            for (Index j=0; j<nt; ++j) {
                for (Index i=0; i<mt; ++i) {
                    const Index  mi  = std::min(nb, m-i*nb);
                    const Index  nj  = std::min(nb, n-j*nb);
                    double       *Wij = &A[i*nb*rs+j*nb*cs];
                    const double *Sij = &src->A[i*nb*src->rs+j*nb*src->cs];

                    CHECKDEPS_CREATE(rc, log, "COPYIN", -1, i, j);
                    CHECKDEPS_ACCESS(rc, i*nt+j, true);
                    #pragma omp task depend(out: T[i*nt+j]) \
                                     priority(j<=1 ? PRIO_NEXT : PRIO_UPD)
                    {
                        taskEnter();
                        CHECKDEPS_START(rc);
                        if (!empty) {
                            copyBlock(mi, nj, Sij, src->rs, src->cs,
                                      Wij, rs, cs, false);
                        }
                        CHECKDEPS_END(rc);
                        taskLeave();
                    }
                }
            }
        }

        for (Index k=0; k<kt; ++k) {
            const Index r0 = k*nb;
            const Index mp = m-r0;
            const Index wp = std::min(nb, n-r0);
            const Index kp = std::min(mp, wp);
            double      *Akk = &A[r0*(rs+cs)];
            Index       *pk  = &piv[r0];

            CHECKDEPS_CREATE(rp, log, "PANEL", k, k, k);
            for (Index ii=k; ii<mt; ++ii) {
                CHECKDEPS_ACCESS(rp, ii*nt+k, true);
            }
            #pragma omp task depend(iterator(ii=k:mt), inout: T[ii*nt+k]) \
                             priority(PRIO_PANEL)
            {
                taskEnter();
                CHECKDEPS_START(rp);
                if (!empty) {
                    PI[k] = panel(mp, wp, Akk, rs, cs, pk, ib, panelThreads,
                                  minRows, joinWait, generator.load());
                } else {
                    for (Index l=0; l<kp; ++l) {
                        pk[l] = l;
                    }
                }
                CHECKDEPS_END(rp);
                taskLeave();
            }

            // pivoting to the left
            for (Index c=0; c<k; ++c) {
                double *Ac = &A[r0*rs+c*nb*cs];

                CHECKDEPS_CREATE(rl, log, "LSWAP", k, k, c);
                CHECKDEPS_ACCESS(rl, k*nt+k, false);
                for (Index ii=k; ii<mt; ++ii) {
                    CHECKDEPS_ACCESS(rl, ii*nt+c, true);
                }
                #pragma omp task depend(in: T[k*nt+k]) \
                                 depend(iterator(ii=k:mt), inout: T[ii*nt+c]) \
                                 priority(PRIO_LSWAP)
                {
                    taskEnter();
                    CHECKDEPS_START(rl);
                    if (!empty) {
                        laswp(nb, Ac, rs, cs, 0, kp, pk);
                    }
                    CHECKDEPS_END(rl);
                    taskLeave();
                }
            }

            // trailing matrix
            for (Index j=k+1; j<nt; ++j) {
                const Index wj  = std::min(nb, n-j*nb);
                const int   pri = (j==k+1) ? PRIO_NEXT : PRIO_UPD;
                double      *Akj = &A[r0*rs+j*nb*cs];

                CHECKDEPS_CREATE(rsw, log, "SWAP_TRSM", k, k, j);
                CHECKDEPS_ACCESS(rsw, k*nt+k, false);
                for (Index ii=k; ii<mt; ++ii) {
                    CHECKDEPS_ACCESS(rsw, ii*nt+j, true);
                }
                #pragma omp task depend(in: T[k*nt+k]) \
                                 depend(iterator(ii=k:mt), inout: T[ii*nt+j]) \
                                 priority(pri)
                {
                    taskEnter();
                    CHECKDEPS_START(rsw);
                    if (!empty) {
                        laswp(wj, Akj, rs, cs, 0, kp, pk);
                        ulmBLAS::trlsm(kp, wj, 1.0, false, true,
                                       Akk, rs, cs, Akj, rs, cs);
                    }
                    CHECKDEPS_END(rsw);
                    taskLeave();
                }
                for (Index i=k+1; i<mt; ++i) {
                    const Index  mi  = std::min(nb, m-i*nb);
                    const double *Aik = &A[i*nb*rs+r0*cs];
                    double       *Aij = &A[i*nb*rs+j*nb*cs];

                    CHECKDEPS_CREATE(rg, log, "GEMM", k, i, j);
                    CHECKDEPS_ACCESS(rg, i*nt+k, false);
                    CHECKDEPS_ACCESS(rg, k*nt+j, false);
                    CHECKDEPS_ACCESS(rg, i*nt+j, true);
#ifndef ULMLAPACK_LU_GEMM_2DEPS
                    #pragma omp task depend(in: T[i*nt+k], T[k*nt+j]) \
                                     depend(inout: T[i*nt+j]) \
                                     priority(pri)
#else
                    // U(k,j) is ordered through tile (i,j): SWAP_TRSM(k,j)
                    // writes all tiles (ii,j), ii >= k, and no later task
                    // writes U(k,j)
                    #pragma omp task depend(in: T[i*nt+k]) \
                                     depend(inout: T[i*nt+j]) \
                                     priority(pri)
#endif
                    {
                        taskEnter();
                        CHECKDEPS_START(rg);
                        if (!empty) {
                            ulmBLAS::gemm(mi, wj, kp, -1.0,
                                          false, Aik, rs, cs,
                                          false, Akj, rs, cs,
                                          1.0, Aij, rs, cs);
                        }
                        CHECKDEPS_END(rg);
                        taskLeave();
                    }
                }
            }
        }
        //  and one that copies it back after its last update
        if (src) {
            for (Index j=0; j<nt; ++j) {
                for (Index i=0; i<mt; ++i) {
                    const Index  mi  = std::min(nb, m-i*nb);
                    const Index  nj  = std::min(nb, n-j*nb);
                    const double *Wij = &A[i*nb*rs+j*nb*cs];
                    double       *Sij = &src->A[i*nb*src->rs+j*nb*src->cs];

                    CHECKDEPS_CREATE(rb, log, "COPYBACK", kt, i, j);
                    CHECKDEPS_ACCESS(rb, i*nt+j, false);
                    #pragma omp task depend(in: T[i*nt+j]) priority(PRIO_LSWAP)
                    {
                        taskEnter();
                        CHECKDEPS_START(rb);
                        if (!empty) {
                            copyBlock(mi, nj, Wij, rs, cs,
                                      Sij, src->rs, src->cs, false);
                        }
                        CHECKDEPS_END(rb);
                        taskLeave();
                    }
                }
            }
        }
        generator.store(-1);
        CHECKDEPS_TIME_SET(tGenEnd);
    }
    CHECKDEPS_TIME(tEnd);
    CHECKDEPS_PROFILE(log, threads, tStart, tGenEnd, tEnd);
    if (CHECKDEPS_CHECK(log, "dgetrf")) {
        std::abort();
    }

    Index info = 0;
    for (Index k=0; k<kt; ++k) {
        if (PI[k]) {
            info = k*nb+PI[k];
            break;
        }
    }
    for (Index k=0; k<kt; ++k) {
        const Index r0 = k*nb;
        const Index kp = std::min(m-r0, std::min(nb, n-r0));
        for (Index l=0; l<kp; ++l) {
            piv[r0+l] += r0;
        }
    }
    return info;
}

} // namespace

} // namespace ulmlapack

namespace ulmlapack {

Index
getrf(Index m, Index n, double *A, Index incRowA, Index incColA,
      Index *piv, Index incPiv, const ulmlapack_param *param)
{
    if (m<=0 || n<=0) {
        return 0;
    }
    ulmlapack_param p;
    if (param) {
        p = *param;
    } else {
        ulmlapack_param_init(&p);
    }
    ulmlapack_dgetrf_resolve(int(m), int(n), &p);

    const Index kmin = std::min(m, n);
    std::vector<Index> pv(kmin);
    Index info;

    if (kmin<=p.serial_n || p.threads==1 || kmin<=p.nb) {
        info = getrf_packed(m, n, A, incRowA, incColA, pv.data());
    } else {
        info = getrf_tasks(m, n, A, incRowA, incColA, 0, pv.data(),
                           p.nb, p.ib, p.threads, p.panel_threads,
                           /*minRows*/ 64, /*joinWait*/ 20e-6,
                           p.empty_tasks!=0);
    }
    for (Index i=0; i<kmin; ++i) {
        piv[i*incPiv] = pv[i];
    }
    return info;
}

Index
getrf_colmajor(Index m, Index n, double *A, Index incRowA, Index incColA,
               Index *piv, Index incPiv, const ulmlapack_param *param)
{
    if (m<=0 || n<=0) {
        return 0;
    }
    ulmlapack_param p;
    if (param) {
        p = *param;
    } else {
        ulmlapack_param_init(&p);
    }
    ulmlapack_dgetrf_resolve(int(m), int(n), &p);

    const Index kmin = std::min(m, n);
    if (kmin<=p.serial_n || p.threads==1 || kmin<=p.nb) {
        return getrf(m, n, A, incRowA, incColA, piv, incPiv, &p);
    }
    std::vector<Index> pv(kmin);
    double       *B  = workspace(std::size_t(m)*n);
    const Source src = {A, incRowA, incColA};
    const Index info = getrf_tasks(m, n, B, 1, m, &src, pv.data(),
                                   p.nb, p.ib, p.threads, p.panel_threads,
                                   /*minRows*/ 64, /*joinWait*/ 20e-6,
                                   p.empty_tasks!=0);
    for (Index i=0; i<kmin; ++i) {
        piv[i*incPiv] = pv[i];
    }
    return info;
}

} // namespace ulmlapack

using ulmlapack::Index;

extern "C" {

void
ulmlapack_dgetrf_resolve(int m, int n, ulmlapack_param *p)
{
    // Defaults tuned on one socket of a Xeon Platinum 8358 (Ice Lake, 32
    // cores) with sizes off the benchmark grid; see README (Tuning).
    const int kmin = std::min(m, n);
    if (p->threads<=0) {
        p->threads = omp_get_max_threads();
    }
    if (p->nb<=0) {
        p->nb = (kmin<600) ? 48 : (kmin<900) ? 64 : (kmin<2800) ? 96 : 128;
    }
    if (p->ib<=0) {
        p->ib = 16;
    }
    if (p->panel_threads<=0) {
        p->panel_threads = 8;
    }
    if (p->serial_n<0) {
        p->serial_n = 150;
    }
    // about one thread per 64 rows (at least 4)
    p->threads = std::max(1, std::min(p->threads, std::max(4, kmin/64)));
    const Index mt = (m+p->nb-1)/p->nb;
    const Index nt = (n+p->nb-1)/p->nb;
    p->threads = (int) std::max<Index>(1, std::min<Index>(p->threads, mt*nt));
    p->panel_threads = std::min(p->panel_threads, p->threads);
}

int
ulmlapack_dgetrf_param(int m, int n, double *A, int ldA, int *ipiv,
                       const ulmlapack_param *param)
{
    if (m<0) {
        return -1;
    }
    if (n<0) {
        return -2;
    }
    if (ldA<std::max(1, m)) {
        return -4;
    }
    const Index kmin = std::min(m, n);
    std::vector<Index> piv(kmin);
    const Index info = ulmlapack::getrf(m, n, A, Index(1), Index(ldA),
                                        piv.data(), Index(1), param);
    for (Index i=0; i<kmin; ++i) {
        ipiv[i] = int(piv[i]+1);
    }
    return (int) info;
}

#ifdef ULMLAPACK_CHECK_DEPS
void
ulmlapack_last_profile(long *tasks, int *threads, double *generate,
                       double *wall, double *busy)
{
    const ulmlapack::checkdeps::Profile &p
        = ulmlapack::checkdeps::lastProfile();
    *tasks = p.tasks; *threads = p.threads;
    *generate = p.generate; *wall = p.wall; *busy = p.busy;
}
#endif

void
ulmlapack_dgetrf_stats(long *panels, long *teams, long *members)
{
    using namespace ulmlapack;
    *panels  = statPanels.exchange(0);
    *teams   = statTeams.exchange(0);
    *members = statMembers.exchange(0);
}

int
ulmlapack_dgetrf(int m, int n, double *A, int ldA, int *ipiv)
{
    return ulmlapack_dgetrf_param(m, n, A, ldA, ipiv, 0);
}

} // extern "C"
