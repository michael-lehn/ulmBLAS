//
//  prof-hsw: prof-ob.cc of campaign 016-2026-10-27-uc3-icelake-openblas-kernel-prof
//  for OpenBLAS's HASWELL kernel (compile with
//  -DULMBLAS_DGEMM_UKERNEL_openblas_haswell_4x8).  Modes in addition to prof.cc:
//    macro        as inst, but the macro kernel of ulmBLAS (mgemm: one call
//                 of the kernel per panel of B, n = NR = 8) is replaced by one
//                 call of OpenBLAS's kernel function for the whole packed
//                 mc x nc block (as OpenBLAS's driver calls it; the kernel then
//                 uses its 4 x 12 path for groups of 24 columns)
//    macronopack  the same without the packing calls (as nopack)
//  Only blocks with mc % 4 == 0 and nc % 8 == 0 are passed to OpenBLAS's
//  kernel as a whole (the packed layout of full panels); other blocks use
//  mgemm.  The result is checked by mode macrocheck.  Changes against
//  prof-ob.cc: this comment and the kernel called in mode macro.
//
//  prof: breakdown of the time of ulmBLAS DGEMM (C <- C + A*B, column major)
//  into packing, micro kernel and other overhead.
//
//      prof [-o ulm|blis] [-b L2,KC,L3] [-m MODES] m n k
//
//  The driver is a copy of ulmBLAS::gemm (ulmblas/level3/gemm.tcc) with
//  runtime block sizes and timers around the packing and macro kernel calls;
//  it calls the packing routines, the macro kernel and the micro kernel of
//  ulmBLAS.
//
//  Loop orders (-o):
//    ulm   ulmBLAS: jc over n (L3 = NC), pc over k (KC), pack B (kc x nc),
//          ic over m (L2 = MC), pack A (mc x kc), macro kernel with the B
//          panels outer and the A panels inner (mgemm).
//    blis  BLIS for column major C with the row preferential kernel (C^T =
//          B^T A^T): ic over m (L3 = 8184), pc over k (KC), pack A (mc x kc),
//          jc over n (L2 = 256), pack B (kc x nc), macro kernel with the A
//          panels outer and the B panels inner.
//  -b gives the block sizes L2,KC,L3 (default 960,1024,4080 for ulm and
//  256,3072,8184 for blis, the block sizes of ulmBLAS and of BLIS 2.0).
//  -p selects the packing routines of the driver (packvar.h; default
//  ulm:ulm, the routines of ulmBLAS).
//
//  Modes (-m, comma separated, default lib,inst,nopack,pack,kern):
//    lib     ulmBLAS::gemm (the library, compiled into this program)
//    inst    the instrumented driver: total time and time in pack B, pack A
//            and the macro kernel
//    nopack  the instrumented driver without the calls of the packing
//            routines (the buffers are packed once before; the result is
//            wrong, the flops are the same)
//    pack    only the calls of the packing routines
//    kern    the micro kernel alone on L1/L2 resident packed panels (as
//            ukernel.cc) with the kc of the driver
//    pool    the calls of the memory pool of ulmBLAS::gemm alone (allocate
//            and release of the buffers of A and B; GFLOPS refers to the
//            flops of the problem)
//    cblas   cblas_dgemm of the library linked to the program (compiled
//            with -DWITH_CBLAS, e.g. BLIS or OpenBLAS)
//
//  Each mode: warm-up of at least 0.2 s, then S samples (default 5, -s),
//  each sample repeats the call until at least 0.1 s elapsed (-t); median.
//  Output (one line per mode):
//      mode order L2 KC L3 m n k GFLOPS t_call[s] packB packA mgemm other PA:PB
//  where packB, packA, mgemm and other are fractions of the time of a call
//  (inst only; for nopack mgemm only).
//
//  Mode check (not timed): computes C + A*B with ulmBLAS::gemm and with the
//  driver (order, block sizes and packing variants as given) and prints the
//  largest difference relative to the largest element of |C| + |A|*|B|;
//  exit status 2 if it exceeds 1e-14 * k.
//
#if defined(__APPLE__)
#include <pthread/qos.h>
#include <mach/mach_time.h>
#endif
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>
#include <ulmblas/level3/level3.h>
#include "packvar.h"
#if defined(WITH_CBLAS)
#include <cblas.h>
#endif

typedef long Index;

static double
now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

#if defined(__APPLE__)
static double tick;
static inline unsigned long long ticks() { return mach_absolute_time(); }
#else
static double tick = 1e-9;
static inline unsigned long long
ticks()
{
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
           .count();
}
#endif

static const Index MR = ulmBLAS::BlockSizeUGemm<double>::MR;
static const Index NR = ulmBLAS::BlockSizeUGemm<double>::NR;

struct Timers { unsigned long long packA, packB, mgemm; };

//  64 byte aligned vectors for the packed buffers (the AVX micro kernel
//  needs 32 byte alignment, as provided by the memory pool of ulmBLAS)
template <typename T>
struct AlignedAlloc
{
    typedef T value_type;
    AlignedAlloc() {}
    template <typename U> AlignedAlloc(const AlignedAlloc<U> &) {}
    T *
    allocate(size_t n)
    {
        void *p;
        if (posix_memalign(&p, 64, n*sizeof(T))) throw std::bad_alloc();
        return static_cast<T *>(p);
    }
    void deallocate(T *p, size_t) { std::free(p); }
};
template <typename T, typename U>
bool operator==(const AlignedAlloc<T> &, const AlignedAlloc<U> &) { return true; }
template <typename T, typename U>
bool operator!=(const AlignedAlloc<T> &, const AlignedAlloc<U> &) { return false; }
typedef std::vector<double, AlignedAlloc<double> > Buffer;

//  macro kernel with the A panels outer and the B panels inner (BLIS order)
static void
mgemm_aouter(Index mc, Index nc, Index kc, const double *A_, const double *B_,
             double *C, Index incRowC, Index incColC)
{
    const Index mp = (mc+MR-1)/MR, np = (nc+NR-1)/NR;
    const Index mr_ = mc%MR, nr_ = nc%NR;

    for (Index i=0; i<mp; ++i) {
        Index mr = (i!=mp-1 || mr_==0) ? MR : mr_;
        for (Index j=0; j<np; ++j) {
            Index nr = (j!=np-1 || nr_==0) ? NR : nr_;
            const double *nextA = &A_[i*kc*MR], *nextB = &B_[(j+1)*kc*NR];
            if (j==np-1) {
                nextB = B_;
                nextA = (i==mp-1) ? A_ : &A_[(i+1)*kc*MR];
            }
            double *C_ = &C[i*MR*incRowC+j*NR*incColC];
            if (mr==MR && nr==NR) {
                ulmBLAS::ugemm(kc, 1.0, &A_[i*kc*MR], &B_[j*kc*NR], 1.0,
                               C_, incRowC, incColC, nextA, nextB);
            } else {
                ulmBLAS::ugemm(mr, nr, kc, 1.0, &A_[i*kc*MR], &B_[j*kc*NR],
                               1.0, C_, incRowC, incColC, nextA, nextB);
            }
        }
    }
}

struct Problem
{
    Index m, n, k, L2, KC, L3;
    bool blis;
    packvar::PackA packA;
    packvar::PackB packB;
    std::vector<double> A, B, C;
    Buffer A_, B_;
    bool macro = false;

    //  pack: call the packing routines, compute: call the macro kernel
    void
    run(bool pack, bool compute, Timers *t)
    {
        unsigned long long t0 = 0;
        if (!blis) {
            const Index MC = L2, NC = L3;
            for (Index j0=0; j0<n; j0+=NC) {
                Index nc = std::min(NC, n-j0);
                for (Index l0=0; l0<k; l0+=KC) {
                    Index kc = std::min(KC, k-l0);
                    if (pack) {
                        if (t) t0 = ticks();
                        packB(kc, nc, &B[l0+j0*k], 1L, k, &B_[0]);
                        if (t) t->packB += ticks()-t0;
                    }
                    for (Index i0=0; i0<m; i0+=MC) {
                        Index mc = std::min(MC, m-i0);
                        if (pack) {
                            if (t) t0 = ticks();
                            packA(mc, kc, &A[i0+l0*m], 1L, m, &A_[0]);
                            if (t) t->packA += ticks()-t0;
                        }
                        if (compute) {
                            if (t) t0 = ticks();
                            if (macro && mc%MR==0 && nc%NR==0) {
                                ulmBLAS::openblas::kernel::haswell_4x8::
                                dgemm_kernel(mc, nc, kc, 1.0,
                                             &A_[0], &B_[0],
                                             &C[i0+j0*m], m);
                            } else {
                                ulmBLAS::mgemm(mc, nc, kc, 1.0, &A_[0], &B_[0],
                                               1.0, &C[i0+j0*m], 1L, m);
                            }
                            if (t) t->mgemm += ticks()-t0;
                        }
                    }
                }
            }
        } else {
            const Index NC = L2, MC = L3;
            for (Index i0=0; i0<m; i0+=MC) {
                Index mc = std::min(MC, m-i0);
                for (Index l0=0; l0<k; l0+=KC) {
                    Index kc = std::min(KC, k-l0);
                    if (pack) {
                        if (t) t0 = ticks();
                        packA(mc, kc, &A[i0+l0*m], 1L, m, &A_[0]);
                        if (t) t->packA += ticks()-t0;
                    }
                    for (Index j0=0; j0<n; j0+=NC) {
                        Index nc = std::min(NC, n-j0);
                        if (pack) {
                            if (t) t0 = ticks();
                            packB(kc, nc, &B[l0+j0*k], 1L, k, &B_[0]);
                            if (t) t->packB += ticks()-t0;
                        }
                        if (compute) {
                            if (t) t0 = ticks();
                            mgemm_aouter(mc, nc, kc, &A_[0], &B_[0],
                                         &C[i0+j0*m], 1L, m);
                            if (t) t->mgemm += ticks()-t0;
                        }
                    }
                }
            }
        }
    }
};

int
main(int argc, char **argv)
{
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    tick = 1e-9*tb.numer/tb.denom;
#endif
    std::string order = "ulm", pack = "ulm:ulm", modes = "lib,inst,nopack,pack,kern";
    Index b[3] = {0, 0, 0};
    int samples = 5;
    double tsample = 0.1;
    int a = 1;
    for (; a<argc && argv[a][0]=='-'; a+=2) {
        if (a+1>=argc) return 1;
        if (!std::strcmp(argv[a], "-o")) order = argv[a+1];
        else if (!std::strcmp(argv[a], "-b"))
            std::sscanf(argv[a+1], "%ld,%ld,%ld", &b[0], &b[1], &b[2]);
        else if (!std::strcmp(argv[a], "-m")) modes = argv[a+1];
        else if (!std::strcmp(argv[a], "-p")) pack = argv[a+1];
        else if (!std::strcmp(argv[a], "-s")) samples = std::atoi(argv[a+1]);
        else if (!std::strcmp(argv[a], "-t")) tsample = std::atof(argv[a+1]);
        else return 1;
    }
    if (argc-a!=3) {
        std::fprintf(stderr, "usage: prof [-o ulm|blis] [-b L2,KC,L3] "
                     "[-m MODES] [-s S] [-t T] m n k\n");
        return 1;
    }
    Problem p;
    p.m = std::atol(argv[a]); p.n = std::atol(argv[a+1]);
    p.k = std::atol(argv[a+2]);
    p.blis = order=="blis";
    {
        std::string::size_type c = pack.find(':');
        p.packA = packvar::selectA(pack.substr(0, c).c_str());
        p.packB = packvar::selectB(c==std::string::npos ? "ulm"
                                   : pack.substr(c+1).c_str());
        if (!p.packA || !p.packB) {
            std::fprintf(stderr, "unknown packing variant %s\n",
                         pack.c_str());
            return 1;
        }
    }
    if (!b[0]) {
        if (p.blis) { b[0] = 256; b[1] = 3072; b[2] = 8184; }
        else        { b[0] = 960; b[1] = 1024; b[2] = 4080; }
    }
    p.L2 = b[0]; p.KC = b[1]; p.L3 = b[2];
    Index MCb = p.blis ? p.L3 : p.L2, NCb = p.blis ? p.L2 : p.L3;
    MCb = std::min(MCb, p.m); NCb = std::min(NCb, p.n);
    Index KCb = std::min(p.KC, p.k);
    p.A.resize(p.m*p.k); p.B.resize(p.k*p.n); p.C.resize(p.m*p.n);
    p.A_.resize(((MCb+MR-1)/MR)*MR*KCb+MR);
    p.B_.resize(((NCb+NR-1)/NR)*NR*KCb+NR);
    std::srand(1);
    for (auto &x : p.A) x = double(std::rand())/RAND_MAX - 0.5;
    for (auto &x : p.B) x = double(std::rand())/RAND_MAX - 0.5;
    for (auto &x : p.C) x = double(std::rand())/RAND_MAX - 0.5;
    const double flops = 2.0*p.m*p.n*p.k;

    auto measure = [&](const char *mode, auto call, double work, bool timed) {
        Timers acc = {0, 0, 0}, t;
        double t0 = now(), t1;
        long   reps = 0;
        do { call(nullptr); ++reps; t1 = now(); } while (t1-t0 < 0.2);
        reps = long(reps*tsample/(t1-t0)) + 1;
        std::vector<double> g;
        std::vector<Timers> ts;
        for (int s=0; s<samples; ++s) {
            t = {0, 0, 0};
            t0 = now();
            for (long r=0; r<reps; ++r) call(timed ? &t : nullptr);
            t1 = now();
            g.push_back((t1-t0)/reps);
            ts.push_back(t);
        }
        //  median sample (the timers of the same sample)
        std::vector<int> idx(samples);
        for (int s=0; s<samples; ++s) idx[s] = s;
        std::sort(idx.begin(), idx.end(),
                  [&](int x, int y) { return g[x]<g[y]; });
        int med = idx[samples/2];
        double tc = g[med];
        acc = ts[med];
        double pb = acc.packB*tick/reps/tc, pa = acc.packA*tick/reps/tc,
               mg = acc.mgemm*tick/reps/tc;
        std::printf("%-6s %-4s %4ld %4ld %4ld %5ld %5ld %5ld %8.3f %11.4e "
                    "%6.4f %6.4f %6.4f %6.4f %s\n",
                    mode, order.c_str(), p.L2, p.KC, p.L3, p.m, p.n, p.k,
                    work/tc*1e-9, tc, pb, pa, mg,
                    timed ? 1-pb-pa-mg : 0.0, pack.c_str());
        std::fflush(stdout);
    };

    std::string ms = modes + ",";
    for (size_t pos; (pos = ms.find(','))!=std::string::npos;
         ms.erase(0, pos+1)) {
        std::string mode = ms.substr(0, pos);
        if (mode=="check" || mode=="macrocheck") {
            p.macro = (mode=="macrocheck");
            std::vector<double> C0 = p.C, C1, W(p.m*p.n, 0.0);
            ulmBLAS::gemm(p.m, p.n, p.k, 1.0, false, &p.A[0], 1L, p.m,
                          false, &p.B[0], 1L, p.k, 1.0, &p.C[0], 1L, p.m);
            C1 = p.C;
            p.C = C0;
            p.run(true, true, nullptr);
            double err = 0, scale = 0;
            for (Index j=0; j<p.n; ++j) {
                for (Index i=0; i<p.m; ++i) {
                    double w = std::abs(C0[i+j*p.m]);
                    for (Index l=0; l<p.k; ++l)
                        w += std::abs(p.A[i+l*p.m]*p.B[l+j*p.k]);
                    scale = std::max(scale, w);
                    err = std::max(err, std::abs(p.C[i+j*p.m]-C1[i+j*p.m]));
                }
            }
            double rel = scale>0 ? err/scale : err;
            bool ok = rel <= 1e-14*std::max(p.k, Index(1));
            p.macro = false;
            std::printf("%-6s %-4s %4ld %4ld %4ld %5ld %5ld %5ld %s %.3e "
                        "%s\n", mode=="check" ? "check" : "mcheck", order.c_str(), p.L2, p.KC, p.L3, p.m, p.n,
                        p.k, pack.c_str(), rel, ok ? "ok" : "FAILED");
            if (!ok) return 2;
        } else if (mode=="lib") {
            measure("lib", [&](Timers *) {
                ulmBLAS::gemm(p.m, p.n, p.k, 1.0, false, &p.A[0], 1L, p.m,
                              false, &p.B[0], 1L, p.k, 1.0, &p.C[0], 1L, p.m);
            }, flops, false);
#if defined(WITH_CBLAS)
        } else if (mode=="cblas") {
            measure("cblas", [&](Timers *) {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            int(p.m), int(p.n), int(p.k), 1.0, &p.A[0],
                            int(p.m), &p.B[0], int(p.k), 1.0, &p.C[0],
                            int(p.m));
            }, flops, false);
#endif
        } else if (mode=="pool") {
            static ulmBLAS::MemoryPool<double> pool;
            const Index MC = ulmBLAS::BlockSize<double>::MC,
                        KC = ulmBLAS::BlockSize<double>::KC,
                        NC = ulmBLAS::BlockSize<double>::NC;
            measure("pool", [&](Timers *) {
                double *a = pool.allocate(MC*KC+MR);
                double *b = pool.allocate(KC*NC+NR);
                asm volatile("" : : "r"(a), "r"(b) : "memory");
                pool.release(a);
                pool.release(b);
            }, flops, false);
        } else if (mode=="inst") {
            measure("inst", [&](Timers *t) { p.run(true, true, t); },
                    flops, true);
        } else if (mode=="macro") {
            p.macro = true;
            measure("macro", [&](Timers *t) { p.run(true, true, t); },
                    flops, true);
            p.macro = false;
        } else if (mode=="macronopack") {
            p.macro = true;
            p.run(true, false, nullptr);
            measure("macronopack", [&](Timers *t) { p.run(false, true, t); },
                    flops, true);
            p.macro = false;
        } else if (mode=="nopack") {
            p.run(true, false, nullptr);
            measure("nopack", [&](Timers *t) { p.run(false, true, t); },
                    flops, true);
        } else if (mode=="pack") {
            measure("pack", [&](Timers *t) { p.run(true, false, t); },
                    flops, true);
        } else if (mode=="kern") {
            //  micro kernel on 8 A panels x 4 B panels (as ukernel.cc)
            const Index NP = 8, NQ = 4, kc = KCb;
            Buffer A(NP*MR*kc+MR), B(NQ*NR*kc+NR);
            std::vector<double> C(MR*NR*NP*NQ);
            for (auto &x : A) x = double(std::rand())/RAND_MAX - 0.5;
            for (auto &x : B) x = double(std::rand())/RAND_MAX - 0.5;
            measure("kern", [&](Timers *) {
                for (Index q=0; q<NQ; ++q)
                    for (Index i=0; i<NP; ++i)
                        ulmBLAS::ugemm(kc, 1.0, &A[i*MR*kc], &B[q*NR*kc],
                                       1.0, &C[(q*NP+i)*MR*NR], 1L, MR,
                                       &A[((i+1)%NP)*MR*kc], &B[q*NR*kc]);
            }, 2.0*MR*NR*kc*NP*NQ, false);
        } else {
            std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
            return 1;
        }
    }
    return p.C[0]==12345.0;
}
