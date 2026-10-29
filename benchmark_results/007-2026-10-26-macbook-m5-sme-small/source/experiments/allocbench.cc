//
//  allocbench: cost of the packing buffers of one DGEMM call of ulmBLAS
//  (master: gemm takes MC*KC+MR and KC*NC+NR doubles = 3 MiB + 8 MiB from a
//  MemoryPool and releases them at the end of the call), for different
//  allocation strategies.  One executable per strategy (-DSTRATEGY=...):
//
//      POOL        ulmBLAS::MemoryPool of master (mutex, hash tables, list)
//      CACHE       MemoryPool with a per-thread cache (draft)
//      MALLOC      malloc/free of the same fixed sizes
//      NEW         new double[] / delete [] of the same fixed sizes
//      ALIGNED     aligned_alloc(128) / free of the same fixed sizes
//      EXACT       malloc/free of the sizes the problem needs
//      TLS         thread_local buffers of the fixed sizes, allocated once
//
//  A "call" allocates both buffers, writes TOUCH_A and TOUCH_B bytes of
//  them (as packing does: the panels of the problem), reads them back and
//  releases them.
//
//      allocbench SCENARIO [CALLS]
//          SCENARIO small16: touch 32 x 16 + 16 x 16 doubles (n = 16)
//                   small64: touch 64 x 64 + 64 x 64 doubles (n = 64)
//                   large:   touch the full 3 MiB + 8 MiB (n >= 1024)
//
//  Prints the time of the first call (cold, fresh process) and the median
//  of the warm calls, in ns.
//
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#if defined(STRATEGY_POOL) || defined(STRATEGY_CACHE)
#   include <ulmblas/auxiliary/memorypool.h>
#endif

static double
now()
{
    timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

constexpr std::size_t MC = 384, KC = 1024, NC = 1024, MR = 32, NR = 16;
constexpr std::size_t NA = MC*KC + MR, NB = KC*NC + NR;

static double *
get(std::size_t n, std::size_t exact, int which)
{
#if defined(STRATEGY_POOL) || defined(STRATEGY_CACHE)
    (void)n; (void)exact; (void)which;
    return nullptr;     // not used (GET)
#elif defined(STRATEGY_MALLOC)
    (void)exact; (void)which;
    return static_cast<double *>(std::malloc(n*sizeof(double)));
#elif defined(STRATEGY_NEW)
    (void)exact; (void)which;
    return new double[n];
#elif defined(STRATEGY_ALIGNED)
    (void)exact; (void)which;
    std::size_t bytes = (n*sizeof(double) + 127) / 128 * 128;
    return static_cast<double *>(std::aligned_alloc(128, bytes));
#elif defined(STRATEGY_EXACT)
    (void)n; (void)which;
    return static_cast<double *>(std::malloc(exact*sizeof(double)));
#elif defined(STRATEGY_TLS)
    (void)exact;
    thread_local std::vector<double> a(NA), b(NB);
    return which==0 ? a.data() : b.data();
#endif
}

static void
put(double *p)
{
#if defined(STRATEGY_POOL) || defined(STRATEGY_CACHE)
    (void)p;            // not used (PUT)
#elif defined(STRATEGY_MALLOC) || defined(STRATEGY_ALIGNED) \
   || defined(STRATEGY_EXACT)
    std::free(p);
#elif defined(STRATEGY_NEW)
    delete [] p;
#elif defined(STRATEGY_TLS)
    (void)p;
#endif
}

#if defined(STRATEGY_POOL) || defined(STRATEGY_CACHE)
//  allocate and release must use the same pool object
static ulmBLAS::MemoryPool<double> &
thePool()
{
    static ulmBLAS::MemoryPool<double> pool;
    return pool;
}
#   define GET(n, e, w) thePool().allocate(n)
#   define PUT(p)       thePool().release(p)
#else
#   define GET(n, e, w) get(n, e, w)
#   define PUT(p)       put(p)
#endif

static double sink = 0;

//  packing writes the panels, the kernel reads them (not inlined: the same
//  machine code for all strategies)
__attribute__((noinline)) static void
touch(double *X, std::size_t n)
{
    std::memset(X, 0x11, n*sizeof(double));
    double s = 0;
    for (std::size_t i=0; i<n; i+=8) s += X[i];
    sink += s;
}

static void
call(std::size_t touchA, std::size_t touchB)
{
    double *A = GET(NA, touchA, 0);
    double *B = GET(NB, touchB, 1);
    touch(A, touchA);
    touch(B, touchB);
    PUT(B);
    PUT(A);
}

int
main(int argc, char **argv)
{
    const char *scenario = argc>1 ? argv[1] : "small16";
    long calls = argc>2 ? std::atol(argv[2]) : 20000;
    std::size_t ta, tb;
    if (!std::strcmp(scenario, "none")) {
        ta = 1; tb = 1;
    } else if (!std::strcmp(scenario, "small16")) {
        ta = 32*16; tb = 16*16;
    } else if (!std::strcmp(scenario, "small64")) {
        ta = 64*64; tb = 64*64;
    } else {
        ta = NA; tb = NB; calls = argc>2 ? calls : 200;
    }
    double t0 = now();
    call(ta, tb);
    double cold = now() - t0;

    std::vector<double> t;
    for (int r=0; r<9; ++r) {
        double t1 = now();
        for (long c=0; c<calls; ++c) {
            call(ta, tb);
        }
        t.push_back((now() - t1)/calls);
    }
    std::sort(t.begin(), t.end());
    std::printf("%-8s %-8s cold %10.0f ns  warm median %10.1f ns  (min %.1f, max %.1f)\n",
                STRATEGY_NAME, scenario, 1e9*cold, 1e9*t[4], 1e9*t[0], 1e9*t[8]);
    return sink==-1;
}
