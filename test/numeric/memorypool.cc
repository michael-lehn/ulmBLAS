//
//  Tests of the buffers of the packing routines (MemoryPool): blocks in use
//  at the same time are distinct and do not overlap, released blocks are
//  reused, blocks of different sizes and of different pools are not mixed,
//  more blocks than the cache of a thread holds, release(nullptr), and a
//  block released by another thread than the one that allocated it.  The
//  alignment the micro kernels require (T0353).
//
#include <ulmblas/ulmblas.h>
#include <test/numeric/testing.h>

#include <complex>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

using namespace test;

//
//  Fill block b (n elements) with a pattern of its own and check it.
//
void
fill(double *b, std::size_t n, int id)
{
    for (std::size_t i=0; i<n; ++i) {
        b[i] = id*100000.0 + double(i);
    }
}

bool
intact(const double *b, std::size_t n, int id)
{
    for (std::size_t i=0; i<n; ++i) {
        if (b[i]!=id*100000.0 + double(i)) {
            return false;
        }
    }
    return true;
}

ulmBLAS::MemoryPool<double> &
pool(int p)
{
    static ulmBLAS::MemoryPool<double> pool0, pool1;
    return p==0 ? pool0 : pool1;
}

bool
testBlocks(Context &ctx)
{
    // 40 blocks in use at the same time (more than a thread caches), two
    // pools, three sizes
    const std::size_t sizes[] = { 1, 77, 4096 };
    const int         numBlocks = 40;

    for (int round=0; round<3; ++round) {
        std::vector<double *>     b(numBlocks);
        std::vector<std::size_t>  n(numBlocks);
        for (int i=0; i<numBlocks; ++i) {
            n[i] = sizes[i%3];
            b[i] = pool(i%2).allocate(n[i]);
            if (!ctx.check(b[i]!=nullptr, "allocate(%zu) returned null", n[i])
             || !ctx.check(reinterpret_cast<std::uintptr_t>(b[i])
                           % alignof(double)==0,
                           "block %d not aligned for double", i))
            {
                return false;
            }
            fill(b[i], n[i], i);
        }
        for (int i=0; i<numBlocks; ++i) {
            if (!ctx.check(intact(b[i], n[i], i),
                           "round %d: block %d (%zu elements) overlaps another "
                           "block", round, i, n[i]))
            {
                return false;
            }
        }
        // release in an order different from the allocation
        for (int i=0; i<numBlocks; ++i) {
            int j = (7*i) % numBlocks;
            pool(j%2).release(b[j]);
        }
    }
    // a released block is handed out again (same pool and size)
    double *a = pool(0).allocate(123);
    pool(0).release(a);
    double *a2 = pool(0).allocate(123);
    double *a3 = pool(0).allocate(123);
    bool ok = ctx.check(a2==a, "released block not reused")
           && ctx.check(a3!=a2, "block handed out twice");
    pool(0).release(a3);
    pool(0).release(a2);
    // release(nullptr) has no effect
    pool(1).release(nullptr);
    double *c = pool(1).allocate(5);
    fill(c, 5, 1);
    ok = ok && ctx.check(intact(c, 5, 1), "block of 5 elements corrupted");
    pool(1).release(c);
    return ok;
}

//
//  A block allocated (and cached) by a worker thread is released by the
//  main thread, so the allocator may hand out its address again; the
//  worker then allocates blocks of various sizes (including the released
//  one) and checks that they do not overlap.
//
bool
testCrossThread(Context &ctx)
{
    const std::size_t sizes[] = { 1000, 2000, 1000, 7, 1000 };
    bool    ok = true;
    double *handOver = nullptr;

    std::thread first([&] {
        double *keep = pool(0).allocate(1000);
        handOver     = pool(0).allocate(1000);
        fill(handOver, 1000, 1);
        pool(0).release(keep);
    });
    first.join();
    pool(0).release(handOver);          // not the allocating thread

    std::thread second([&] {
        for (int round=0; round<20; ++round) {
            std::vector<double *> b;
            for (std::size_t i=0; i<5; ++i) {
                b.push_back(pool(0).allocate(sizes[i]));
                fill(b[i], sizes[i], int(i));
            }
            for (std::size_t i=0; i<5; ++i) {
                ok = ok && intact(b[i], sizes[i], int(i));
            }
            for (std::size_t i=0; i<5; ++i) {
                pool(0).release(b[(i+round) % 5]);
            }
        }
    });
    second.join();

    std::vector<double *> handed(4);
    std::thread third([&] {
        for (int i=0; i<4; ++i) {
            handed[i] = pool(1).allocate(300);
        }
    });
    third.join();
    for (int i=0; i<4; ++i) {
        fill(handed[i], 300, 10+i);
    }
    for (int i=0; i<4; ++i) {
        ok = ok && intact(handed[i], 300, 10+i);
        pool(1).release(handed[i]);     // the allocating thread has ended
    }
    return ctx.check(ok, "blocks overlap after a release by another thread");
}

//
//  The packing buffers come from MemoryPool.  The AVX micro kernel loads the
//  packed B with vmovapd (32 bytes), the SSE kernels load A and B with
//  movapd (16 bytes); the other kernels need the alignment of T only.  A
//  vendored kernel (USE_BLIS_DGEMM, USE_BLIS_SGEMM, USE_BLIS_CGEMM,
//  USE_BLIS_ZGEMM) may need more (ULMBLAS_BLIS_..._ALIGN, e.g. 64 bytes for
//  aligned zmm loads).
//
#if defined(USE_BLIS_DGEMM)
constexpr std::size_t vendoredDgemmAlign = ULMBLAS_BLIS_DGEMM_ALIGN;
#else
constexpr std::size_t vendoredDgemmAlign = 1;
#endif
#if defined(USE_BLIS_ZGEMM)
constexpr std::size_t vendoredZgemmAlign0 = ULMBLAS_BLIS_ZGEMM_ALIGN;
#else
constexpr std::size_t vendoredZgemmAlign0 = 1;
#endif
#if defined(USE_BLIS_SGEMM)
constexpr std::size_t vendoredSgemmAlign = ULMBLAS_BLIS_SGEMM_ALIGN;
#else
constexpr std::size_t vendoredSgemmAlign = 1;
#endif
#if defined(USE_BLIS_CGEMM)
constexpr std::size_t vendoredCgemmAlign = ULMBLAS_BLIS_CGEMM_ALIGN;
#else
constexpr std::size_t vendoredCgemmAlign = 1;
#endif
//  the largest alignment of the vendored kernels for the other types
constexpr std::size_t vendoredZgemmAlign1 = vendoredZgemmAlign0>vendoredSgemmAlign
                                          ? vendoredZgemmAlign0 : vendoredSgemmAlign;
constexpr std::size_t vendoredZgemmAlign = vendoredZgemmAlign1>vendoredCgemmAlign
                                         ? vendoredZgemmAlign1 : vendoredCgemmAlign;

constexpr std::size_t
maxOf(std::size_t a, std::size_t b)
{
    return a>b ? a : b;
}

constexpr std::size_t
requiredAlignment(std::size_t alignOfT)
{
#if defined(USE_AVX)
    return maxOf(maxOf(32, vendoredDgemmAlign), vendoredZgemmAlign);
#elif defined(USE_SSE)
    return maxOf(maxOf(16, vendoredDgemmAlign), vendoredZgemmAlign);
#else
    return maxOf(maxOf(alignOfT, vendoredDgemmAlign), vendoredZgemmAlign);
#endif
}

template <typename T>
bool
testAlignment(Context &ctx, const char *type)
{
    static ulmBLAS::MemoryPool<T> pool;

    const std::size_t sizes[] = { 1, 3, 77, 4096, 1024*1024+7 };
    const std::size_t align = requiredAlignment(alignof(T));
    bool ok = true;

    for (int round=0; round<2; ++round) {           // fresh and reused blocks
        std::vector<T *> b;
        for (std::size_t n : sizes) {
            b.push_back(pool.allocate(n));
            ok = ok && reinterpret_cast<std::uintptr_t>(b.back()) % align==0;
            b.back()[0] = b.back()[n-1] = T(1);
        }
        for (T *p : b) {
            pool.release(p);
        }
    }
    return ctx.check(ok, "%s blocks not aligned to %zu bytes", type, align);
}

} // namespace

ULMBLAS_TEST(T0350, "auxiliary/MemoryPool: distinct and reused blocks, sizes, "
             "pools, more blocks than cached, release(nullptr), release by "
             "another thread")
{
    if (testBlocks(ctx) && testCrossThread(ctx)) {
        ctx.pass();
    }
}

ULMBLAS_TEST(T0353, "auxiliary/MemoryPool: blocks aligned as the micro kernels "
             "require (32 bytes with AVX, 16 with SSE, more for some "
             "vendored kernels)")
{
    if (testAlignment<float>(ctx, "float")
     && testAlignment<double>(ctx, "double")
     && testAlignment<std::complex<float>>(ctx, "complex<float>")
     && testAlignment<std::complex<double>>(ctx, "complex<double>"))
    {
        ctx.pass();
    }
}
