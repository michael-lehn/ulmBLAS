#ifndef ULMBLAS_AUXILIARY_MEMORYPOOL_TCC
#define ULMBLAS_AUXILIARY_MEMORYPOOL_TCC 1

#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/simd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace ulmBLAS {

//
//  Out of memory: a buffer for packed operands could not be allocated.  The
//  BLAS interfaces have no way to report it (and returning without
//  computing would leave wrong results), so the message goes to stderr and
//  the program is terminated (as in BLIS).  The buffers have a fixed size
//  (block sizes), independent of the matrix dimensions; if they cannot be
//  allocated, the process is out of memory.  No exception is thrown:
//  exceptions must not pass through the C and Fortran interfaces.
//
[[noreturn]] inline void
outOfMemory(std::size_t n, std::size_t size)
{
    std::fprintf(stderr,
                 "ulmBLAS: out of memory (buffer of %zu elements of "
                 "%zu bytes), program terminated\n",
                 n, size);
    std::abort();
}

//
//  Allocates a block of n elements with posix_memalign (_aligned_malloc on
//  Windows, which has no posix_memalign); does not throw, never returns
//  null.
//
template <typename T>
T *
malloc(std::size_t n)
{
//  A vendored kernel (BLIS, AOCL, OpenBLAS) may need a larger alignment of
//  the packed panels than the kernel set it is combined with (e.g. 32 bytes
//  for the AVX kernel sandybridge_8x4 built without FMA, i.e. with the SSE
//  kernel set; 64 bytes for aligned loads of zmm registers).
#if defined(USE_BLIS_DGEMM) || defined(USE_BLIS_XGEMM)
#if defined(USE_AVX)
    constexpr std::size_t base = 32;
#else
    constexpr std::size_t base = 16;
#endif
#if defined(USE_BLIS_DGEMM)
    constexpr std::size_t dalign = ULMBLAS_BLIS_DGEMM_ALIGN;
#else
    constexpr std::size_t dalign = 0;
#endif
#if defined(USE_BLIS_ZGEMM)
    constexpr std::size_t zalign = ULMBLAS_BLIS_ZGEMM_ALIGN;
#else
    constexpr std::size_t zalign = 0;
#endif
#if defined(USE_BLIS_SGEMM)
    constexpr std::size_t salign = ULMBLAS_BLIS_SGEMM_ALIGN;
#else
    constexpr std::size_t salign = 0;
#endif
#if defined(USE_BLIS_CGEMM)
    constexpr std::size_t calign = ULMBLAS_BLIS_CGEMM_ALIGN;
#else
    constexpr std::size_t calign = 0;
#endif
    constexpr std::size_t align0 = (dalign > base) ? dalign : base;
    constexpr std::size_t align1 = (zalign > align0) ? zalign : align0;
    constexpr std::size_t align2 = (salign > align1) ? salign : align1;
    constexpr std::size_t align = (calign > align2) ? calign : align2;
#elif defined(USE_AVX)
    constexpr std::size_t align = 32;
#else
    //  At least the alignment of T, a multiple of sizeof(void *) as required by
    //  posix_memalign.
    constexpr std::size_t align = (alignof(T) > 16) ? alignof(T) : 16;
#endif
    void *block = nullptr;

    if (n == 0) {
	n = 1;
    }
#if defined(_WIN32)
    if (n > SIZE_MAX / sizeof(T) ||
        !(block = _aligned_malloc(n * sizeof(T), align))) {
	outOfMemory(n, sizeof(T));
    }
#else
    if (n > SIZE_MAX / sizeof(T) ||
        posix_memalign(&block, align, n * sizeof(T))) {
	outOfMemory(n, sizeof(T));
    }
#endif
    return static_cast<T *>(block);
}

//
//  The elements are trivially destructible (float, double, complex).
//
template <typename T>
void
free(T *block)
{
#if defined(_WIN32)
    _aligned_free(block);
#else
    std::free(block);
#endif
}

template <typename T>
typename MemoryPool<T>::ThreadCache &
MemoryPool<T>::threadCache()
{
    thread_local ThreadCache cache;
    return cache;
}

template <typename T> MemoryPool<T>::ThreadCache::~ThreadCache()
{
    for (Entry &e : entry) {
	if (e.block && !e.used) {
	    free(e.block);
	}
    }
}

template <typename T>
T *
MemoryPool<T>::allocate(std::size_t n)
{
    ThreadCache &cache = threadCache();

    for (Entry &e : cache.entry) {
	if (e.block && !e.used && e.n == n) {
	    e.used = true;
	    return e.block;
	}
    }

    T *block = malloc<T>(n);
    Entry *empty = nullptr, *unused = nullptr;

    for (Entry &e : cache.entry) {
	//  An entry with the address of a new block is stale: its block
	//  was released by another thread and freed.
	if (e.block == block) {
	    e = Entry();
	}
	if (!e.block && !empty) {
	    empty = &e;
	}
	if (e.block && !e.used && !unused) {
	    unused = &e;
	}
    }
    //  If the cache is full, a block of another size that is not in use
    //  makes room.
    if (!empty && unused) {
	free(unused->block);
	empty = unused;
    }
    if (empty) {
	*empty = Entry{block, n, true};
    }
    return block;
}

template <typename T>
void
MemoryPool<T>::release(T *block)
{
    if (!block) {
	return;
    }
    for (Entry &e : threadCache().entry) {
	if (e.block == block) {
	    e.used = false;
	    return;
	}
    }
    free(block);
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_MEMORYPOOL_TCC
