#ifndef ULMBLAS_AUXILIARY_MEMORYPOOL_TCC
#define ULMBLAS_AUXILIARY_MEMORYPOOL_TCC 1

#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/simd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#if defined(_WIN32)
#include <malloc.h>
#include <new>

//  Fiber local storage of kernel32 (declared here, not by <windows.h>,
//  whose macros could clash with names of ulmBLAS)
extern "C" {
__declspec(dllimport) unsigned long __stdcall
FlsAlloc(void(__stdcall *)(void *));
__declspec(dllimport) int __stdcall
FlsFree(unsigned long);
__declspec(dllimport) void *__stdcall
FlsGetValue(unsigned long);
__declspec(dllimport) int __stdcall
FlsSetValue(unsigned long, void *);
}
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
//  The alignment of the blocks in bytes: the packed panels are read with
//  aligned loads by the micro kernels.  The kernels of ulmblas/level3/
//  ukernel/kernels.h (vmovapd/vmovaps from the packed panels) need the
//  width of their registers (ALIGN of the kernel): 32 bytes for the AVX
//  kernels, compiled whenever __AVX__ is defined (also without USE_AVX:
//  USE_TESTPARAM configurations, AVX without FMA), and 64 bytes for the
//  AVX-512 kernels (zmm registers), compiled whenever __AVX512F__ is
//  defined: the selection tables use them, whichever kernel the
//  configuration names (e.g. -march=native on an AVX-512 machine).
//  KernelBackend checks this at compile time (ulmblas/level3/ukernel/
//  kernelbackend.h).  A vendored kernel (BLIS, AOCL, OpenBLAS) chosen by
//  the configuration may need more (e.g. 32 bytes for the AVX kernel
//  sandybridge_8x4 built without FMA, i.e. with the SSE kernel set).
//
#if defined(__AVX512F__)
constexpr std::size_t memoryPoolBaseAlign = 64;
#elif defined(USE_AVX) || defined(__AVX__)
constexpr std::size_t memoryPoolBaseAlign = 32;
#else
constexpr std::size_t memoryPoolBaseAlign = 16;
#endif

#if defined(USE_BLIS_DGEMM)
constexpr std::size_t memoryPoolDAlign = ULMBLAS_BLIS_DGEMM_ALIGN;
#else
constexpr std::size_t memoryPoolDAlign = 0;
#endif
#if defined(USE_BLIS_ZGEMM)
constexpr std::size_t memoryPoolZAlign = ULMBLAS_BLIS_ZGEMM_ALIGN;
#else
constexpr std::size_t memoryPoolZAlign = 0;
#endif
#if defined(USE_BLIS_SGEMM)
constexpr std::size_t memoryPoolSAlign = ULMBLAS_BLIS_SGEMM_ALIGN;
#else
constexpr std::size_t memoryPoolSAlign = 0;
#endif
#if defined(USE_BLIS_CGEMM)
constexpr std::size_t memoryPoolCAlign = ULMBLAS_BLIS_CGEMM_ALIGN;
#else
constexpr std::size_t memoryPoolCAlign = 0;
#endif

constexpr std::size_t
memoryPoolMax(std::size_t a, std::size_t b)
{
    return a > b ? a : b;
}

//  the alignment of all blocks (any element type)
constexpr std::size_t memoryPoolAlign = memoryPoolMax(
    memoryPoolMax(memoryPoolBaseAlign, memoryPoolDAlign),
    memoryPoolMax(memoryPoolMax(memoryPoolZAlign, memoryPoolSAlign),
                  memoryPoolCAlign));

//
//  Allocates a block of n elements with posix_memalign (_aligned_malloc on
//  Windows, which has no posix_memalign); does not throw, never returns
//  null.
//
template <typename T>
T *
malloc(std::size_t n)
{
    //  at least the alignment of T, a multiple of sizeof(void *) as
    //  required by posix_memalign
    constexpr std::size_t align = memoryPoolMax(memoryPoolAlign, alignof(T));
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

#if !defined(_WIN32)

template <typename T>
typename MemoryPool<T>::ThreadCache &
MemoryPool<T>::threadCache()
{
    thread_local ThreadCache cache;
    return cache;
}

#else

//
//  Windows: the cache of a thread is kept in fiber local storage, whose
//  callback frees it when the thread ends; FlsFree (when the library is
//  unloaded or the process ends) frees the caches of all threads.  A
//  thread_local object with a destructor is destroyed by libstdc++'s
//  __cxa_thread_atexit, which with some MinGW-w64 runtimes (conda-forge's
//  GCC 15) runs the destructors a second time at process exit, on freed
//  memory.
//
namespace memorypool {

template <typename Cache>
void __stdcall
releaseCache(void *cache)
{
    if (cache) {
	static_cast<Cache *>(cache)->~Cache();
	std::free(cache);
    }
}

struct FlsSlot
{
    static constexpr unsigned long none = 0xFFFFFFFFul; // FLS_OUT_OF_INDEXES

    unsigned long index;

    explicit FlsSlot(void(__stdcall *callback)(void *))
        : index(FlsAlloc(callback))
    {
    }

    ~FlsSlot()
    {
	if (index != none) {
	    FlsFree(index);
	    index = none;
	}
    }
};

} // namespace memorypool

template <typename T>
typename MemoryPool<T>::ThreadCache &
MemoryPool<T>::threadCache()
{
    static memorypool::FlsSlot slot(&memorypool::releaseCache<ThreadCache>);
    //  without a slot (none left, or freed at process exit) the cache of
    //  the thread is not freed
    thread_local ThreadCache *unmanaged = nullptr;

    ThreadCache *cache = unmanaged;
    if (!cache && slot.index != memorypool::FlsSlot::none) {
	cache = static_cast<ThreadCache *>(FlsGetValue(slot.index));
    }
    if (!cache) {
	void *p = std::malloc(sizeof(ThreadCache));
	if (!p) {
	    outOfMemory(1, sizeof(ThreadCache));
	}
	cache = new (p) ThreadCache();
	if (slot.index == memorypool::FlsSlot::none ||
	    !FlsSetValue(slot.index, cache)) {
	    unmanaged = cache;
	}
    }
    return *cache;
}

#endif

template <typename T> MemoryPool<T>::ThreadCache::~ThreadCache()
{
    for (Entry &e : entry) {
	if (e.block && !e.used) {
	    free(e.block);
	}
    }
}

namespace memorypool {

inline std::atomic<std::size_t> &
largest()
{
    static std::atomic<std::size_t> bytes(0);
    return bytes;
}

//  a relaxed load per call; a store only for a new maximum
inline void
recordSize(std::size_t bytes)
{
    std::atomic<std::size_t> &l = largest();
    std::size_t current = l.load(std::memory_order_relaxed);
    while (bytes > current && !l.compare_exchange_weak(
                                  current, bytes, std::memory_order_relaxed)) {
    }
}

} // namespace memorypool

inline std::size_t
memoryPoolLargest()
{
    return memorypool::largest().load(std::memory_order_relaxed);
}

inline void
memoryPoolResetLargest()
{
    memorypool::largest().store(0, std::memory_order_relaxed);
}

template <typename T>
T *
MemoryPool<T>::allocate(std::size_t n)
{
    memorypool::recordSize(n > SIZE_MAX / sizeof(T) ? SIZE_MAX : n * sizeof(T));
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
