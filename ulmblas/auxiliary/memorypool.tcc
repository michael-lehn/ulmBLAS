#ifndef ULMBLAS_AUXILIARY_MEMORYPOOL_TCC
#define ULMBLAS_AUXILIARY_MEMORYPOOL_TCC 1

#include <cassert>
#include <list>
#include <unordered_map>

#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/simd.h>

#if defined(HAVE_SSE)
#   include <xmmintrin.h>
#endif

namespace ulmBLAS {

template <typename T>
T *
malloc(size_t n)
{
#if defined(HAVE_SSE)
    return reinterpret_cast<T *>(_mm_malloc(n*sizeof(T), 16));
#   else
    return new T[n];
#   endif
}

template <typename T>
void
free(T *block)
{
#if defined(HVAE_SSE)
    _mm_free(block);
#   else
    delete [] block;
#   endif
}

template <typename T>
T *
MemoryPool<T>::allocate(size_t n)
{
    _mutex.lock();

    BlockList &free = _free[n];
    T         *block;

    if (free.empty()) {
        block = malloc<T>(n);
        _allocated.push_back(block);
    } else {
        block = free.back();
        free.pop_back();
    }
    _used[block] = n;

    _mutex.unlock();
    return block;
}

template <typename T>
void
MemoryPool<T>::release(T *block)
{
    _mutex.lock();

    assert(_used.count(block)==1);
    size_t n = _used[block];
    _free[n].push_back(block);

    _mutex.unlock();
}

template <typename T>
MemoryPool<T>::~MemoryPool()
{
    for (auto it=_allocated.begin(); it!=_allocated.end(); ++it) {
        free(*it);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_MEMORYPOOL_TCC
