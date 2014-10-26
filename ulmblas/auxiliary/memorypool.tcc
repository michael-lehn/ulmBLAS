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
#if defined(HAVE_SSE)
    _mm_free(block);
#   else
    delete [] block;
#   endif
}

template <typename T>
T *
MemoryPool<T>::allocate(size_t n)
{
    mutex_.lock();

    BlockList &free = free_[n];
    T         *block;

    if (free.empty()) {
        block = malloc<T>(n);
        allocated_.push_back(block);
    } else {
        block = free.back();
        free.pop_back();
    }
    used_[block] = n;

    mutex_.unlock();
    return block;
}

template <typename T>
void
MemoryPool<T>::release(T *block)
{
    mutex_.lock();

    if (block) {
        assert(used_.count(block)==1);
        size_t n = used_[block];
        free_[n].push_back(block);
    }

    mutex_.unlock();
}

template <typename T>
MemoryPool<T>::~MemoryPool()
{
    while (!allocated_.empty()) {
        free(allocated_.back());
        allocated_.pop_back();
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_MEMORYPOOL_TCC
