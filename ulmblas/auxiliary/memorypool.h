#ifndef ULMBLAS_AUXILIARY_MEMORYPOOL_H
#define ULMBLAS_AUXILIARY_MEMORYPOOL_H 1

#include <cstddef>

namespace ulmBLAS {

//
//  Buffers for the packed operands of the level 2 and level 3 routines: a
//  routine takes its buffers with allocate(n) at the beginning of a call
//  and returns them with release(block) at the end.
//
//  Each thread keeps the blocks it released (up to numEntries per element
//  type; if all are taken, a block not in use makes room) and hands them
//  out again for the same size, without a lock.
//  Measured on an Apple M5 (macOS 26) for the buffers of DGEMM (MC*KC+MR
//  and KC*NC+NR elements):
//
//  - Allocating and freeing the blocks in every call made DGEMM up to two
//    times slower for n <= 256 (n = 8: 0.39 instead of 0.20 us), although
//    the calls of malloc and free themselves took little time: the
//    allocator marks freed large blocks for deferred reclamation, and the
//    whole call (packing and micro kernel) became slower.  Blocks have to
//    be kept.
//  - The previous pool shared by all threads (lists of free blocks in hash
//    tables behind a mutex) kept them, but its bookkeeping took about a
//    third of the time of a 16 x 16 x 16 DGEMM.
//
//  A block is owned by the cache of the thread that allocated it; the
//  cache frees its blocks when the thread ends.  A block released by
//  another thread (or not cached because the cache was full) is freed.
//
template <typename T>
class MemoryPool
{
    public:

        T *
        allocate(std::size_t n);

        void
        release(T *block);

    private:

        struct Entry
        {
            T            *block;
            std::size_t  n;
            bool         used;
        };

        struct ThreadCache
        {
            static constexpr int numEntries = 16;

            Entry  entry[numEntries] = {};

            ~ThreadCache();
        };

        static ThreadCache &
        threadCache();
};


} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_MEMORYPOOL_H

#include <ulmblas/auxiliary/memorypool.tcc>
