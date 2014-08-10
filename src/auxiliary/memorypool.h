#ifndef ULMBLAS_SRC_AUXILIARY_MEMORYPOOL_H
#define ULMBLAS_SRC_AUXILIARY_MEMORYPOOL_H 1

#include <list>
#include <unordered_map>
#include <mutex>

namespace ulmBLAS {

template <typename T>
class MemoryPool
{
    public:

        T *
        allocate(size_t n);

        void
        release(T *block);

        virtual
        ~MemoryPool();

    private:

        typedef std::list<T *>                           BlockList;
        typedef std::unordered_map<size_t, BlockList>    Free;
        typedef std::unordered_map<T *, size_t>          Used;

        Free        _free;
        Used        _used;
        BlockList   _allocated;
        std::mutex  _mutex;
};


} // namespace ulmBLAS

#endif // ULMBLAS_SRC_AUXILIARY_MEMORYPOOL_H
