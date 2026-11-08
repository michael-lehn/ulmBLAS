#ifndef ULMBLAS_LAPACK_BUFFER_H
#define ULMBLAS_LAPACK_BUFFER_H 1

#include <cstddef>
#include <ulmblas/auxiliary/memorypool.h>

namespace ulmBLAS {

//
//  Work space of the LAPACK functions: n elements of type T (not
//  initialized) from a MemoryPool, released when the buffer goes out of
//  scope.  Like the packing buffers of BLAS it never throws: if the memory
//  cannot be allocated the program is terminated with a message (no C++
//  exception may reach the C and Fortran interfaces).  T must be trivial.
//
template <typename T>
class Buffer
{
    public:
        explicit
        Buffer(std::size_t n = 0)
            : data_(n ? pool().allocate(n) : nullptr), n_(n)
        {
        }

        ~Buffer()
        {
            if (data_) {
                pool().release(data_);
            }
        }

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;

        //  (re)allocates n elements; the old content is lost
        void
        resize(std::size_t n)
        {
            if (n==n_) {
                return;
            }
            if (data_) {
                pool().release(data_);
            }
            data_ = n ? pool().allocate(n) : nullptr;
            n_    = n;
        }

        void
        fill(const T &value)
        {
            for (std::size_t i=0; i<n_; ++i) {
                data_[i] = value;
            }
        }

        T *
        data() const
        {
            return data_;
        }

        T &
        operator[](std::size_t i) const
        {
            return data_[i];
        }

        std::size_t
        size() const
        {
            return n_;
        }

    private:
        static MemoryPool<T> &
        pool()
        {
            static MemoryPool<T> p;
            return p;
        }

        T            *data_;
        std::size_t  n_;
};

} // namespace ulmBLAS

#endif // ULMBLAS_LAPACK_BUFFER_H
