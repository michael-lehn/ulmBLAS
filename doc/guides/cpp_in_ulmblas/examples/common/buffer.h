// A work buffer as in ulmBLAS (ulmblas/lapack/buffer.h): n elements of type
// T, released by the destructor, not copyable; aborts if memory is missing
// (no exceptions).  Explained in Chapter 9 with raii/buffer.cc.
#ifndef BUFFER_H
#define BUFFER_H 1

#include <cstdio>
#include <cstdlib>

template <typename T> class Buffer
{
  public:
    explicit Buffer(long n)
        : data_(static_cast<T *>(std::malloc(n * sizeof(T))))
    {
	if (!data_) {
	    std::fprintf(stderr, "out of memory\n");
	    std::abort();
	}
    }

    ~Buffer()
    {
	std::free(data_);
    }

    Buffer(const Buffer &) = delete;
    Buffer &
    operator=(const Buffer &) = delete;

    T *
    data() const
    {
	return data_;
    }

    T &
    operator[](long i) const
    {
	return data_[i];
    }

  private:
    T *data_;
};

#endif // BUFFER_H
