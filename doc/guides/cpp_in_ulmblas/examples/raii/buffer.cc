// RAII (resource acquisition is initialization): the constructor acquires
// a work buffer, the destructor releases it, on every way out of the
// scope.  As the class Buffer of ulmBLAS (ulmblas/lapack/buffer.h).
#include <cstdio>
#include <cstdlib>

long live = 0; // buffers allocated and not yet released

template <typename T> class Buffer
{
  public:
    explicit Buffer(long n)
        : data_(static_cast<T *>(std::malloc(n * sizeof(T))))
    {
	if (!data_) {
	    std::fprintf(stderr, "out of memory\n");
	    std::abort(); // no exceptions: callers may be C or Fortran
	}
	++live;
    }

    ~Buffer()
    {
	std::free(data_);
	--live;
    }

    // a copy would free the same memory twice
    Buffer(const Buffer &) = delete;
    Buffer &
    operator=(const Buffer &) = delete;

    T &
    operator[](long i) const
    {
	return data_[i];
    }

  private:
    T *data_;
};

// returns the index of the first negative element, -1 if there is none
long
firstNegative(long n, const double *x)
{
    Buffer<double> work(n); // e.g. a copy of x for a computation
    for (long i = 0; i < n; ++i) {
	work[i] = x[i];
	if (work[i] < 0) {
	    std::printf("  in the loop: %ld buffer(s) live\n", live);
	    return i; // early return: work is released here
	}
    }
    return -1; // and here
}

int
main()
{
    const double x[4] = {1, 2, -3, 4};
    std::printf("first negative: %ld\n", firstNegative(4, x));
    std::printf("after the call: %ld buffer(s) live\n", live);
}
