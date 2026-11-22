// What ulmBLAS does not need: a class that owns memory, with a copy
// constructor and a move constructor.  The move constructor takes an
// rvalue reference and steals the memory of an object that is about to
// die (a prvalue) or that the caller gives up (an xvalue, std::move).
#include <cstdio>
#include <cstring>
#include <utility>

class Vector
{
  public:
    explicit Vector(long n) : n_(n), data_(new double[n]())
    {
	std::printf("  Vector(%ld): allocate\n", n);
    }

    Vector(const Vector &other) : n_(other.n_), data_(new double[other.n_])
    {
	std::memcpy(data_, other.data_, n_ * sizeof(double));
	std::printf("  copy: allocate and copy %ld elements\n", n_);
    }

    Vector(Vector &&other) : n_(other.n_), data_(other.data_)
    {
	other.n_ = 0;
	other.data_ = nullptr; // other is left empty but valid
	std::printf("  move: take over the memory\n");
    }

    Vector &
    operator=(const Vector &) = delete;
    Vector &
    operator=(Vector &&) = delete;

    ~Vector()
    {
	delete[] data_;
    }

    long
    size() const
    {
	return n_;
    }

  private:
    long n_;
    double *data_;
};

Vector
make(long n, bool big)
{
    Vector a(n), b(2 * n);
    return big ? std::move(b) : std::move(a); // xvalue: moved
}

int
main()
{
    std::printf("Vector x(3):\n");
    Vector x(3);

    std::printf("Vector y(x):  x is an lvalue\n");
    Vector y(x);

    std::printf("Vector z(std::move(x)):  an xvalue\n");
    Vector z(std::move(x));
    std::printf("  now x.size() = %ld, z.size() = %ld\n", x.size(), z.size());

    std::printf("Vector w(make(2, true)):  a prvalue\n");
    Vector w(make(2, true));
    std::printf("  w.size() = %ld\n", w.size());
}
