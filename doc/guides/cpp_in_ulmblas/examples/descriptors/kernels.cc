// Structs that describe a micro kernel: types and constants as static
// members, the kernel as a static member function.  A function template
// takes such a struct as its parameter and adapts to it at compile time.
#include <cstdio>

// C++11: a constexpr function consists of a single return statement
constexpr long
roundUp(long n, long r)
{
    return (n / r + (n % r != 0)) * r;
}

static_assert(roundUp(10, 4) == 12, "roundUp");

// a "kernel" in C, as the vendored kernels of BLIS or OpenBLAS: C = A B
// for an MR x NR block, A row by row, B column by column
extern "C" void
kernel_2x2(long k, const double *A, const double *B, double *C)
{
    for (int i = 0; i < 2; ++i) {
	for (int j = 0; j < 2; ++j) {
	    double s = 0;
	    for (long l = 0; l < k; ++l) {
		s += A[i * k + l] * B[j * k + l];
	    }
	    C[i + j * 2] = s;
	}
    }
}

extern "C" void
kernel_4x2(long k, const double *A, const double *B, double *C)
{
    for (int i = 0; i < 4; ++i) {
	for (int j = 0; j < 2; ++j) {
	    double s = 0;
	    for (long l = 0; l < k; ++l) {
		s += A[i * k + l] * B[j * k + l];
	    }
	    C[i + j * 4] = s;
	}
    }
}

// the descriptors
struct Kernel2x2
{
    typedef double T;
    static constexpr long MR = 2, NR = 2;

    static const char *
    name()
    {
	return "kernel_2x2";
    }

    template <typename... Args>
    static void
    call(Args... args)
    {
	kernel_2x2(args...);
    }
};

struct Kernel4x2
{
    typedef double T;
    static constexpr long MR = 4, NR = 2;

    static const char *
    name()
    {
	return "kernel_4x2";
    }

    template <typename... Args>
    static void
    call(Args... args)
    {
	kernel_4x2(args...);
    }
};

// generic code: everything about the kernel comes from K
template <typename K>
void
info(long m, long n)
{
    typedef typename K::T T;
    // a buffer whose size is known at compile time
    T C[K::MR * K::NR];
    const long mp = roundUp(m, K::MR) / K::MR, np = roundUp(n, K::NR) / K::NR;

    T A[K::MR * 3], B[K::NR * 3];
    for (long i = 0; i < K::MR * 3; ++i) {
	A[i] = 1;
    }
    for (long i = 0; i < K::NR * 3; ++i) {
	B[i] = 2;
    }
    K::call(3L, A, B, C);
    std::printf("%s: %ld x %ld blocks for C of %ld x %ld, C[0] = %g\n",
                K::name(), mp, np, m, n, C[0]);
}

int
main()
{
    info<Kernel2x2>(10, 7);
    info<Kernel4x2>(10, 7);
}
