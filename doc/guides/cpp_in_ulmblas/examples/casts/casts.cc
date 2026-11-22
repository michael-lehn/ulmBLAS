// The three casts of ulmBLAS, and why not the cast of C.
#include <complex>
#include <cstdint>
#include <cstdio>

// a kernel from a C library that does not declare its input const
extern "C" double
sum_c(long n, double *x)
{
    double s = 0;
    for (long i = 0; i < n; ++i) {
	s += x[i];
    }
    return s;
}

double
sum(long n, const double *x)
{
    // const_cast: only adds or removes const (or volatile); correct as
    // long as sum_c does not write
    return sum_c(n, const_cast<double *>(x));
}

int
main()
{
    const double x[4] = {1, 2, 3, 4};

    // static_cast: a conversion the language defines (here: a value of
    // another type, rounded toward zero)
    const long k = static_cast<long>(2.9);
    const double avg = sum(4, x) / static_cast<double>(4);

    // reinterpret_cast: the same bits, viewed with another type
    std::complex<double> z[2] = {{1, 2}, {3, 4}};
    double *d = reinterpret_cast<double *>(z);
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(d);

    std::printf("k = %ld, avg = %g, d[3] = %g, aligned to 16 bytes: %s\n", k,
                avg, d[3], address % 16 == 0 ? "yes" : "no");

#ifdef WRONG
    // the cast of C does all of this without a word: here it removes const
    // and changes the type at once
    float *f = (float *)x;
    // a C++ cast says what it does, and refuses the rest
    float *g = static_cast<float *>(x);
#endif
}
