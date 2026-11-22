// std::complex<T> has the layout of T[2]: real part, imaginary part.  A C
// or Fortran caller passes complex vectors as arrays of float or double.
#include <complex>
#include <cstdio>

int
main()
{
    static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
                  "two doubles");

    // what a C caller passes: re, im, re, im, ...
    double buffer[6] = {1, 2, 3, 4, 5, 6};

    // the interface of ulmBLAS views it as complex numbers
    std::complex<double> *z = reinterpret_cast<std::complex<double> *>(buffer);
    for (int i = 0; i < 3; ++i) {
	z[i] *= std::complex<double>(0, 1); // multiply by i
    }

    // and a kernel may view complex numbers as pairs of doubles again
    double *d = reinterpret_cast<double *>(z);
    for (int i = 0; i < 6; ++i) {
	std::printf("%g ", d[i]);
    }
    std::printf("\n");
}
