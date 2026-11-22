// x <- alpha x: in C one function per type, in C++ one function template.
#include <complex>
#include <cstddef>
#include <cstdio>

// C: the same loop for every type, under a different name
void
sscal(std::ptrdiff_t n, float alpha, float *x, std::ptrdiff_t incX)
{
    for (std::ptrdiff_t i = 0; i < n; ++i) {
	x[i * incX] *= alpha;
    }
}

void
dscal(std::ptrdiff_t n, double alpha, double *x, std::ptrdiff_t incX)
{
    for (std::ptrdiff_t i = 0; i < n; ++i) {
	x[i * incX] *= alpha;
    }
}

// C++: one template; the compiler generates a function for every
// combination of types it is called with
template <typename IndexType, typename Alpha, typename VX>
void
scal(IndexType n, const Alpha &alpha, VX *x, IndexType incX)
{
    for (IndexType i = 0; i < n; ++i) {
	x[i * incX] *= alpha;
    }
}

int
main()
{
    float xs[3] = {1, 2, 3}, ys[3] = {1, 2, 3};
    double xd[3] = {1, 2, 3}, yd[3] = {1, 2, 3};
    std::complex<double> xz[3] = {{1, 1}, {2, 2}, {3, 3}};

    sscal(3, 2.0f, ys, 1);
    dscal(3, 2.0, yd, 1);

    // IndexType = std::ptrdiff_t, Alpha = float, VX = float
    scal(std::ptrdiff_t(3), 2.0f, xs, std::ptrdiff_t(1));
    // IndexType = int, Alpha = double, VX = double
    scal(3, 2.0, xd, 1);
    // IndexType = int, Alpha = double, VX = std::complex<double>
    scal(3, 2.0, xz, 1);

    std::printf("ys = %g %g %g (C)\n", ys[0], ys[1], ys[2]);
    std::printf("yd = %g %g %g (C)\n", yd[0], yd[1], yd[2]);
    std::printf("xs = %g %g %g\n", xs[0], xs[1], xs[2]);
    std::printf("xd = %g %g %g\n", xd[0], xd[1], xd[2]);
    std::printf("xz = (%g,%g) (%g,%g) (%g,%g)\n", xz[0].real(), xz[0].imag(),
                xz[1].real(), xz[1].imag(), xz[2].real(), xz[2].imag());
}
