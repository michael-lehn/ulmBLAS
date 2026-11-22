// A template in a namespace and the C and Fortran 77 functions that call it.
// run.sh compiles this file with -c and lists its symbols with nm.
#include <cstddef>

namespace ulm {

template <typename IndexType, typename Alpha, typename VX>
void
scal(IndexType n, const Alpha &alpha, VX *x, IndexType incX)
{
    for (IndexType i = 0; i < n; ++i) {
	x[i * incX] *= alpha;
    }
}

} // namespace ulm

extern "C" {

// C interface: arguments by value
void
cblas_dscal(int n, double alpha, double *x, int incX)
{
    if (incX < 0) {
	x -= std::ptrdiff_t(incX) * (n - 1);
    }
    ulm::scal<std::ptrdiff_t>(n, alpha, x, incX);
}

// Fortran 77 interface: every argument by address, trailing underscore
void
dscal_(const int *n, const double *alpha, double *x, const int *incX)
{
    cblas_dscal(*n, *alpha, x, *incX);
}

} // extern "C"

// a C++ function with the same parameters: its name is mangled
void
dscalCpp(int n, double alpha, double *x, int incX)
{
    ulm::scal<std::ptrdiff_t>(n, alpha, x, incX);
}
