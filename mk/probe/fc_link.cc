// Probe: C++ function called from Fortran (name mangling: lower case and
// a trailing underscore, as assumed by the F77 interface of ulmBLAS).
#include <vector>

extern "C" int
ulmprb_(const int *n)
{
    std::vector<int> v(2, *n);
    return v[0] + v[1];
}
