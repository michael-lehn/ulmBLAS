#ifndef ULMBLAS_LEVEL1_SDSDOT_TCC
#define ULMBLAS_LEVEL1_SDSDOT_TCC 1

#include <ulmblas/level1/sdsdot.h>

namespace ulmBLAS {

//
//  result += x^T y with the products and the sum computed in the precision
//  of Result (SDSDOT, DSDOT: single precision vectors, double precision
//  accumulation), in the order of the elements as in the Reference BLAS.
//
template <typename IndexType, typename VX, typename VY, typename Result>
void
sdsdot(IndexType    n,
       const VX     *x,
       IndexType    incX,
       const VY     *y,
       IndexType    incY,
       Result       &result)
{
    for (IndexType i=0; i<n; ++i) {
        result += Result(x[i*incX])*Result(y[i*incY]);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_SDSDOT_TCC 1
