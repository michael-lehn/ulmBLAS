#ifndef ULMBLAS_LEVEL1_SDSDOT_H
#define ULMBLAS_LEVEL1_SDSDOT_H 1

namespace ulmBLAS {

template <typename IndexType, typename VX, typename VY, typename Result>
    void
    sdsdot(IndexType    n,
           const VX     *x,
           IndexType    incX,
           const VY     *y,
           IndexType    incY,
           Result       &result);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_SDSDOT_H 1

#include <ulmblas/level1/sdsdot.tcc>
