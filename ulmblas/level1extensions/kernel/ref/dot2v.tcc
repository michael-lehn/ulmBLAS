#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOT2V_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOT2V_TCC 1

#include <ulmblas/level1extensions/kernel/ref/dot2v.h>

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename VX0, typename VX1, typename VY,
          typename Result>
void
dotu2v(IndexType      n,
       const VX0      *x0,
       IndexType      incX0,
       const VX1      *x1,
       IndexType      incX1,
       VY             *y,
       IndexType      incY,
       Result         *result,
       IndexType      resultInc)
{
    Result &result0 = result[0*resultInc];
    Result &result1 = result[1*resultInc];

    result0 = result1 = Result(0);

    for (IndexType i=0; i<n; ++i) {
        result0 += x0[i*incX1]*y[i*incY];
        result1 += x1[i*incX1]*y[i*incY];
    }
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOT2V_TCC 1
