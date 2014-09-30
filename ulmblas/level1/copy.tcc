#ifndef ULMBLAS_LEVEL1_COPY_TCC
#define ULMBLAS_LEVEL1_COPY_TCC 1

#include <ulmblas/level1/copy.h>

namespace ulmBLAS {

template <typename IndexType, typename VX, typename VY>
void
copy(IndexType      n,
     const VX       *x,
     IndexType      incX,
     VY             *y,
     IndexType      incY)
{
    for (IndexType i=0; i<n; ++i) {
        y[i*incY] = x[i*incX];
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_COPY_TCC 1
