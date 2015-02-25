#ifndef ULMBLAS_LEVEL1_COPY_TCC
#define ULMBLAS_LEVEL1_COPY_TCC 1

#include <ulmblas/auxiliary/conjugate.h>
#include <ulmblas/level1/copy.h>

namespace ulmBLAS {

template <typename IndexType, typename VX, typename VY>
void
copy(IndexType      n,
     bool           conjX,
     const VX       *x,
     IndexType      incX,
     VY             *y,
     IndexType      incY)
{
    if (!conjX) {
        for (IndexType i=0; i<n; ++i) {
            y[i*incY] = x[i*incX];
        }
    } else {
        for (IndexType i=0; i<n; ++i) {
            y[i*incY] = conjugate(x[i*incX]);
        }
    }
}

template <typename IndexType, typename VX, typename VY>
void
copy(IndexType      n,
     const VX       *x,
     IndexType      incX,
     VY             *y,
     IndexType      incY)
{
    copy(n, false, x, incX, y, incY);
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_COPY_TCC 1
