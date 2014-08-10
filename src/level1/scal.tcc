#ifndef ULMBLAS_SRC_LEVEL1_SCAL_TCC
#define ULMBLAS_SRC_LEVEL1_SCAL_TCC 1

#include <src/level1/scal.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename VX>
void
scal(IndexType      n,
     const Alpha    &alpha,
     VX             *x,
     IndexType      incX)
{
    for (IndexType i=0; i<n; ++i) {
        x[i*incX] *= alpha;
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_SCAL_TCC 1
