#ifndef ULMBLAS_LEVEL1_SCAL_TCC
#define ULMBLAS_LEVEL1_SCAL_TCC 1

#include <ulmblas/level1/scal.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename VX>
void
scal(IndexType      n,
     const Alpha    &alpha,
     VX             *x,
     IndexType      incX)
{
    if (alpha!=Alpha(1)) {
        for (IndexType i=0; i<n; ++i) {
            x[i*incX] *= alpha;
        }
    } else if (alpha==Alpha(0)) {
        for (IndexType i=0; i<n; ++i) {
            x[i*incX] = Alpha(0);
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_SCAL_TCC 1
