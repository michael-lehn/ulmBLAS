#ifndef ULMBLAS_LEVEL1_IAMAX_TCC
#define ULMBLAS_LEVEL1_IAMAX_TCC 1

#include <cmath>
#include <ulmblas/level1/iamax.h>

namespace ulmBLAS {

template <typename IndexType, typename VX>
IndexType
iamax(IndexType      n,
      const VX       *x,
      IndexType      incX)
{
    IndexType iAbsMaxX = 0;
    VX        absMaxX  = std::abs(x[iAbsMaxX]);

    if (n<=0) {
        return -1;
    }

    for (IndexType i=0; i<n; ++i) {
        if (std::abs(x[i*incX])>absMaxX) {
            iAbsMaxX = i;
            absMaxX = std::abs(x[i*incX]);
        }
    }
    return iAbsMaxX;
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_IAMAX_TCC 1
