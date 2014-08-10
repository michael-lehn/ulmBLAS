#ifndef ULMBLAS_SRC_LEVEL1_SWAP_TCC
#define ULMBLAS_SRC_LEVEL1_SWAP_TCC 1

#include <utility>
#include <src/level1/swap.h>

namespace ulmBLAS {

template <typename IndexType, typename VX, typename VY>
void
swap(IndexType      n,
     VX             *x,
     IndexType      incX,
     VY             *y,
     IndexType      incY)
{
    for (IndexType i=0; i<n; ++i) {
        std::swap(x[i*incX], y[i*incY]);
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1_SWAP_TCC 1
