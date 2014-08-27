#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_GEAXPY_TCC
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_GEAXPY_TCC 1

#include <src/level1/axpy.h>
#include <src/level1extensions/geaxpy.h>

namespace ulmBLAS {

template <typename IndexType, typename Alpha, typename MX, typename MY>
void
geaxpy(IndexType      m,
       IndexType      n,
       const Alpha    &alpha,
       const MX       *X,
       IndexType      incRowX,
       IndexType      incColX,
       MY             *Y,
       IndexType      incRowY,
       IndexType      incColY)
{
    const IndexType    UnitStride(1);

    if (incRowX==UnitStride && incRowY==UnitStride) {
//
//      X and Y are both column major
//
        for (IndexType j=0; j<n; ++j) {
            axpy(m, alpha,
                 &X[j*incColX], UnitStride,
                 &Y[j*incColY], UnitStride);
        }
    } else if (incColX==UnitStride && incColY==UnitStride) {
//
//      X and Y are both row major
//
        for (IndexType i=0; i<m; ++i) {
            axpy(n, alpha,
                 &X[i*incRowX], UnitStride,
                 &Y[i*incRowY], UnitStride);
        }
    } else {
//
//      General case
//
        for (IndexType j=0; j<n; ++j) {
            for (IndexType i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
            }
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_GEAXPY_TCC 1
