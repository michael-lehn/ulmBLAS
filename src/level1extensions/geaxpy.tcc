#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_GEAXPY_TCC
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_GEAXPY_TCC 1

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
    for (IndexType j=0; j<n; ++j) {
        for (IndexType i=0; i<m; ++i) {
            Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
        }
    }
}

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_GEAXPY_TCC 1
