#ifndef ULMBLAS_AUXILIARY_PRINTMATRIX_TCC
#define ULMBLAS_AUXILIARY_PRINTMATRIX_TCC 1

#include <cstdio>
#include <complex>
#include <type_traits>
#include <ulmblas/auxiliary/printmatrix.h>

namespace ulmBLAS {

template <typename T, typename IndexType>
void
printMatrix(IndexType m, IndexType n,
            const T *X, IndexType incRowX, IndexType incColX)
{
    if (std::is_same<double,T>::value) {
        for (IndexType i=0; i<m; ++i) {
            for (IndexType j=0; j<n; ++j) {
                //printf(" %7.4lf", X[i*incRowX+j*incColX]);
                printf(" %15.3lf", X[i*incRowX+j*incColX]);
            }
            printf("\n");
        }
        printf("\n");
    } else if (std::is_same<float, T>::value) {
        for (IndexType i=0; i<m; ++i) {
            for (IndexType j=0; j<n; ++j) {
                //printf(" %7.4lf", X[i*incRowX+j*incColX]);
                printf(" %15.3f", X[i*incRowX+j*incColX]);
            }
            printf("\n");
        }
        printf("\n");
    } else if (std::is_same<std::complex<double>, T>::value) {
        for (IndexType i=0; i<m; ++i) {
            for (IndexType j=0; j<n; ++j) {
                //printf(" %7.4lf", X[i*incRowX+j*incColX]);
                printf(" (%15.3lf, %15.3lf) ",
                       std::real(X[i*incRowX+j*incColX]),
                       std::imag(X[i*incRowX+j*incColX]));
            }
            printf("\n");
        }
        printf("\n");
    }
}

template <typename T, typename IndexType>
void
printSylMatrix(IndexType m,
               const T *X, IndexType incRowX, IndexType incColX)
{
    for (IndexType i=0; i<m; ++i) {
        for (IndexType j=0; j<m; ++j) {
            if (i>j) {
                printf(" %5.3lf", X[i*incRowX+j*incColX]);
            } else {
                printf(" %5.3lf", X[j*incRowX+i*incColX]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

template <typename T, typename IndexType>
void
printSyuMatrix(IndexType m,
               const T *X, IndexType incRowX, IndexType incColX)
{
    for (IndexType i=0; i<m; ++i) {
        for (IndexType j=0; j<m; ++j) {
            if (i<j) {
                printf(" %5.3lf", X[i*incRowX+j*incColX]);
            } else {
                printf(" %5.3lf", X[j*incRowX+i*incColX]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_PRINTMATRIX_TCC
