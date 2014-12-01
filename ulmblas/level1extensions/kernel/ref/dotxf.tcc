#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXF_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXF_TCC 1

#include <type_traits>
#include <ulmblas/level1extensions/kernel/ref/dotxf.h>

namespace ulmBLAS { namespace ref {

template <typename T>
int
dotuxf_fusefactor()
{
    if (std::is_same<T,double>::value) {
        return 4;
    }
    return 1;
}

template <typename IndexType, typename TX, typename TY, typename Result>
void
dotuxf(IndexType      n,
       const TX       *X,
       IndexType      incRowX,
       IndexType      incColX,
       const TY       *y,
       IndexType      incY,
       Result         *result,
       IndexType      resultInc)
{
    Result    &result1 = result[0*resultInc];
    Result    &result2 = result[1*resultInc];
    Result    &result3 = result[2*resultInc];
    Result    &result4 = result[3*resultInc];

    const TX  *X1      = &X[0*incRowX];
    const TX  *X2      = &X[1*incRowX];
    const TX  *X3      = &X[2*incRowX];
    const TX  *X4      = &X[3*incRowX];

    result1 = result2 = result3 = result4 = Result(0);

    for (IndexType i=0; i<n; ++i) {
        result1 += X1[i*incColX]*y[i*incY];
        result2 += X2[i*incColX]*y[i*incY];
        result3 += X3[i*incColX]*y[i*incY];
        result4 += X4[i*incColX]*y[i*incY];
    }
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXF_TCC
