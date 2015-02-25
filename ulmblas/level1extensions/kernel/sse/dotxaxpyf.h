#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXAXPYF_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXAXPYF_H 1

#include <ulmblas/config/fusefactor.h>
#include <ulmblas/level1extensions/kernel/ref/dotxaxpyf.h>
#include <type_traits>

namespace ulmBLAS { namespace sse {

using ref::dotxaxpyf;

template <typename IndexType>
    typename std::enable_if<std::is_integral<IndexType>::value
                         && FuseFactor<double>::dotuxf==2,
    void>::type
    dotxaxpyf(IndexType      n,
              bool           conjX,
              bool           conjXt,
              bool           conjY,
              const double   &alpha,
              const double   *a,
              IndexType      incA,
              const double   *X,
              IndexType      incRowX,
              IndexType      incColX,
              const double   *y,
              IndexType      incY,
              double         *z,
              IndexType      incZ,
              double         *rho,
              IndexType      incRho);

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTXAXPYF_H 1

#include <ulmblas/level1extensions/kernel/sse/dotxaxpyf.tcc>
