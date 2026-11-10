#ifndef ULMBLAS_LEVEL1_KERNEL_SSE_DOT_H
#define ULMBLAS_LEVEL1_KERNEL_SSE_DOT_H 1

#include <ulmblas/level1/kernel/ref/dot.h>

namespace ulmBLAS {
namespace sse {

using ref::dotc;
using ref::dotu;

template <typename IndexType>
void
dotu(IndexType n, const double *x, IndexType incX, const double *y,
     IndexType incY, double &result);

} // namespace sse
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_KERNEL_SSE_DOT_H 1

#include <ulmblas/level1/kernel/sse/dot.tcc>
