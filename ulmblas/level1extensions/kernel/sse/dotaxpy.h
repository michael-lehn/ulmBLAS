#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTAXPY_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTAXPY_H 1

#include <ulmblas/level1extensions/kernel/ref/dotaxpy.h>

namespace ulmBLAS { namespace sse {

using ref::dotaxpy;

//
//  Fuse the computations:
//  (1) $z \leftarrow z + \alpha x$
//  (2) $\rho = x^T y$
//
//  Arguments $x$, $x^T$ or $y$ can be conjugated in the computation
//
template <typename IndexType>
    void
    dotaxpy(IndexType      n,
            bool           conjX,
            bool           conjXt,
            bool           conjY,
            const double   &alpha,
            const double   *x,
            IndexType      incX,
            const double   *y,
            IndexType      incY,
            double         *z,
            IndexType      incZ,
            double         &rho);

} } // namespace sse, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_SSE_DOTAXPY_H 1

#include <ulmblas/level1extensions/kernel/sse/dotaxpy.tcc>
