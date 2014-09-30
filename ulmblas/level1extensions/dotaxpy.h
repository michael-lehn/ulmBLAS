#ifndef ULMBLAS_LEVEL1EXTENSIONS_DOTAXPY_H
#define ULMBLAS_LEVEL1EXTENSIONS_DOTAXPY_H 1

namespace ulmBLAS {

//
//  Fuse the computations:
//  (1) $z \leftarrow z + \alpha x$
//  (2) $\rho = x^T y$
//
//  Arguments $x$, $x^T$ or $y$ can be conjugated in the computation
//
template <typename IndexType, typename Alpha, typename VX, typename VY,
          typename VZ, typename Rho>
    void
    dotaxpy(IndexType      n,
            bool           conjX,
            bool           conjXt,
            bool           conjY,
            const Alpha    &alpha,
            const VX       *x,
            IndexType      incX,
            const VY       *y,
            IndexType      incY,
            VZ             *z,
            IndexType      incZ,
            Rho            &rho);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_DOTAXPY_H 1

#include <ulmblas/level1extensions/dotaxpy.tcc>
