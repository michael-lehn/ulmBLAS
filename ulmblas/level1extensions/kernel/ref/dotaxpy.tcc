#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTAXPY_TCC
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTAXPY_TCC 1

#include <complex>
#include <ulmblas/auxiliary/conj.h>
#include <ulmblas/level1extensions/kernel/ref/dotaxpy.h>

namespace ulmBLAS { namespace ref {

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
        Rho            &rho)
{
    rho = Rho(0);
    for (IndexType i=0; i<n; ++i) {
        z[i*incZ] += alpha*conj(x[i*incX], conjX);
        rho       += conj(x[i*incX], conjXt) * conj(y[i*incY], conjY);
    }
}

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTAXPY_TCC 1
