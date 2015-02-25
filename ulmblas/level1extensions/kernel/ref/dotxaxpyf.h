#ifndef ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXAXPYF_H
#define ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXAXPYF_H 1

namespace ulmBLAS { namespace ref {

template <typename IndexType, typename Alpha, typename VA, typename MX,
          typename VY, typename VZ, typename Rho>
    void
    dotxaxpyf(IndexType      n,
              bool           conjX,
              bool           conjXt,
              bool           conjY,
              const Alpha    &alpha,
              const VA       *a,
              IndexType      incA,
              const MX       *X,
              IndexType      incRowX,
              IndexType      incColX,
              const VY       *y,
              IndexType      incY,
              VZ             *z,
              IndexType      incZ,
              Rho            *rho,
              IndexType      incRho);

} } // namespace ref, ulmBLAS

#endif // ULMBLAS_LEVEL1EXTENSIONS_KERNEL_REF_DOTXAXPYF_H 1

#include <ulmblas/level1extensions/kernel/ref/dotxaxpyf.tcc>
