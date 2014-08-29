#ifndef ULMBLAS_SRC_LEVEL1EXTENSIONS_DOTXAXPYF_H
#define ULMBLAS_SRC_LEVEL1EXTENSIONS_DOTXAXPYF_H 1

namespace ulmBLAS {

template <typename T>
    int
    dotaxpyf_fusefactor();

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
              const VY       *Y,
              IndexType      incY,
              VZ             *Z,
              IndexType      incZ,
              Rho            *rho,
              IndexType      incRho);

} // namespace ulmBLAS

#endif // ULMBLAS_SRC_LEVEL1EXTENSIONS_DOTXAXPYF_H 1

#include <src/level1extensions/dotxaxpyf.tcc>
