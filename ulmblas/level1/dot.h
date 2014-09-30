#ifndef ULMBLAS_LEVEL1_DOT_H
#define ULMBLAS_LEVEL1_DOT_H 1

namespace ulmBLAS {

template <typename IndexType, typename VT>
    VT
    dotu(IndexType      n,
         const VT       *x,
         IndexType      incX,
         const VT       *y,
         IndexType      incY);

template <typename IndexType, typename VX, typename VY, typename Result>
    void
    dotu(IndexType      n,
         const VX       *x,
         IndexType      incX,
         const VY       *y,
         IndexType      incY,
         Result         &result);

template <typename IndexType, typename VT>
    VT
    dotc(IndexType      n,
         const VT       *x,
         IndexType      incX,
         const VT       *y,
         IndexType      incY);

template <typename IndexType, typename VX, typename VY, typename Result>
    void
    dotc(IndexType      n,
         const VX       *x,
         IndexType      incX,
         const VY       *y,
         IndexType      incY,
         Result         &result);

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL1_DOT_H 1

#include <ulmblas/level1/dot.tcc>
