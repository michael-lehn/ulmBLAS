#ifndef ULMBLAS_AUXILIARY_ISALIGNED_H
#define ULMBLAS_AUXILIARY_ISALIGNED_H 1

namespace ulmBLAS {

template <typename T, typename IndexType>
    bool
    isAligned(const T *address, IndexType bytes);

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_ISALIGNED_H

#include <ulmblas/auxiliary/isaligned.tcc>
