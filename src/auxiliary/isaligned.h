#ifndef SRC_AUXILIARY_ISALIGNED_H
#define SRC_AUXILIARY_ISALIGNED_H 1

namespace ulmBLAS {

template <typename T, typename IndexType>
    bool
    isAligned(const T *address, IndexType bytes);

} // namespace ulmBLAS

#endif // SRC_AUXILIARY_ISALIGNED_H

#include <src/auxiliary/isaligned.tcc>
