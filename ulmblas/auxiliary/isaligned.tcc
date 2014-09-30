#ifndef ULMBLAS_AUXILIARY_ISALIGNED_TCC
#define ULMBLAS_AUXILIARY_ISALIGNED_TCC 1

#include <cstdlib>
#include <ulmblas/auxiliary/isaligned.h>

namespace ulmBLAS {

template <typename T, typename IndexType>
bool
isAligned(const T *address, IndexType bytes)
{
    return (reinterpret_cast<size_t>(address) % size_t(bytes) == 0);
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_ISALIGNED_TCC
