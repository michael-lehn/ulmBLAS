#ifndef SRC_AUXILIARY_ISALIGNED_TCC
#define SRC_AUXILIARY_ISALIGNED_TCC 1

namespace ulmBLAS {

template <typename T, typename IndexType>
bool
isAligned(const T *address, IndexType bytes)
{
    return (reinterpret_cast<size_t>(address) % size_t(bytes) == 0);
}

} // namespace ulmBLAS

#endif // SRC_AUXILIARY_ISALIGNED_TCC
