#ifndef CULMBLAS_CHECK_H
#define CULMBLAS_CHECK_H 1

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

//
//  CULMBLAS_CHECK_CUDA(call): throws std::runtime_error with the CUDA error
//  message, the call and its location if the call does not return
//  cudaSuccess.
//
#define CULMBLAS_CHECK_CUDA(call)                                           \
    cULMblas::checkCuda((call), #call, __FILE__, __LINE__)

namespace cULMblas {

inline void
checkCuda(cudaError_t status, const char *call, const char *file, int line)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":"
                                 + std::to_string(line) + ": " + call + ": "
                                 + cudaGetErrorName(status) + ": "
                                 + cudaGetErrorString(status));
    }
}

} // namespace cULMblas

#endif // CULMBLAS_CHECK_H
