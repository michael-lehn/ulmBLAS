//
//  cULMblas::gemm: the current implementation (the fastest step so far,
//  step 13), and the argument checks shared by all implementations.
//
#include <culmblas/gemm.h>
#include <step13/gemm.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace cULMblas {

void
checkGemmArguments(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
                   const double *A, std::ptrdiff_t ldA,
                   const double *B, std::ptrdiff_t ldB,
                   const double *C, std::ptrdiff_t ldC)
{
    const char *error = 0;

    if (m < 0) {
        error = "m < 0";
    } else if (n < 0) {
        error = "n < 0";
    } else if (k < 0) {
        error = "k < 0";
    } else if (ldA < std::max<std::ptrdiff_t>(1, m)) {
        error = "ldA < max(1, m)";
    } else if (ldB < std::max<std::ptrdiff_t>(1, k)) {
        error = "ldB < max(1, k)";
    } else if (ldC < std::max<std::ptrdiff_t>(1, m)) {
        error = "ldC < max(1, m)";
    } else if ((m > 0 && k > 0 && !A) || (k > 0 && n > 0 && !B)
            || (m > 0 && n > 0 && !C))
    {
        error = "null pointer to a matrix that is not empty";
    }
    if (error) {
        throw std::invalid_argument(std::string("cULMblas::gemm: ") + error);
    }
}

void
gemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
     double alpha,
     const double *A, std::ptrdiff_t ldA,
     const double *B, std::ptrdiff_t ldB,
     double beta,
     double *C, std::ptrdiff_t ldC,
     cudaStream_t stream)
{
    step13::gemm(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
}

} // namespace cULMblas
