#ifndef CULMBLAS_SUPPORT_H
#define CULMBLAS_SUPPORT_H 1

//
//  Support code shared by the test suite and the benchmark (not part of
//  cULMblas): a device buffer, the cuBLAS reference, and the list of GEMM
//  implementations to test and to benchmark.
//

#include <culmblas/check.h>
#include <culmblas/gemm.h>
#include <step01/gemm.h>
#include <step02/gemm.h>
#include <step03/gemm.h>
#include <step04/gemm.h>
#include <step05/gemm.h>
#include <step06/gemm.h>
#include <step07/gemm.h>
#include <step08/gemm.h>
#include <step09/gemm.h>
#include <step10/gemm.h>
#include <step11/gemm.h>
#include <step12/gemm.h>
#include <step13/gemm.h>
#include <step14/gemm.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#define CULMBLAS_CHECK_CUBLAS(call)                                         \
    cULMblas::support::checkCublas((call), #call, __FILE__, __LINE__)

namespace cULMblas { namespace support {

inline void
checkCublas(cublasStatus_t status, const char *call, const char *file,
            int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(file) + ":"
                                 + std::to_string(line) + ": " + call
                                 + ": cuBLAS status " + std::to_string(status));
    }
}

//
//  A buffer of n doubles in device memory.
//
class DeviceBuffer
{
    public:
        explicit
        DeviceBuffer(std::size_t n_)
            : data(0), n(n_)
        {
            if (n > 0) {
                CULMBLAS_CHECK_CUDA(cudaMalloc(&data, n*sizeof(double)));
            }
        }

        ~DeviceBuffer()
        {
            cudaFree(data);
        }

        void
        upload(const std::vector<double> &x)
        {
            if (x.size() != n) {
                throw std::logic_error("DeviceBuffer::upload: size mismatch");
            }
            if (n > 0) {
                CULMBLAS_CHECK_CUDA(cudaMemcpy(data, x.data(),
                                               n*sizeof(double),
                                               cudaMemcpyHostToDevice));
            }
        }

        std::vector<double>
        download() const
        {
            std::vector<double> x(n);
            if (n > 0) {
                CULMBLAS_CHECK_CUDA(cudaMemcpy(x.data(), data,
                                               n*sizeof(double),
                                               cudaMemcpyDeviceToHost));
            }
            return x;
        }

        double       *data;
        std::size_t  n;

    private:
        DeviceBuffer(const DeviceBuffer &);
        DeviceBuffer &operator=(const DeviceBuffer &);
};

//
//  The cuBLAS reference: cublasDgemm on a handle that lives as long as this
//  object (as cuBLAS::Init in the course).
//
class Cublas
{
    public:
        Cublas()
        {
            CULMBLAS_CHECK_CUBLAS(cublasCreate(&handle));
        }

        ~Cublas()
        {
            cublasDestroy(handle);
        }

        int
        version() const
        {
            int v = 0;
            CULMBLAS_CHECK_CUBLAS(cublasGetVersion(handle, &v));
            return v;
        }

        void
        gemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
             double alpha,
             const double *A, std::ptrdiff_t ldA,
             const double *B, std::ptrdiff_t ldB,
             double beta,
             double *C, std::ptrdiff_t ldC,
             cudaStream_t stream = 0)
        {
            //  the 32-bit interface of cuBLAS
            if (m > INT_MAX || n > INT_MAX || k > INT_MAX || ldA > INT_MAX
             || ldB > INT_MAX || ldC > INT_MAX)
            {
                throw std::invalid_argument("Cublas::gemm: dimension "
                                            "exceeds INT_MAX");
            }
            CULMBLAS_CHECK_CUBLAS(cublasSetStream(handle, stream));
            CULMBLAS_CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N,
                                              CUBLAS_OP_N,
                                              int(m), int(n), int(k),
                                              &alpha, A, int(ldA),
                                              B, int(ldB),
                                              &beta, C, int(ldC)));
        }

        cublasHandle_t handle;

    private:
        Cublas(const Cublas &);
        Cublas &operator=(const Cublas &);
};

} } // namespace support, cULMblas

#if defined(CULMBLAS_WITH_MAGMA)
#include <support/magma.h>
#endif

namespace cULMblas { namespace support {

//
//  The GEMM implementations of cULMblas, in tutorial order.  A new step is
//  added here and is then tested and benchmarked like the others.  With
//  MAGMA (make MAGMA=...) the list also contains MAGMA's GEMMs as external
//  references: the tests report how they differ from BLAS, but only the
//  implementations of cULMblas have to pass.
//
typedef void (*GemmFunction)(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t,
                             double, const double *, std::ptrdiff_t,
                             const double *, std::ptrdiff_t,
                             double, double *, std::ptrdiff_t,
                             cudaStream_t);

struct Variant
{
    const char    *name;
    const char    *description;
    GemmFunction  gemm;
    bool          external;     // a reference of another library
};

inline const std::vector<Variant> &
variants()
{
    static const std::vector<Variant> list = {
        { "step01", "step 01: one thread per element of C, 32 x 32 blocks",
          &cULMblas::step01::gemm, false },
        { "step02", "step 02: a thread computes 4 elements of a row of C",
          &cULMblas::step02::gemm, false },
        { "step03", "step 03: a thread computes an 8 x 4 block of C",
          &cULMblas::step03::gemm, false },
        { "step04", "step 04: 4 x 4 block per thread, passes of 4 values of l",
          &cULMblas::step04::gemm, false },
        { "step05", "step 05: panels of A and B in shared memory",
          &cULMblas::step05::gemm, false },
        { "step06", "step 06: Tensor Cores (WMMA), a warp per 8 x 8 block of C",
          &cULMblas::step06::gemm, false },
        { "step07", "step 07: Tensor Cores, a warp per 32 x 32 block of C",
          &cULMblas::step07::gemm, false },
        { "step08", "step 08: Tensor Cores, panels in shared memory for the block",
          &cULMblas::step08::gemm, false },
        { "step09", "step 09: Tensor Cores, panels copied with cp.async, 2 buffers",
          &cULMblas::step09::gemm, false },
        { "step10", "step 10: mma.sync, operands with 16-byte loads",
          &cULMblas::step10::gemm, false },
        { "step11", "step 11: step 10, blocks in groups of 8 block rows",
          &cULMblas::step11::gemm, false },
        { "step12", "step 12: step 11, block per warp by the size of C",
          &cULMblas::step12::gemm, false },
        { "step13", "step 13: step 12, panels copied without checks",
          &cULMblas::step13::gemm, false },
        { "step14", "step 14: step 13, more panel buffers",
          &cULMblas::step14::gemm, false },
#if defined(CULMBLAS_WITH_MAGMA)
        { "magma", "MAGMA magma_dgemm (calls cublasDgemm)", &magmaGemm,
          true },
        { "magmablas", "MAGMA magmablas_dgemm (MAGMA's own kernel)",
          &magmablasGemm, true },
#endif
    };
    return list;
}

//
//  Selects variants by name (comma separated list, empty: all).
//
inline std::vector<Variant>
selectVariants(const std::string &names)
{
    std::vector<Variant> selected;
    if (names.empty()) {
        return variants();
    }
    std::size_t pos = 0;
    while (pos <= names.size()) {
        std::size_t end = names.find(',', pos);
        if (end == std::string::npos) {
            end = names.size();
        }
        std::string name = names.substr(pos, end - pos);
        bool found = false;
        for (const Variant &v : variants()) {
            if (name == v.name) {
                selected.push_back(v);
                found = true;
            }
        }
        if (!found) {
            throw std::invalid_argument("unknown variant '" + name + "'");
        }
        pos = end + 1;
    }
    return selected;
}

//
//  The first CUDA device, or an exception with a clear message if there is
//  none (no driver, no GPU).
//
inline cudaDeviceProp
requireDevice()
{
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        throw std::runtime_error(std::string("no CUDA device found")
                                 + (status != cudaSuccess
                                    ? std::string(" (")
                                      + cudaGetErrorString(status) + ")"
                                    : std::string()));
    }
    cudaDeviceProp prop;
    CULMBLAS_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    CULMBLAS_CHECK_CUDA(cudaSetDevice(0));
    return prop;
}

} } // namespace support, cULMblas

#endif // CULMBLAS_SUPPORT_H
