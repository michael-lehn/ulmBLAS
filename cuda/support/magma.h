#ifndef CULMBLAS_SUPPORT_MAGMA_H
#define CULMBLAS_SUPPORT_MAGMA_H 1

//
//  MAGMA as a reference (optional: make MAGMA=/path/to/magma).
//
//  - magma_dgemm is MAGMA's BLAS interface for device pointers.  In MAGMA
//    2.10.0 it is a wrapper: it calls cublasDgemm on the cuBLAS handle of
//    the queue (interface_cuda/blas_d_v2.cpp), so it measures cuBLAS.
//  - magmablas_dgemm is MAGMA's own CUDA kernel (magmablas/dgemm.cu, the
//    template gemm_template_nn: 16 x 16 threads, blocks of 64 x 64 of C,
//    k in steps of 16 through shared memory).
//
//  Both run on a MAGMA queue made from the default stream and a cuBLAS
//  handle of its own (magma_queue_create_from_cuda), i.e. on the stream
//  the benchmark times.
//

#include <magma_v2.h>

#include <climits>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace cULMblas { namespace support {

class Magma
{
    public:
        static Magma &
        instance()
        {
            static Magma magma;
            return magma;
        }

        std::string
        version() const
        {
            magma_int_t vmajor, vminor, vmicro;
            magma_version(&vmajor, &vminor, &vmicro);
            return std::to_string(vmajor) + "." + std::to_string(vminor)
                 + "." + std::to_string(vmicro);
        }

        magma_queue_t queue;

    private:
        Magma()
        {
            if (magma_init() != MAGMA_SUCCESS) {
                throw std::runtime_error("magma_init failed");
            }
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cublasCreate failed (MAGMA queue)");
            }
            magma_device_t device;
            magma_getdevice(&device);
            magma_queue_create_from_cuda(device, 0, handle, 0, &queue);
        }

        ~Magma()
        {
            magma_queue_destroy(queue);
            cublasDestroy(handle);
            magma_finalize();
        }

        cublasHandle_t handle;
};

inline void
magmaCheck(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
           std::ptrdiff_t ldA, std::ptrdiff_t ldB, std::ptrdiff_t ldC,
           cudaStream_t stream)
{
    if (stream != 0) {
        throw std::invalid_argument("MAGMA reference: default stream only");
    }
    if (m > INT_MAX || n > INT_MAX || k > INT_MAX || ldA > INT_MAX
     || ldB > INT_MAX || ldC > INT_MAX)
    {
        throw std::invalid_argument("MAGMA reference: dimension exceeds "
                                    "INT_MAX");
    }
}

//  magma_dgemm (calls cublasDgemm)
inline void
magmaGemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
          double alpha,
          const double *A, std::ptrdiff_t ldA,
          const double *B, std::ptrdiff_t ldB,
          double beta,
          double *C, std::ptrdiff_t ldC,
          cudaStream_t stream)
{
    magmaCheck(m, n, k, ldA, ldB, ldC, stream);
    magma_dgemm(MagmaNoTrans, MagmaNoTrans, magma_int_t(m), magma_int_t(n),
                magma_int_t(k), alpha, A, magma_int_t(ldA), B,
                magma_int_t(ldB), beta, C, magma_int_t(ldC),
                Magma::instance().queue);
}

//  magmablas_dgemm (MAGMA's own kernel)
inline void
magmablasGemm(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
              double alpha,
              const double *A, std::ptrdiff_t ldA,
              const double *B, std::ptrdiff_t ldB,
              double beta,
              double *C, std::ptrdiff_t ldC,
              cudaStream_t stream)
{
    magmaCheck(m, n, k, ldA, ldB, ldC, stream);
    magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, magma_int_t(m),
                    magma_int_t(n), magma_int_t(k), alpha, A,
                    magma_int_t(ldA), B, magma_int_t(ldB), beta, C,
                    magma_int_t(ldC), Magma::instance().queue);
}

} } // namespace support, cULMblas

#endif // CULMBLAS_SUPPORT_MAGMA_H
