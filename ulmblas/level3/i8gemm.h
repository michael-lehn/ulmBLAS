#ifndef ULMBLAS_LEVEL3_I8GEMM_H
#define ULMBLAS_LEVEL3_I8GEMM_H 1

//
//  GEMM with A and B in 8-bit integers and C in int32:
//
//      C <- alpha*A*B + beta*C      (modulo 2^32, as int32 with wraparound)
//
//  A and B are signed (int8_t) or unsigned (uint8_t), as given by unsignedA
//  and unsignedB; any strides.  With the SME2 kernels the driver of
//  ukernel/sme/i8gemm.tcc (int8 outer products), otherwise the generic gemm
//  computing in uint32 (the low 32 bits are those of the exact result).
//
#include <cstdint>
#include <ulmblas/config/simd.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/gemm.h>

#if defined(USE_SME)
#include <ulmblas/level3/ukernel/sme/i8gemm.tcc>
#endif

namespace ulmBLAS {

namespace i8gemm_detail {

template <typename TA, typename TB>
void
generic(std::int64_t m, std::int64_t n, std::int64_t k, std::uint32_t alpha,
        const void *A, std::int64_t incRowA, std::int64_t incColA,
        const void *B, std::int64_t incRowB, std::int64_t incColB,
        std::uint32_t beta, std::uint32_t *C, std::int64_t incRowC,
        std::int64_t incColC)
{
    gemm<std::ptrdiff_t>(m, n, k, alpha, false, static_cast<const TA *>(A),
                         incRowA, incColA, false, static_cast<const TB *>(B),
                         incRowB, incColB, beta, C, incRowC, incColC);
}

} // namespace i8gemm_detail

inline void
i8gemm(std::int64_t m, std::int64_t n, std::int64_t k, std::int32_t alpha,
       bool unsignedA, const void *A, std::int64_t incRowA,
       std::int64_t incColA, bool unsignedB, const void *B,
       std::int64_t incRowB, std::int64_t incColB, std::int32_t beta,
       std::int32_t *C, std::int64_t incRowC, std::int64_t incColC)
{
    if (m <= 0 || n <= 0) {
	return;
    }
    std::uint32_t *Cu = reinterpret_cast<std::uint32_t *>(C);
    if (alpha == 0 || k <= 0) {
	gescal(std::ptrdiff_t(m), std::ptrdiff_t(n), std::uint32_t(beta), Cu,
	       std::ptrdiff_t(incRowC), std::ptrdiff_t(incColC));
	return;
    }
#if defined(USE_SME)
    const std::uint8_t *A8 = static_cast<const std::uint8_t *>(A);
    const std::uint8_t *B8 = static_cast<const std::uint8_t *>(B);
    if (!unsignedA && !unsignedB) {
	sme::i8gemm_sme<false, false>(m, n, k, alpha, A8, incRowA, incColA, B8,
	                              incRowB, incColB, beta, C, incRowC,
	                              incColC);
    } else if (unsignedA && !unsignedB) {
	sme::i8gemm_sme<true, false>(m, n, k, alpha, A8, incRowA, incColA, B8,
	                             incRowB, incColB, beta, C, incRowC,
	                             incColC);
    } else if (!unsignedA && unsignedB) {
	sme::i8gemm_sme<false, true>(m, n, k, alpha, A8, incRowA, incColA, B8,
	                             incRowB, incColB, beta, C, incRowC,
	                             incColC);
    } else {
	sme::i8gemm_sme<true, true>(m, n, k, alpha, A8, incRowA, incColA, B8,
	                            incRowB, incColB, beta, C, incRowC,
	                            incColC);
    }
#else
    using namespace i8gemm_detail;
    const std::uint32_t a = std::uint32_t(alpha), b = std::uint32_t(beta);
    if (!unsignedA && !unsignedB) {
	generic<std::int8_t, std::int8_t>(m, n, k, a, A, incRowA, incColA, B,
	                                  incRowB, incColB, b, Cu, incRowC,
	                                  incColC);
    } else if (unsignedA && !unsignedB) {
	generic<std::uint8_t, std::int8_t>(m, n, k, a, A, incRowA, incColA, B,
	                                   incRowB, incColB, b, Cu, incRowC,
	                                   incColC);
    } else if (!unsignedA && unsignedB) {
	generic<std::int8_t, std::uint8_t>(m, n, k, a, A, incRowA, incColA, B,
	                                   incRowB, incColB, b, Cu, incRowC,
	                                   incColC);
    } else {
	generic<std::uint8_t, std::uint8_t>(m, n, k, a, A, incRowA, incColA, B,
	                                    incRowB, incColB, b, Cu, incRowC,
	                                    incColC);
    }
#endif
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_I8GEMM_H
