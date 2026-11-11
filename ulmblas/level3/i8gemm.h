#ifndef ULMBLAS_LEVEL3_I8GEMM_H
#define ULMBLAS_LEVEL3_I8GEMM_H 1

//
//  GEMM with A and B in 8-bit integers and C in int32:
//
//      C <- alpha*A*B + beta*C      (modulo 2^32, as int32 with wraparound)
//
//  A and B are signed (int8_t) or unsigned (uint8_t), as given by unsignedA
//  and unsignedB; any strides.  The driver of level3/i8gemm/driver.h with
//  the micro kernel of the build and the CPU:
//
//      SME2 (USE_SME)       32 x 32, 4-way outer products (sme/i8gemm.tcc)
//      AArch64              12 x 8, SDOT/UDOT; mixed signs USDOT/SUDOT if
//                           the CPU has FEAT_I8MM (neon/i8gemm.tcc)
//      AVX512-VNNI          32 x 12, VPDPBUSD on zmm (avx/i8gemm_busd.tcc)
//      AVX-VNNI             16 x 6, VPDPBUSD on ymm (avx/i8gemm_busd.tcc)
//      AVX2                 8 x 6, int16 VPMADDWD (avx/i8gemm.tcc)
//
//  otherwise (and on AArch64 without the dot product instructions) the
//  generic gemm computing in uint32 (the low 32 bits are those of the exact
//  result).
//
#include <cstdint>
#include <ulmblas/config/simd.h>
#include <ulmblas/level1extensions/gescal.h>
#include <ulmblas/level3/fpsrscope.h>
#include <ulmblas/level3/gemm.h>
#include <ulmblas/level3/i8gemm/driver.h>

#if defined(USE_SME)
#include <ulmblas/level3/ukernel/sme/i8gemm.tcc>
#endif
#if defined(__aarch64__)
#include <ulmblas/level3/ukernel/neon/i8gemm.tcc>
#endif
#if defined(__AVX2__)
#include <ulmblas/level3/ukernel/avx/i8gemm.tcc>
#include <ulmblas/level3/ukernel/avx/i8gemm_busd.tcc>
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

inline void
generic(std::int64_t m, std::int64_t n, std::int64_t k, std::int32_t alpha,
        bool unsignedA, const void *A, std::int64_t incRowA,
        std::int64_t incColA, bool unsignedB, const void *B,
        std::int64_t incRowB, std::int64_t incColB, std::int32_t beta,
        std::int32_t *C, std::int64_t incRowC, std::int64_t incColC)
{
    std::uint32_t *Cu = reinterpret_cast<std::uint32_t *>(C);
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
}

//  the driver with kernel K for the signs
template <typename K>
void
blocked(std::int64_t m, std::int64_t n, std::int64_t k, std::int32_t alpha,
        bool unsignedA, const void *A_, std::int64_t incRowA,
        std::int64_t incColA, bool unsignedB, const void *B_,
        std::int64_t incRowB, std::int64_t incColB, std::int32_t beta,
        std::int32_t *C, std::int64_t incRowC, std::int64_t incColC)
{
    const std::uint8_t *A = static_cast<const std::uint8_t *>(A_);
    const std::uint8_t *B = static_cast<const std::uint8_t *>(B_);
    if (!unsignedA && !unsignedB) {
	i8::driver<K, false, false>(m, n, k, alpha, A, incRowA, incColA, B,
	                            incRowB, incColB, beta, C, incRowC,
	                            incColC);
    } else if (unsignedA && !unsignedB) {
	i8::driver<K, true, false>(m, n, k, alpha, A, incRowA, incColA, B,
	                           incRowB, incColB, beta, C, incRowC, incColC);
    } else if (!unsignedA && unsignedB) {
	i8::driver<K, false, true>(m, n, k, alpha, A, incRowA, incColA, B,
	                           incRowB, incColB, beta, C, incRowC, incColC);
    } else {
	i8::driver<K, true, true>(m, n, k, alpha, A, incRowA, incColA, B,
	                          incRowB, incColB, beta, C, incRowC, incColC);
    }
}

} // namespace i8gemm_detail

inline void
i8gemm(std::int64_t m, std::int64_t n, std::int64_t k, std::int32_t alpha,
       bool unsignedA, const void *A, std::int64_t incRowA,
       std::int64_t incColA, bool unsignedB, const void *B,
       std::int64_t incRowB, std::int64_t incColB, std::int32_t beta,
       std::int32_t *C, std::int64_t incRowC, std::int64_t incColC)
{
    using namespace i8gemm_detail;

    if (m <= 0 || n <= 0) {
	return;
    }
    if (alpha == 0 || k <= 0) {
	gescal(std::ptrdiff_t(m), std::ptrdiff_t(n), std::uint32_t(beta),
	       reinterpret_cast<std::uint32_t *>(C), std::ptrdiff_t(incRowC),
	       std::ptrdiff_t(incColC));
	return;
    }
    //  C row major (e.g. NumPy): C^T = B^T A^T, so that the kernels update C
    //  directly (unit row stride) instead of through a local tile
    if (incRowC != 1 && incColC == 1) {
	i8gemm(n, m, k, alpha, unsignedB, B, incColB, incRowB, unsignedA, A,
	       incColA, incRowA, beta, C, incColC, incRowC);
	return;
    }
    const FpsrScope fpsrScope; // see level3/fpsrscope.h

#if defined(USE_SME)
    blocked<sme::I8Kernel>(m, n, k, alpha, unsignedA, A, incRowA, incColA,
                           unsignedB, B, incRowB, incColB, beta, C, incRowC,
                           incColC);
#elif defined(__aarch64__)
    if (neon::hasDotProd() && (unsignedA == unsignedB || neon::hasI8MM())) {
	blocked<neon::I8Kernel>(m, n, k, alpha, unsignedA, A, incRowA,
	                        incColA, unsignedB, B, incRowB, incColB, beta,
	                        C, incRowC, incColC);
	return;
    }
    generic(m, n, k, alpha, unsignedA, A, incRowA, incColA, unsignedB, B,
            incRowB, incColB, beta, C, incRowC, incColC);
#elif defined(__AVX2__)
#if defined(ULMBLAS_HAVE_AVX512VNNI_BUSD)
    if (avx::hasAvx512Vnni()) {
	blocked<avx::I8KernelBusd32x12>(m, n, k, alpha, unsignedA, A, incRowA,
	                                incColA, unsignedB, B, incRowB,
	                                incColB, beta, C, incRowC, incColC);
	return;
    }
#endif
#if defined(ULMBLAS_HAVE_AVXVNNI_BUSD)
    if (avx::hasAvxVnniBusd()) {
	blocked<avx::I8KernelBusd16x6>(m, n, k, alpha, unsignedA, A, incRowA,
	                               incColA, unsignedB, B, incRowB, incColB,
	                               beta, C, incRowC, incColC);
	return;
    }
#endif
    blocked<avx::I8Kernel>(m, n, k, alpha, unsignedA, A, incRowA, incColA,
                           unsignedB, B, incRowB, incColB, beta, C, incRowC,
                           incColC);
#else
    generic(m, n, k, alpha, unsignedA, A, incRowA, incColA, unsignedB, B,
            incRowB, incColB, beta, C, incRowC, incColC);
#endif
}

//
//  GEMM with A and B in 16-bit integers (signed) and C in int32, modulo 2^32
//  (int32 with wraparound):
//
//      C <- alpha*A*B + beta*C
//
//  SME2: 32 x 32 kernel with 2-way outer products; AVX2: the int16 kernel
//  of avx/i8gemm.tcc; otherwise the generic gemm in uint32.
//
inline void
i16gemm(std::int64_t m, std::int64_t n, std::int64_t k, std::int32_t alpha,
        const std::int16_t *A, std::int64_t incRowA, std::int64_t incColA,
        const std::int16_t *B, std::int64_t incRowB, std::int64_t incColB,
        std::int32_t beta, std::int32_t *C, std::int64_t incRowC,
        std::int64_t incColC)
{
    if (m <= 0 || n <= 0) {
	return;
    }
    if (alpha == 0 || k <= 0) {
	gescal(std::ptrdiff_t(m), std::ptrdiff_t(n), std::uint32_t(beta),
	       reinterpret_cast<std::uint32_t *>(C), std::ptrdiff_t(incRowC),
	       std::ptrdiff_t(incColC));
	return;
    }
    if (incRowC != 1 && incColC == 1) { // C row major: C^T = B^T A^T
	i16gemm(n, m, k, alpha, B, incColB, incRowB, A, incColA, incRowA, beta,
	        C, incColC, incRowC);
	return;
    }
    const FpsrScope fpsrScope; // see level3/fpsrscope.h

#if defined(USE_SME)
    i8::driver<sme::I16Kernel, false, false, std::int16_t>(
        m, n, k, alpha, A, incRowA, incColA, B, incRowB, incColB, beta, C,
        incRowC, incColC);
#elif defined(__AVX2__)
    i8::driver<avx::I8Kernel, false, false, std::int16_t>(
        m, n, k, alpha, A, incRowA, incColA, B, incRowB, incColB, beta, C,
        incRowC, incColC);
#else
    gemm<std::ptrdiff_t>(m, n, k, std::uint32_t(alpha), false, A, incRowA,
                         incColA, false, B, incRowB, incColB,
                         std::uint32_t(beta),
                         reinterpret_cast<std::uint32_t *>(C), incRowC,
                         incColC);
#endif
}

} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_I8GEMM_H
