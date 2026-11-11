#ifndef ULMBLAS_LEVEL3_UKERNEL_NEON_I8GEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_NEON_I8GEMM_TCC 1

//
//  ulmBLAS: micro kernel of the 8-bit integer GEMM (level3/i8gemm/driver.h)
//  for AArch64 with the dot product instructions: a 12 x 8 tile of C (int32)
//  in 24 registers, four rows per register.  Per group of 4 values of k:
//  three loads of A (12 rows x 4 bytes), two of B (8 columns x 4 bytes) and
//  24 indexed dot products (the column j of B is lane j % 4 of a register
//  of B):
//
//      signed x signed       SDOT          (FEAT_DotProd)
//      unsigned x unsigned   UDOT          (FEAT_DotProd)
//      unsigned x signed     USDOT         (FEAT_I8MM)
//      signed x unsigned     SUDOT         (FEAT_I8MM)
//
//  The functions have target attributes, so the kernel is also in builds
//  for a baseline without these features (Linux on generic AArch64); the
//  dispatcher (level3/i8gemm.h) checks them at run time (hasDotProd,
//  hasI8MM).
//
#include <arm_neon.h>
#include <cstdint>
#include <type_traits>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#endif

#define ULMBLAS_NEON_DOT __attribute__((target("+dotprod")))
#define ULMBLAS_NEON_I8MM __attribute__((target("+dotprod,+i8mm")))

namespace ulmBLAS {
namespace neon {

inline bool
hasDotProd()
{
#if defined(__ARM_FEATURE_DOTPROD)
    return true;
#elif defined(__APPLE__)
    return true; // all Apple Silicon
#elif defined(__linux__)
    static const bool has = (getauxval(AT_HWCAP) & (1UL << 20)) != 0;
    return has; // HWCAP_ASIMDDP
#else
    return false;
#endif
}

inline bool
hasI8MM()
{
#if defined(__ARM_FEATURE_MATMUL_INT8)
    return true;
#elif defined(__APPLE__)
    static const bool has = [] {
	int v = 0;
	size_t len = sizeof(v);
	return sysctlbyname("hw.optional.arm.FEAT_I8MM", &v, &len, nullptr,
	                    0) == 0 &&
	       v == 1;
    }();
    return has;
#elif defined(__linux__)
    static const bool has = (getauxval(AT_HWCAP2) & (1UL << 13)) != 0;
    return has; // HWCAP2_I8MM
#else
    return false;
#endif
}

//  acc += A(4 rows) . B(column = lane L) for the signedness UA, UB
template <bool UA, bool UB, int L> struct I8Dot;

template <int L> struct I8Dot<false, false, L>
{
    ULMBLAS_NEON_DOT static inline int32x4_t
    dot(int32x4_t acc, uint8x16_t a, uint8x16_t b)
    {
	return vdotq_laneq_s32(acc, vreinterpretq_s8_u8(a),
	                       vreinterpretq_s8_u8(b), L);
    }
};

template <int L> struct I8Dot<true, true, L>
{
    ULMBLAS_NEON_DOT static inline int32x4_t
    dot(int32x4_t acc, uint8x16_t a, uint8x16_t b)
    {
	return vreinterpretq_s32_u32(
	    vdotq_laneq_u32(vreinterpretq_u32_s32(acc), a, b, L));
    }
};

template <int L> struct I8Dot<true, false, L>
{
    ULMBLAS_NEON_I8MM static inline int32x4_t
    dot(int32x4_t acc, uint8x16_t a, uint8x16_t b)
    {
	return vusdotq_laneq_s32(acc, a, vreinterpretq_s8_u8(b), L);
    }
};

template <int L> struct I8Dot<false, true, L>
{
    ULMBLAS_NEON_I8MM static inline int32x4_t
    dot(int32x4_t acc, uint8x16_t a, uint8x16_t b)
    {
	return vsudotq_laneq_s32(acc, vreinterpretq_s8_u8(a), b, L);
    }
};

//  C(4 rows) <- alpha*acc + beta*C (modulo 2^32), beta = 0: C not read
inline void
i8store4(int32x4_t acc, std::int32_t alpha, std::int32_t beta, std::int32_t *C)
{
    uint32x4_t z = vreinterpretq_u32_s32(acc);
    if (alpha != 1) {
	z = vmulq_n_u32(z, std::uint32_t(alpha));
    }
    if (beta == 1) {
	z = vaddq_u32(z, vld1q_u32(reinterpret_cast<std::uint32_t *>(C)));
    } else if (beta != 0) {
	z = vmlaq_n_u32(z, vld1q_u32(reinterpret_cast<std::uint32_t *>(C)),
	                std::uint32_t(beta));
    }
    vst1q_u32(reinterpret_cast<std::uint32_t *>(C), z);
}

#define ULMBLAS_I8_ROW(r)                                                      \
    c##r##0 = I8Dot<UA, UB, 0>::dot(c##r##0, a##r, b0);                        \
    c##r##1 = I8Dot<UA, UB, 1>::dot(c##r##1, a##r, b0);                        \
    c##r##2 = I8Dot<UA, UB, 2>::dot(c##r##2, a##r, b0);                        \
    c##r##3 = I8Dot<UA, UB, 3>::dot(c##r##3, a##r, b0);                        \
    c##r##4 = I8Dot<UA, UB, 0>::dot(c##r##4, a##r, b1);                        \
    c##r##5 = I8Dot<UA, UB, 1>::dot(c##r##5, a##r, b1);                        \
    c##r##6 = I8Dot<UA, UB, 2>::dot(c##r##6, a##r, b1);                        \
    c##r##7 = I8Dot<UA, UB, 3>::dot(c##r##7, a##r, b1);

#define ULMBLAS_I8_ZERO(r)                                                     \
    int32x4_t c##r##0 = vdupq_n_s32(0), c##r##1 = c##r##0, c##r##2 = c##r##0,  \
              c##r##3 = c##r##0, c##r##4 = c##r##0, c##r##5 = c##r##0,         \
              c##r##6 = c##r##0, c##r##7 = c##r##0;

#define ULMBLAS_I8_STORE(j)                                                    \
    i8store4(c0##j, alpha, beta, C + (j) * ldC);                               \
    i8store4(c1##j, alpha, beta, C + (j) * ldC + 4);                           \
    i8store4(c2##j, alpha, beta, C + (j) * ldC + 8);

//  the 12 x 8 tile (all of it: the kernel is not partial); same signs
//  with FEAT_DotProd, mixed signs with FEAT_I8MM
#define ULMBLAS_I8_KERNEL(NAME, TARGET)                                        \
    template <bool UA, bool UB>                                                \
    TARGET void NAME(long kg, std::int32_t alpha, const std::uint8_t *A,       \
                     const std::uint8_t *B, std::int32_t beta,                 \
                     std::int32_t *C, long ldC)                                \
    {                                                                          \
	ULMBLAS_I8_ZERO(0)                                                     \
	ULMBLAS_I8_ZERO(1)                                                     \
	ULMBLAS_I8_ZERO(2)                                                     \
	for (long g = 0; g < kg; ++g) {                                        \
	    const uint8x16_t a0 = vld1q_u8(A);                                 \
	    const uint8x16_t a1 = vld1q_u8(A + 16);                            \
	    const uint8x16_t a2 = vld1q_u8(A + 32);                            \
	    const uint8x16_t b0 = vld1q_u8(B);                                 \
	    const uint8x16_t b1 = vld1q_u8(B + 16);                            \
	    ULMBLAS_I8_ROW(0)                                                  \
	    ULMBLAS_I8_ROW(1)                                                  \
	    ULMBLAS_I8_ROW(2)                                                  \
	    A += 48;                                                           \
	    B += 32;                                                           \
	}                                                                      \
	ULMBLAS_I8_STORE(0)                                                    \
	ULMBLAS_I8_STORE(1)                                                    \
	ULMBLAS_I8_STORE(2)                                                    \
	ULMBLAS_I8_STORE(3)                                                    \
	ULMBLAS_I8_STORE(4)                                                    \
	ULMBLAS_I8_STORE(5)                                                    \
	ULMBLAS_I8_STORE(6)                                                    \
	ULMBLAS_I8_STORE(7)                                                    \
    }

ULMBLAS_I8_KERNEL(i8kernel_12x8_dot, ULMBLAS_NEON_DOT)
ULMBLAS_I8_KERNEL(i8kernel_12x8_i8mm, ULMBLAS_NEON_I8MM)

#undef ULMBLAS_I8_KERNEL
#undef ULMBLAS_I8_ROW
#undef ULMBLAS_I8_ZERO
#undef ULMBLAS_I8_STORE

//  for the driver of level3/i8gemm/driver.h
struct I8Kernel : i8::KernelBase
{
    typedef std::uint8_t P;
    static constexpr long MR = 12, NR = 8, KG = 4;
    static constexpr long MC = 384, KC = 2048, NC = 2048;

    template <bool UA, bool UB>
    static void
    kernel(long kg, std::int32_t alpha, const P *A, const P *B,
           std::int32_t beta, std::int32_t *C, long ldC, long, long)
    {
	run<UA, UB>(std::integral_constant<bool, UA == UB>(), kg, alpha, A, B,
	            beta, C, ldC);
    }

    //  only the variant for the signs is instantiated
    template <bool UA, bool UB>
    static void
    run(std::true_type, long kg, std::int32_t alpha, const P *A, const P *B,
        std::int32_t beta, std::int32_t *C, long ldC)
    {
	i8kernel_12x8_dot<UA, UB>(kg, alpha, A, B, beta, C, ldC);
    }

    template <bool UA, bool UB>
    static void
    run(std::false_type, long kg, std::int32_t alpha, const P *A, const P *B,
        std::int32_t beta, std::int32_t *C, long ldC)
    {
	i8kernel_12x8_i8mm<UA, UB>(kg, alpha, A, B, beta, C, ldC);
    }
};

} // namespace neon
} // namespace ulmBLAS

#undef ULMBLAS_NEON_DOT
#undef ULMBLAS_NEON_I8MM

#endif // ULMBLAS_LEVEL3_UKERNEL_NEON_I8GEMM_TCC
