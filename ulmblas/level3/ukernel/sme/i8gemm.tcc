#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_I8GEMM_TCC
#define ULMBLAS_LEVEL3_UKERNEL_SME_I8GEMM_TCC 1

//
//  ulmBLAS: GEMM with A and B in 8-bit integers (signed or unsigned) and C
//  in int32 for SME2 (SVL = 512 bits, Apple M4, M5):
//
//      C <- alpha*A*B + beta*C     (modulo 2^32)
//
//  The 4-way outer products (SMOPA, UMOPA, SUMOPA, USMOPA into 32-bit
//  tiles) multiply 4 consecutive values of k per instruction, so the
//  panels are packed in groups of 4 values of k: for the panel of 32 rows
//  of A (or 32 columns of B) and the group g
//
//      buffer[128*g + 4*i + q] = A(i, 4*g + q),   0 <= i < 32, 0 <= q < 4,
//
//  zero padded in i and in k.  The 32 x 32 tile of C is held in ZA0.S to
//  ZA3.S as in the float kernel (sgemm.tcc, the operands of the outer
//  products swapped: a horizontal slice is a part of a column of C).  Per
//  group of 4 values of k: one load of 128 bytes of A, one of B, four outer
//  products.
//
//  The driver (blocks, packing) is that of level3/i8gemm/driver.h.
//
#include <arm_sme.h>
#include <cstdint>
#include <ulmblas/level3/ukernel/sme/fpsr.h>

namespace ulmBLAS {
namespace sme {

//
//  The 4-way outer product for the signedness of A (UA: unsigned) and B
//  (UB): zn is the vector of B, zm that of A.
//
template <bool UA, bool UB, unsigned T>
inline void
i8mopa(svbool_t p, svuint8_t b, svuint8_t a) __arm_streaming __arm_inout("za")
{
    if (!UA && !UB) {
	svmopa_za32_s8_m(T, p, p, svreinterpret_s8(b), svreinterpret_s8(a));
    } else if (UA && !UB) {
	svsumopa_za32_s8_m(T, p, p, svreinterpret_s8(b), a);
    } else if (!UA && UB) {
	svusmopa_za32_u8_m(T, p, p, b, svreinterpret_s8(a));
    } else {
	svmopa_za32_u8_m(T, p, p, b, a);
    }
}

//  columns 0 to n-1 (n <= 16) of the tiles T and T+1, see sgemm.tcc
template <unsigned T>
inline void
i32load_32x16(long n, const std::int32_t *C, long ldC, svbool_t p0,
              svbool_t p1) __arm_streaming __arm_inout("za")
{
    for (long c = 0; c < n; ++c) {
	svld1_hor_za32(T + 0, c, p0, C + c * ldC);
	svld1_hor_za32(T + 1, c, p1, C + c * ldC + 16);
    }
}

template <unsigned T>
inline void
i32store_32x16(long n, std::int32_t *C, long ldC, svbool_t p0,
               svbool_t p1) __arm_streaming __arm_in("za")
{
    for (long c = 0; c < n; ++c) {
	svst1_hor_za32(T + 0, c, p0, C + c * ldC);
	svst1_hor_za32(T + 1, c, p1, C + c * ldC + 16);
    }
}

//  C <- alpha*ZA + beta*C (C not read for beta = 0), modulo 2^32
template <unsigned T>
inline void
i32update_32x16(long n, std::uint32_t alpha, std::uint32_t beta,
                std::int32_t *C_, long ldC,
                svcount_t pc) __arm_streaming __arm_in("za")
{
    const svbool_t p = svptrue_b32();
    const svuint32_t zero = svdup_u32(0);
    std::uint32_t *C = reinterpret_cast<std::uint32_t *>(C_);

    for (long c = 0; c < n; ++c) {
	std::uint32_t *Cc = C + c * ldC;

	svuint32_t z0 =
	    svmul_n_u32_x(p, svread_hor_za32_u32_m(zero, p, T, c), alpha);
	svuint32_t z1 =
	    svmul_n_u32_x(p, svread_hor_za32_u32_m(zero, p, T + 1, c), alpha);

	if (beta != 0) {
	    svuint32x2_t x = svld1_u32_x2(pc, Cc);

	    z0 = svmla_n_u32_x(p, z0, svget2(x, 0), beta);
	    z1 = svmla_n_u32_x(p, z1, svget2(x, 1), beta);
	}
	svst1_u32_x2(pc, Cc, svcreate2(z0, z1));
    }
}

//
//  The rows 0 to mr-1 and columns 0 to nr-1 of the 32 x 32 tile of C (unit
//  row stride, column stride ldC); kg groups of 4 values of k.
//
template <bool UA, bool UB>
__arm_new("za") __arm_locally_streaming inline void i8gemm_32x32_streaming(
    long kg, std::int32_t alpha, const std::uint8_t *A, const std::uint8_t *B,
    std::int32_t beta, std::int32_t *C, long ldC, long mr, long nr)
{
    const svbool_t p = svptrue_b8();
    const svcount_t pc = svptrue_c8();
    const bool direct = alpha == 1 && (beta == 0 || beta == 1);

    const int64_t m = mr;
    const svbool_t p0 = svwhilelt_b32_s64(int64_t(0), m);
    const svbool_t p1 = svwhilelt_b32_s64(int64_t(16), m);
    const svcount_t pcr = svwhilelt_c32_s64(int64_t(0), m, 2);
    const long n0 = (nr < 16) ? nr : 16;
    const long n1 = nr - n0;

    svzero_za();
    if (direct && beta == 1) {
	i32load_32x16<0>(n0, C, ldC, p0, p1);
	i32load_32x16<2>(n1, C + 16 * ldC, ldC, p0, p1);
    }

    for (long g = 0; g < kg; ++g) {
	svuint8x2_t a = svld1_u8_x2(pc, A);
	svuint8x2_t b = svld1_u8_x2(pc, B);

	i8mopa<UA, UB, 0>(p, svget2(b, 0), svget2(a, 0));
	i8mopa<UA, UB, 1>(p, svget2(b, 0), svget2(a, 1));
	i8mopa<UA, UB, 2>(p, svget2(b, 1), svget2(a, 0));
	i8mopa<UA, UB, 3>(p, svget2(b, 1), svget2(a, 1));

	A += 128;
	B += 128;
    }

    if (direct) {
	i32store_32x16<0>(n0, C, ldC, p0, p1);
	i32store_32x16<2>(n1, C + 16 * ldC, ldC, p0, p1);
    } else {
	i32update_32x16<0>(n0, std::uint32_t(alpha), std::uint32_t(beta), C,
	                   ldC, pcr);
	i32update_32x16<2>(n1, std::uint32_t(alpha), std::uint32_t(beta),
	                   C + 16 * ldC, ldC, pcr);
    }
}

//
//  16-bit integers (signed), C int32 modulo 2^32: the same tile with the
//  2-way outer products of 16-bit integers (SMOPA into 32-bit tiles, at
//  the rate of FMOPA for float), panels in groups of 2 values of k.
//
__arm_new("za") __arm_locally_streaming inline void i16gemm_32x32_streaming(
    long kg, std::int32_t alpha, const std::int16_t *A, const std::int16_t *B,
    std::int32_t beta, std::int32_t *C, long ldC, long mr, long nr)
{
    const svbool_t p = svptrue_b16();
    const svcount_t pc = svptrue_c16();
    const bool direct = alpha == 1 && (beta == 0 || beta == 1);

    const int64_t m = mr;
    const svbool_t p0 = svwhilelt_b32_s64(int64_t(0), m);
    const svbool_t p1 = svwhilelt_b32_s64(int64_t(16), m);
    const svcount_t pcr = svwhilelt_c32_s64(int64_t(0), m, 2);
    const long n0 = (nr < 16) ? nr : 16;
    const long n1 = nr - n0;

    svzero_za();
    if (direct && beta == 1) {
	i32load_32x16<0>(n0, C, ldC, p0, p1);
	i32load_32x16<2>(n1, C + 16 * ldC, ldC, p0, p1);
    }

    for (long g = 0; g < kg; ++g) {
	svint16x2_t a = svld1_s16_x2(pc, A);
	svint16x2_t b = svld1_s16_x2(pc, B);

	svmopa_za32_s16_m(0, p, p, svget2(b, 0), svget2(a, 0));
	svmopa_za32_s16_m(1, p, p, svget2(b, 0), svget2(a, 1));
	svmopa_za32_s16_m(2, p, p, svget2(b, 1), svget2(a, 0));
	svmopa_za32_s16_m(3, p, p, svget2(b, 1), svget2(a, 1));

	A += 64;
	B += 64;
    }

    if (direct) {
	i32store_32x16<0>(n0, C, ldC, p0, p1);
	i32store_32x16<2>(n1, C + 16 * ldC, ldC, p0, p1);
    } else {
	i32update_32x16<0>(n0, std::uint32_t(alpha), std::uint32_t(beta), C,
	                   ldC, pcr);
	i32update_32x16<2>(n1, std::uint32_t(alpha), std::uint32_t(beta),
	                   C + 16 * ldC, ldC, pcr);
    }
}

struct I16Kernel : i8::KernelBase
{
    typedef std::int16_t P;
    static constexpr long MR = 32, NR = 32, KG = 2;
    static constexpr long MC = 512, KC = 2048, NC = 1024;
    static constexpr bool partial = true;

    template <bool UA, bool UB>
    static void
    kernel(long kg, std::int32_t alpha, const P *A, const P *B,
           std::int32_t beta, std::int32_t *C, long ldC, long mr, long nr)
    {
	const FpsrRestore fpsr; // see sme/fpsr.h
	i16gemm_32x32_streaming(kg, alpha, A, B, beta, C, ldC, mr, nr);
    }
};

//
//  The kernel for the driver of level3/i8gemm/driver.h (packing in groups
//  of 4 bytes, the layout above).
//
struct I8Kernel : i8::KernelBase
{
    typedef std::uint8_t P;
    static constexpr long MR = 32, NR = 32, KG = 4;
    static constexpr long MC = 512, KC = 4096, NC = 2048;
    static constexpr bool partial = true;

    template <bool UA, bool UB>
    static void
    kernel(long kg, std::int32_t alpha, const P *A, const P *B,
           std::int32_t beta, std::int32_t *C, long ldC, long mr, long nr)
    {
	const FpsrRestore fpsr; // see sme/fpsr.h
	i8gemm_32x32_streaming<UA, UB>(kg, alpha, A, B, beta, C, ldC, mr, nr);
    }
};

} // namespace sme
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_I8GEMM_TCC
