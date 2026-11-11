#ifndef ULMBLAS_AUXILIARY_FLOAT16_H
#define ULMBLAS_AUXILIARY_FLOAT16_H 1

#include <cstdint>
#include <cstring>

//
//  float16: IEEE binary16 (1 sign, 5 exponent, 10 fraction bits), stored
//  as std::uint16_t as in the C interface (ulmblas_float16).  The
//  conversion to float is exact (also for subnormals, Inf, NaN); the
//  conversion from float rounds to nearest even (overflow to Inf, NaN
//  stays NaN).  As bfloat16 (bfloat16.h) it converts implicitly to float,
//  so that the templates of ulmBLAS compute in float.
//
namespace ulmBLAS {

struct float16
{
    std::uint16_t bits;

    float16() = default;

    float16(float x)
    {
	std::uint32_t u;
	std::memcpy(&u, &x, sizeof(u));
	const std::uint32_t sign = (u >> 16) & 0x8000u;
	const std::uint32_t a = u & 0x7fffffffu;

	if (a > 0x7f800000u) { // NaN (quiet)
	    bits = std::uint16_t(sign | 0x7e00u | ((a >> 13) & 0x3ffu));
	} else if (a >= 0x477ff000u) { // >= 65520: rounds to Inf
	    bits = std::uint16_t(sign | 0x7c00u);
	} else if (a >= 0x38800000u) {         // normal: >= 2^-14
	    std::uint32_t r = a - 0x38000000u; // rebias exponent 127 -> 15
	    r += 0xfffu + ((r >> 13) & 1u);
	    bits = std::uint16_t(sign | (r >> 13));
	} else if (a >= 0x33000001u) {  // subnormal (or rounds to 2^-14)
	    const int e = int(a >> 23); // 102 ... 112
	    const std::uint32_t m = (a & 0x7fffffu) | 0x800000u;
	    const int shift = 126 - e; // 14 ... 24
	    std::uint32_t r = m >> shift;
	    const std::uint32_t rest = m & ((1u << shift) - 1u);
	    const std::uint32_t half = 1u << (shift - 1);
	    if (rest > half || (rest == half && (r & 1u))) {
		++r;
	    }
	    bits = std::uint16_t(sign | r);
	} else { // |x| <= 2^-25: rounds to zero
	    bits = std::uint16_t(sign);
	}
    }

    operator float() const
    {
	const std::uint32_t sign = std::uint32_t(bits & 0x8000u) << 16;
	const std::uint32_t e = (bits >> 10) & 0x1fu;
	const std::uint32_t m = bits & 0x3ffu;
	std::uint32_t u;

	if (e == 0) { // zero or subnormal: m * 2^-24, exact in float
	    float x = float(m) * 5.9604644775390625e-8f;
	    std::memcpy(&u, &x, sizeof(u));
	    u |= sign;
	} else if (e == 0x1fu) { // Inf, NaN
	    u = sign | 0x7f800000u | (m << 13);
	} else {
	    u = sign | ((e + 112u) << 23) | (m << 13);
	}
	float x;
	std::memcpy(&x, &u, sizeof(x));
	return x;
    }
};

static_assert(sizeof(float16) == 2, "float16 must have 16 bits");

inline const float16 &
conjugate(const float16 &x)
{
    return x;
}

inline const float16 &
conjugate(const float16 &x, bool)
{
    return x;
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_FLOAT16_H
