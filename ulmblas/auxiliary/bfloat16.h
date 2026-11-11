#ifndef ULMBLAS_AUXILIARY_BFLOAT16_H
#define ULMBLAS_AUXILIARY_BFLOAT16_H 1

#include <cstdint>
#include <cstring>

//
//  bfloat16: the upper 16 bits of an IEEE float (1 sign, 8 exponent, 7
//  fraction bits), stored as std::uint16_t as in the C interface
//  (ulmblas_bfloat16).  The conversion to float is exact; the conversion
//  from float rounds to nearest even (NaN stays NaN).  Arithmetic is done
//  in float: bfloat16 converts implicitly, so that the templates of
//  ulmBLAS compute in float with bfloat16 operands (e.g. gemm with A and B
//  in bfloat16 and C in float: the panels are packed as float).
//
namespace ulmBLAS {

struct bfloat16
{
    std::uint16_t bits;

    bfloat16() = default;

    bfloat16(float x)
    {
	std::uint32_t u;
	std::memcpy(&u, &x, sizeof(u));
	if ((u & 0x7fffffffu) > 0x7f800000u) {
	    bits = std::uint16_t((u >> 16) | 0x40u); // quiet NaN
	} else {
	    u += 0x7fffu + ((u >> 16) & 1u);
	    bits = std::uint16_t(u >> 16);
	}
    }

    operator float() const
    {
	std::uint32_t u = std::uint32_t(bits) << 16;
	float x;
	std::memcpy(&x, &u, sizeof(x));
	return x;
    }
};

static_assert(sizeof(bfloat16) == 2, "bfloat16 must have 16 bits");

inline const bfloat16 &
conjugate(const bfloat16 &x)
{
    return x;
}

inline const bfloat16 &
conjugate(const bfloat16 &x, bool)
{
    return x;
}

} // namespace ulmBLAS

#endif // ULMBLAS_AUXILIARY_BFLOAT16_H
