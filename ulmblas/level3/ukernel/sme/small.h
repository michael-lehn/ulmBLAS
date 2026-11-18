#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_SMALL_H
#define ULMBLAS_LEVEL3_UKERNEL_SME_SMALL_H 1

#include <algorithm>
#include <arm_sme.h>
#include <cstddef>
#include <cstdint>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/inline.h>
#include <ulmblas/level3/ukernel/sme/fpsr.h>

//
//  DGEMM for small matrices with the SME2 tiles (Apple M4, M5): the
//  variant v8s of the experiments (small_matrix_experiments/, branch
//  study/small-matrix) with the hoisted row predicates of v8sq.
//
//      C <- beta C + alpha A B,   A m x k, B k x n, C m x n
//
//  One streaming section per call; A in place (columns of 8*MT rows loaded
//  with one multi-vector load), B packed in slivers of 16 columns (two
//  halves of 8 columns), C moved directly into and out of the ZA tiles.
//  C in blocks of 64 columns, tiles of 32 x 16 elements (all eight ZA
//  tiles), flat tiles 16 x 32 and 8 x 64 for the last rows; no blocking of
//  k.
//
//  The tiles: plain loops over the ZA tiles with a constant number of
//  iterations, unrolled (#pragma clang loop unroll(full)), and helpers
//  (za_load, za_store, za_read, za_mopa, get, pick) that select the
//  instruction for a tile number with a switch: the intrinsics require a
//  constant tile number, after the unrolling it is one, and the switch
//  disappears.
//
//  Strides: A, B and C with unit row stride are used as they are; others
//  (any strides, also row major) are copied into column-major buffers
//  first (C also back).  The leading dimensions and m, n, k must be below
//  2^32 (otherwise copied as well).  run (the plan) calls dgemm on blocks
//  so that all buffers are bounded by constants.
//
namespace ulmBLAS {
namespace sme {
namespace small {

//  B(0:k-1, 0:nr-1) (nr <= 8) -> half sliver of 8 columns at Bp (row l:
//  B(l, 0:7), 64 bytes; columns beyond nr zero).  Blocks of 32 rows with
//  four ZA tiles: columns loaded into horizontal slices, rows read as
//  vertical slices four at a time and stored as 4 consecutive vectors.
ULMBLAS_ALWAYS_INLINE void
pack8(std::int64_t k, std::int64_t nr, const double *B, std::int64_t ldB,
      double *Bp) __arm_streaming __arm_inout("za")
{
    const svbool_t p = svptrue_b64();
    const svcount_t pc = svptrue_c64();
    std::int64_t l = 0;

    if (nr == 8) {
	for (; l + 32 <= k; l += 32) {
	    for (std::int64_t j = 0; j < 8; ++j) {
		const double *b = B + l + j * ldB;
		svld1_hor_za64(0, j, p, b);
		svld1_hor_za64(1, j, p, b + 8);
		svld1_hor_za64(2, j, p, b + 16);
		svld1_hor_za64(3, j, p, b + 24);
	    }
	    double *d = Bp + l * 8;
	    for (std::int64_t i = 0; i < 8; i += 4) {
		svst1_f64_x4(pc, d + i * 8, svread_ver_za64_f64_vg4(0, i));
		svst1_f64_x4(pc, d + 64 + i * 8, svread_ver_za64_f64_vg4(1, i));
		svst1_f64_x4(pc, d + 128 + i * 8,
		             svread_ver_za64_f64_vg4(2, i));
		svst1_f64_x4(pc, d + 192 + i * 8,
		             svread_ver_za64_f64_vg4(3, i));
	    }
	}
    }
    for (; l < k; l += 8) {
	const svbool_t pl = svwhilelt_b64_s64(l, k);
	const std::int64_t kl = (k - l < 8) ? k - l : 8;

	svzero_mask_za(0x01);
	for (std::int64_t j = 0; j < nr; ++j) {
	    svld1_hor_za64(0, j, pl, B + l + j * ldB);
	}
	for (std::int64_t i = 0; i < kl; ++i) {
	    svst1_ver_za64(0, i, p, Bp + (l + i) * 8);
	}
    }
}

//  B(0:k-1, 0:nr-1) (nr <= 16) -> sliver of 16 columns at Bp: two half
//  slivers of 8 columns, at Bp and Bp + k*8
ULMBLAS_ALWAYS_INLINE void
pack_B(std::int64_t k, std::int64_t nr, const double *B, std::int64_t ldB,
       double *Bp) __arm_streaming __arm_inout("za")
{
    pack8(k, (nr < 8) ? nr : 8, B, ldB, Bp);
    pack8(k, (nr > 8) ? nr - 8 : 0, B + 8 * ldB, ldB, Bp + k * 8);
}

ULMBLAS_ALWAYS_INLINE void
za_load(std::uint32_t t, std::int64_t c, svbool_t pg,
        const double *x) __arm_streaming __arm_inout("za")
{
    switch (t) {
    case 0:
	svld1_hor_za64(0, c, pg, x);
	break;
    case 1:
	svld1_hor_za64(1, c, pg, x);
	break;
    case 2:
	svld1_hor_za64(2, c, pg, x);
	break;
    case 3:
	svld1_hor_za64(3, c, pg, x);
	break;
    case 4:
	svld1_hor_za64(4, c, pg, x);
	break;
    case 5:
	svld1_hor_za64(5, c, pg, x);
	break;
    case 6:
	svld1_hor_za64(6, c, pg, x);
	break;
    case 7:
	svld1_hor_za64(7, c, pg, x);
	break;
    }
}

ULMBLAS_ALWAYS_INLINE void
za_store(std::uint32_t t, std::int64_t c, svbool_t pg,
         double *x) __arm_streaming __arm_in("za")
{
    switch (t) {
    case 0:
	svst1_hor_za64(0, c, pg, x);
	break;
    case 1:
	svst1_hor_za64(1, c, pg, x);
	break;
    case 2:
	svst1_hor_za64(2, c, pg, x);
	break;
    case 3:
	svst1_hor_za64(3, c, pg, x);
	break;
    case 4:
	svst1_hor_za64(4, c, pg, x);
	break;
    case 5:
	svst1_hor_za64(5, c, pg, x);
	break;
    case 6:
	svst1_hor_za64(6, c, pg, x);
	break;
    case 7:
	svst1_hor_za64(7, c, pg, x);
	break;
    }
}

//  horizontal slice c of tile t
ULMBLAS_ALWAYS_INLINE svfloat64_t
za_read(std::uint32_t t, std::int64_t c) __arm_streaming __arm_in("za")
{
    const svbool_t p = svptrue_b64();
    const svfloat64_t z = svdup_f64(0.0);
    switch (t) {
    case 0:
	return svread_hor_za64_f64_m(z, p, 0, c);
    case 1:
	return svread_hor_za64_f64_m(z, p, 1, c);
    case 2:
	return svread_hor_za64_f64_m(z, p, 2, c);
    case 3:
	return svread_hor_za64_f64_m(z, p, 3, c);
    case 4:
	return svread_hor_za64_f64_m(z, p, 4, c);
    case 5:
	return svread_hor_za64_f64_m(z, p, 5, c);
    case 6:
	return svread_hor_za64_f64_m(z, p, 6, c);
    default:
	return svread_hor_za64_f64_m(z, p, 7, c);
    }
}

//  tile t += x * y^T (outer product)
ULMBLAS_ALWAYS_INLINE void
za_mopa(std::uint32_t t, svbool_t p, svfloat64_t x,
        svfloat64_t y) __arm_streaming __arm_inout("za")
{
    switch (t) {
    case 0:
	svmopa_za64_f64_m(0, p, p, x, y);
	break;
    case 1:
	svmopa_za64_f64_m(1, p, p, x, y);
	break;
    case 2:
	svmopa_za64_f64_m(2, p, p, x, y);
	break;
    case 3:
	svmopa_za64_f64_m(3, p, p, x, y);
	break;
    case 4:
	svmopa_za64_f64_m(4, p, p, x, y);
	break;
    case 5:
	svmopa_za64_f64_m(5, p, p, x, y);
	break;
    case 6:
	svmopa_za64_f64_m(6, p, p, x, y);
	break;
    case 7:
	svmopa_za64_f64_m(7, p, p, x, y);
	break;
    }
}

//  vector i of a tuple of 4 or 2 vectors (the index is part of the
//  instruction as well)
ULMBLAS_ALWAYS_INLINE svfloat64_t
get(svfloat64x4_t v, std::uint32_t i) __arm_streaming __arm_preserves("za")
{
    switch (i) {
    case 0:
	return svget4(v, 0);
    case 1:
	return svget4(v, 1);
    case 2:
	return svget4(v, 2);
    default:
	return svget4(v, 3);
    }
}

ULMBLAS_ALWAYS_INLINE svfloat64_t
get(svfloat64x2_t v, std::uint32_t i) __arm_streaming __arm_preserves("za")
{
    return (i == 0) ? svget2(v, 0) : svget2(v, 1);
}

ULMBLAS_ALWAYS_INLINE svfloat64_t
get(svfloat64_t v, std::uint32_t i) __arm_streaming __arm_preserves("za")
{
    return v;
}

//  rows 8*i .. 8*i+7 of a tile with mr rows in C
ULMBLAS_ALWAYS_INLINE svbool_t
rows(std::uint32_t i, std::int64_t mr) __arm_streaming __arm_preserves("za")
{
    return svwhilelt_b64_s64(8 * std::int64_t(i), mr);
}

//  predicate i of r0 .. r3 (i a constant after unrolling)
ULMBLAS_ALWAYS_INLINE svbool_t
pick(std::uint32_t i, svbool_t r0, svbool_t r1, svbool_t r2,
     svbool_t r3) __arm_streaming __arm_preserves("za")
{
    switch (i) {
    case 0:
	return r0;
    case 1:
	return r1;
    case 2:
	return r2;
    default:
	return r3;
    }
}

//  MT vectors (8*MT rows) of a column of A in one load: a tuple of 4 or 2
//  vectors, or a single vector; the rows mr ... are zero
template <std::uint32_t MT> struct ColumnA;

template <> struct ColumnA<4>
{
    static ULMBLAS_ALWAYS_INLINE svfloat64x4_t
    load(std::int64_t mr, const double *a) __arm_streaming __arm_preserves("za")
    {
	return svld1_f64_x4(svwhilelt_c64_s64(std::int64_t(0), mr, 4), a);
    }
};

template <> struct ColumnA<2>
{
    static ULMBLAS_ALWAYS_INLINE svfloat64x2_t
    load(std::int64_t mr, const double *a) __arm_streaming __arm_preserves("za")
    {
	return svld1_f64_x2(svwhilelt_c64_s64(std::int64_t(0), mr, 2), a);
    }
};

template <> struct ColumnA<1>
{
    static ULMBLAS_ALWAYS_INLINE svfloat64_t
    load(std::int64_t mr, const double *a) __arm_streaming __arm_preserves("za")
    {
	return svld1_f64(rows(0, mr), a);
    }
};

//
//  Tile of C with MT row vectors (8*MT rows) and NT column groups (8*NT
//  columns), MT*NT = 8.  ZA tile t = i + MT*g holds rows 8*i .. 8*i+7 of
//  the columns 8*g .. 8*g+7 (transposed: horizontal slice c of the tile is
//  a part of column 8*g+c of C).  mr rows and nr columns of the tile are in
//  C.  Bp: the packed sliver of the first 16 columns of the tile.  NS: the
//  number of slivers used in the loop over k (<= NT/2, the others are
//  beyond n).  FULL: all 8*NT columns of the tile are in C (nr = 8*NT);
//  then the number of slices is a constant and the loops over them are
//  unrolled.
//
template <std::uint32_t MT, std::uint32_t NT, std::uint32_t NS, bool FULL>
ULMBLAS_ALWAYS_INLINE void
tile(std::int64_t mr, std::int64_t nr, std::int64_t k, double alpha,
     const double *A, std::int64_t ldA, const double *Bp, double beta,
     double *C, std::int64_t ldC) __arm_streaming __arm_inout("za")
{
    static_assert(MT * NT == 8 && 2 * NS <= NT, "eight ZA tiles");

    const svbool_t p = svptrue_b64();
    const bool direct = alpha == 1.0 && (beta == 0.0 || beta == 1.0);

    //  The slices of C are moved slice c of all tiles, then slice c+1:
    //  consecutive accesses to different tiles (one tile after the other
    //  was slower).  Slice c of tile t = i + MT*g is a part of column
    //  8*g+c of C (rows 8*i ...).  Of a tile at the right border of C only
    //  the columns in C (nr < 8*NT).
    const std::int64_t c8 = FULL ? 8 : (nr < 8) ? nr : 8;

    //  the rows of the tile in C, once for all slices (computed in the loops
    //  over the slices the compiler recomputed them for each slice)
    const svbool_t r0 = rows(0, mr), r1 = rows(1, mr), r2 = rows(2, mr),
                   r3 = rows(3, mr);

    svzero_za();
    if (direct && beta == 1.0) {
	for (std::int64_t c = 0; c < c8; ++c) {
#pragma clang loop unroll(full)
	    for (std::uint32_t g = 0; g < NT; ++g) {
		if (!FULL && 8 * std::int64_t(g) + c >= nr) {
		    continue;
		}
#pragma clang loop unroll(full)
		for (std::uint32_t i = 0; i < MT; ++i) {
		    za_load(i + MT * g, c,
		            FULL ? rows(i, mr) : pick(i, r0, r1, r2, r3),
		            C + 8 * i + (8 * g + c) * ldC);
		}
	    }
	}
    }

    const double *a = A;
    const double *b = Bp;

    //  the loop over k starts at a cache line (its body of 15 to 21
    //  instructions then needs one 64 byte line for the 32 x 16 tile, two
    //  otherwise)
    [[clang::code_align(64)]] for (std::int64_t l = 0; l < k;
                                   ++l, a += ldA, b += 8) {
	//  column l of the tile of A: MT vectors in one load
	const auto va = ColumnA<MT>::load(mr, a);
	//  row l of the slivers: two vectors (half slivers) per sliver
#pragma clang loop unroll(full)
	for (std::uint32_t s = 0; s < NS; ++s) {
	    const svfloat64_t b0 = svld1_f64(p, b + 2 * s * k * 8);
	    const svfloat64_t b1 = svld1_f64(p, b + (2 * s + 1) * k * 8);
#pragma clang loop unroll(full)
	    for (std::uint32_t i = 0; i < MT; ++i) {
		za_mopa(i + MT * (2 * s), p, b0, get(va, i));
	    }
#pragma clang loop unroll(full)
	    for (std::uint32_t i = 0; i < MT; ++i) {
		za_mopa(i + MT * (2 * s + 1), p, b1, get(va, i));
	    }
	}
    }

    for (std::int64_t c = 0; c < c8; ++c) {
#pragma clang loop unroll(full)
	for (std::uint32_t g = 0; g < NT; ++g) {
	    if (!FULL && 8 * std::int64_t(g) + c >= nr) {
		continue;
	    }
#pragma clang loop unroll(full)
	    for (std::uint32_t i = 0; i < MT; ++i) {
		const std::uint32_t t = i + MT * g;
		const svbool_t pr =
		    FULL ? rows(i, mr) : pick(i, r0, r1, r2, r3);
		double *Cc = C + 8 * i + (8 * g + c) * ldC;

		if (direct) {
		    za_store(t, c, pr, Cc);
		} else {
		    svfloat64_t z = svmul_x(p, za_read(t, c), alpha);
		    if (beta != 0.0) {
			z = svmla_x(p, z, svld1_f64(pr, Cc), beta);
		    }
		    svst1_f64(pr, Cc, z);
		}
	    }
	}
    }
}

//  the tiles of one row panel (8*MT rows) of C, at most NT*8 columns each
template <std::uint32_t MT, std::uint32_t NT>
ULMBLAS_ALWAYS_INLINE void
panel(std::int64_t mr, std::int64_t n, std::int64_t k, double alpha,
      const double *A, std::int64_t ldA, const double *Bp, double beta,
      double *C, std::int64_t ldC) __arm_streaming __arm_inout("za")
{
    const std::int64_t W = 8 * NT;

    for (std::int64_t j = 0; j < n; j += W) {
	const std::int64_t nr = (n - j < W) ? n - j : W;
	const std::int64_t ns = (nr + 15) / 16;
	const double *b = Bp + j * k;
	double *Cj = C + j * ldC;

	if (nr == W) {
	    tile<MT, NT, NT / 2, true>(mr, nr, k, alpha, A, ldA, b, beta, Cj,
	                               ldC);
	} else if (ns == NT / 2) {
	    tile<MT, NT, NT / 2, false>(mr, nr, k, alpha, A, ldA, b, beta, Cj,
	                                ldC);
	} else if (NT >= 4 && ns == 1) {
	    tile<MT, NT, 1, false>(mr, nr, k, alpha, A, ldA, b, beta, Cj, ldC);
	} else if (NT == 8 && ns == 2) {
	    tile<MT, NT, (NT == 8 ? 2 : 1), false>(mr, nr, k, alpha, A, ldA, b,
	                                           beta, Cj, ldC);
	} else if (NT == 8) {
	    tile<MT, NT, (NT == 8 ? 3 : 1), false>(mr, nr, k, alpha, A, ldA, b,
	                                           beta, Cj, ldC);
	}
    }
}

__attribute__((noinline)) inline void
streaming(std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
          const double *A, std::int64_t ldA, const double *B, std::int64_t ldB,
          double beta, double *C, std::int64_t ldC,
          double *Bp) __arm_streaming __arm_inout("za")
{
    const std::int64_t m32 = m / 32 * 32;
    const std::int64_t mt = m - m32;
    const double *At = A + m32;

    for (std::int64_t j = 0; j < n; j += 64) {
	const std::int64_t nb = (n - j < 64) ? n - j : 64;
	double *Cj = C + j * ldC;

	//  jp: first column of the panel of B (16 columns) in the block
	for (std::int64_t jp = 0; jp < nb; jp += 16) {
	    const std::int64_t nr = (nb - jp < 16) ? nb - jp : 16;
	    pack_B(k, nr, B + (j + jp) * ldB, ldB, Bp + jp * k);
	    for (std::int64_t i = 0; i < m32; i += 32) {
		if (nr == 16) {
		    tile<4, 2, 1, true>(32, nr, k, alpha, A + i, ldA,
		                        Bp + jp * k, beta, Cj + i + jp * ldC,
		                        ldC);
		} else {
		    tile<4, 2, 1, false>(32, nr, k, alpha, A + i, ldA,
		                         Bp + jp * k, beta, Cj + i + jp * ldC,
		                         ldC);
		}
	    }
	}
	if (mt > 24) {
	    panel<4, 2>(mt, nb, k, alpha, At, ldA, Bp, beta, Cj + m32, ldC);
	} else if (mt > 16) {
	    panel<2, 4>(16, nb, k, alpha, At, ldA, Bp, beta, Cj + m32, ldC);
	    panel<1, 8>(mt - 16, nb, k, alpha, At + 16, ldA, Bp, beta,
	                Cj + m32 + 16, ldC);
	} else if (mt > 8) {
	    panel<2, 4>(mt, nb, k, alpha, At, ldA, Bp, beta, Cj + m32, ldC);
	} else if (mt > 0) {
	    panel<1, 8>(mt, nb, k, alpha, At, ldA, Bp, beta, Cj + m32, ldC);
	}
    }
}

//
//  The entry into streaming mode: 8 integer arguments in registers (m and
//  n, k and ldA, ldB and ldC packed in pairs; ab: 0 for alpha = 1, beta =
//  0; 1 for alpha = 1, beta = 1; otherwise the address of {alpha, beta}).
//  Arguments on the stack would be written in ordinary mode and read in
//  streaming mode right after (about 30 ns per cache line).
//
__arm_new("za") __arm_locally_streaming
    __attribute__((noinline, no_stack_protector)) inline void entry(
        const double *A, const double *B, double *C, double *Bp,
        std::uint64_t mn, std::uint64_t kldA, std::uint64_t ldBC,
        std::uintptr_t ab)
{
    double alpha = 1.0, beta = double(ab);
    if (ab > 1) {
	const double *p = reinterpret_cast<const double *>(ab);
	alpha = p[0];
	beta = p[1];
    }
    //  A fresh stack area for the streaming code, aligned to a cache line
    //  (128 bytes on Apple cores): without it the speed for small matrices
    //  depended on where the stack of the program started (n = 16: 60 or
    //  38 GFLOP/s).  The unused array realigns the frame.
    //  no_stack_protector (see the attributes): a stack protector would
    //  call __stack_chk_fail, and a call with live ZA needs a lazy save
    //  buffer.
    alignas(128) std::uint64_t gap[32];
    __asm__ volatile("" : : "r"(gap) : "memory");
    streaming(std::uint32_t(mn), std::int64_t(mn >> 32), std::uint32_t(kldA),
              alpha, A, std::int64_t(kldA >> 32), B, std::uint32_t(ldBC), beta,
              C, std::int64_t(ldBC >> 32), Bp);
}

//
//  C <- beta C + alpha A B with arbitrary strides (m, n, k > 0, alpha !=
//  0).  Operands with a row stride other than 1 (or dimensions of 2^32 and
//  more) are copied into column-major buffers.
//
template <typename IndexType>
void
dgemm(IndexType m, IndexType n, IndexType k, double alpha, const double *A,
      IndexType incRowA, IndexType incColA, const double *B, IndexType incRowB,
      IndexType incColB, double beta, double *C, IndexType incRowC,
      IndexType incColC)
{
    static MemoryPool<double> memoryPool;

    constexpr std::int64_t M = std::int64_t(1) << 32;
    auto fits = [](IndexType incRow, IndexType incCol, IndexType cols) {
	return incRow == IndexType(1) && incCol >= IndexType(1) &&
	       std::int64_t(incCol) < M && std::int64_t(cols) < M;
    };
    const bool directA = fits(incRowA, incColA, k) && std::int64_t(m) < M;
    const bool directB = fits(incRowB, incColB, n) && std::int64_t(k) < M;
    const bool directC = fits(incRowC, incColC, n);

    double *Bp = memoryPool.allocate(std::size_t(64) * std::size_t(k));
    double *Ab = directA ? nullptr
                         : memoryPool.allocate(std::size_t(m) * std::size_t(k));
    double *Bb = directB ? nullptr
                         : memoryPool.allocate(std::size_t(k) * std::size_t(n));
    double *Cb = directC ? nullptr
                         : memoryPool.allocate(std::size_t(m) * std::size_t(n));

    std::int64_t ldA = incColA, ldB = incColB, ldC = incColC;
    if (!directA) {
	for (IndexType l = 0; l < k; ++l) {
	    for (IndexType i = 0; i < m; ++i) {
		Ab[i + l * m] = A[i * incRowA + l * incColA];
	    }
	}
	A = Ab;
	ldA = m;
    }
    if (!directB) {
	for (IndexType j = 0; j < n; ++j) {
	    for (IndexType l = 0; l < k; ++l) {
		Bb[l + j * k] = B[l * incRowB + j * incColB];
	    }
	}
	B = Bb;
	ldB = k;
    }
    double *Cd = C;
    if (!directC) {
	for (IndexType j = 0; j < n; ++j) {
	    for (IndexType i = 0; i < m; ++i) {
		Cb[i + j * m] =
		    beta != 0.0 ? C[i * incRowC + j * incColC] : 0.0;
	    }
	}
	Cd = Cb;
	ldC = m;
    }

    const double ab[2] = {alpha, beta};
    std::uintptr_t code = reinterpret_cast<std::uintptr_t>(ab);
    if (alpha == 1.0 && beta == 0.0) {
	code = 0;
    } else if (alpha == 1.0 && beta == 1.0) {
	code = 1;
    }
    auto pair = [](std::int64_t lo, std::int64_t hi) {
	return std::uint64_t(std::uint32_t(lo)) | (std::uint64_t(hi) << 32);
    };
    {
	const FpsrRestore fpsr; // see sme/fpsr.h
	entry(A, B, Cd, Bp, pair(m, n), pair(k, ldA), pair(ldB, ldC), code);
    }

    if (!directC) {
	for (IndexType j = 0; j < n; ++j) {
	    for (IndexType i = 0; i < m; ++i) {
		C[i * incRowC + j * incColC] = Cb[i + j * m];
	    }
	}
	memoryPool.release(Cb);
    }
    if (Bb) {
	memoryPool.release(Bb);
    }
    if (Ab) {
	memoryPool.release(Ab);
    }
    memoryPool.release(Bp);
}

//
//  The kernel as a plan of gemm (ulmblas/level3/gemm/select.h): double
//  without conjugation with A and B column major (unit row strides),
//  otherwise the frame algorithm Fallback.  For A or B row major the copies
//  cost more than the kernel gains (benchmark matrix on the M5: NT, TN, TT
//  down to 0.12 of OpenBLAS, NN at least 0.9); layout variants of the
//  kernel will take these.
//
template <typename IndexType, typename Alpha, typename TA, typename TB,
          typename Beta, typename TC>
inline bool
run(IndexType m, IndexType n, IndexType k, const Alpha &alpha, bool conjA,
    const TA *A, IndexType incRowA, IndexType incColA, bool conjB, const TB *B,
    IndexType incRowB, IndexType incColB, const Beta &beta, TC *C,
    IndexType incRowC, IndexType incColC)
{
    return false;
}

template <typename IndexType>
inline bool
run(IndexType m, IndexType n, IndexType k, const double &alpha, bool conjA,
    const double *A, IndexType incRowA, IndexType incColA, bool conjB,
    const double *B, IndexType incRowB, IndexType incColB, const double &beta,
    double *C, IndexType incRowC, IndexType incColC)
{
    //  Bounded buffers: k in blocks of KMax (packed B: 64 KMax elements),
    //  the columns of C in blocks of at most CopyMax elements if C has to
    //  be copied (row stride other than 1); A and B are never copied (the
    //  fallback for m above CopyMax and leading dimensions of 2^32 and
    //  more).  The blocks after the first in k add to C (beta = 1).
    constexpr std::int64_t KMax = 2048;
    constexpr std::int64_t CopyMax = std::int64_t(1) << 18;
    constexpr std::int64_t M = std::int64_t(1) << 32;
    if (conjA || conjB || incRowA != IndexType(1) || incRowB != IndexType(1) ||
        std::int64_t(m) > CopyMax || incColA < IndexType(1) ||
        incColB < IndexType(1) || std::int64_t(incColA) >= M ||
        std::int64_t(incColB) >= M) {
	return false;
    }
    const bool directC = incRowC == IndexType(1) && incColC >= IndexType(1) &&
                         std::int64_t(incColC) < M;
    const IndexType nc =
        directC ? IndexType(CopyMax)
	        : std::max(IndexType(1), IndexType(CopyMax / std::int64_t(m)));
    for (IndexType j = 0; j < n; j += nc) {
	const IndexType nb = std::min(nc, n - j);
	for (IndexType l = 0; l < k; l += IndexType(KMax)) {
	    const IndexType kb = std::min(IndexType(KMax), k - l);
	    dgemm(m, nb, kb, alpha, &A[l * incColA], incRowA, incColA,
	          &B[l + j * incColB], incRowB, incColB, l == 0 ? beta : 1.0,
	          &C[j * incColC], incRowC, incColC);
	}
    }
    return true;
}

} // namespace small

template <typename Fallback> struct SmallGemm
{
    //  the frame for the other problems (also for the operations without
    //  a kernel for small matrices, see GemmFrameOf, gemm/plan.h)
    typedef Fallback FallbackType;

    template <typename IndexType, typename Alpha, typename TA, typename TB,
              typename Beta, typename TC>
    static void
    run(IndexType m, IndexType n, IndexType k, const Alpha &alpha, bool conjA,
        const TA *A, IndexType incRowA, IndexType incColA, bool conjB,
        const TB *B, IndexType incRowB, IndexType incColB, const Beta &beta,
        TC *C, IndexType incRowC, IndexType incColC)
    {
	if (!small::run(m, n, k, alpha, conjA, A, incRowA, incColA, conjB, B,
	                incRowB, incColB, beta, C, incRowC, incColC)) {
	    Fallback::template run<IndexType, Alpha, TA, TB, Beta, TC>(
	        m, n, k, alpha, conjA, A, incRowA, incColA, conjB, B, incRowB,
	        incColB, beta, C, incRowC, incColC);
	}
    }
};

} // namespace sme
} // namespace ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_SMALL_H
