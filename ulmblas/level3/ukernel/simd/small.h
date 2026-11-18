#ifndef ULMBLAS_LEVEL3_UKERNEL_SIMD_SMALL_H
#define ULMBLAS_LEVEL3_UKERNEL_SIMD_SMALL_H 1

//
//  GEMM for small matrices with vector registers, without packing:
//
//      C <- beta C + alpha op(A) op(B),   op(A) m x k, op(B) k x n
//
//  for float, double, complex float and complex double, with any strides.
//  The algorithm is the same for every instruction set; the instruction set
//  provides the vector operations (Vec, CVec below) and is selected by a tag
//  type Isa (ulmblas/level3/ukernel/neon/small.h, avx/small.h).
//
//  Real types:
//
//  - A with unit row stride (direct): C in tiles of MV vectors of rows (L
//    elements each) and NC columns, held in registers; for each l one load
//    of the column of A per vector and one broadcast of B(l, j) per column,
//    MV*NC FMA.  The last tile of a column of tiles has fewer vectors (rows
//    beyond m loaded as zero and not stored), the last column of tiles fewer
//    columns.
//  - A row major, B row major and k >= n: the transposed problem
//    C^T <- beta C^T + alpha B^T A^T, direct (C then written element by
//    element, m n; the copy of A would cost m k).
//  - A row major, B column major and k >= 4 L (dot): dot products along l,
//    tiles of DR rows and DC columns, one vector per element of C (for
//    smaller k the reductions cost more than the copy of A).
//  - otherwise A copied into a column-major buffer, direct.
//
//  Complex types: as direct, with two accumulators per element of the tile,
//  R = sum a Re(b) and I = sum a Im(b) (a with real and imaginary part
//  interleaved, b broadcast), combined at the end with the conjugations:
//
//      Re = R.re - I.im,  Im = R.im + I.re     (no conjugation)
//
//  A row major: for B row major and k >= n the transposed problem,
//  otherwise A copied.
//
//  C with any strides: unit row stride by vector loads and stores,
//  otherwise element by element.
//
//  The instruction set provides, for T float and double:
//
//      Vec<Isa, T>    reg, L (lanes), MV, NC (tile), DR, DC (tile of dot,
//                     DC <= 4)
//                     zero(), load(x, r), store(x, v, r) (the first r <= L
//                     elements, the others zero), broadcast(s), fma(a, b,
//                     c) = a b + c, mul, add, reduce(v) (sum of the lanes)
//      CVec<Isa, T>   MV, NC (tile of complex elements, L / 2 per vector),
//                     swap(v) ((re, im) -> (im, re) per element),
//                     pair(s0, s1) ((s0, s1) in every element)
//

#include <algorithm>
#include <complex>
#include <cstddef>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/config/inline.h>
#include <ulmblas/level3/gemm/threads.h>

#if defined(__clang__)
#define ULMBLAS_SIMD_UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define ULMBLAS_SIMD_UNROLL _Pragma("GCC unroll 32")
#else
#define ULMBLAS_SIMD_UNROLL
#endif

namespace ulmBLAS {
namespace simd {

template <typename Isa, typename T> struct Vec;
template <typename Isa, typename T> struct CVec;

namespace small {

//
//  The copy of A (row stride other than 1) has at most CopyMax elements:
//  k is split into blocks (C += A(:,l:l+kb) B(l:l+kb,:) for the blocks
//  after the first), and run leaves m or n above CopyMax to the fallback.
//
constexpr std::ptrdiff_t CopyMax = std::ptrdiff_t(1) << 16;

//
//  The operands of a real problem (direct): A with unit row stride and
//  column stride ldA, B with any strides, C with row stride incRowC and
//  column stride ldC.
//
template <typename T> struct Operands
{
    std::ptrdiff_t k;
    T alpha;
    const T *A;
    std::ptrdiff_t ldA;
    const T *B;
    std::ptrdiff_t incRowB, incColB;
    T beta;
    T *C;
    std::ptrdiff_t incRowC, ldC;
};

//
//  r (1 <= r <= L) elements of a column of C with row stride incRowC:
//  c <- beta c + z (vb: beta in all lanes)
//
template <typename Isa, typename T>
ULMBLAS_ALWAYS_INLINE void
update(T *c, std::ptrdiff_t incRowC, std::ptrdiff_t r,
       typename Vec<Isa, T>::reg z, T beta, typename Vec<Isa, T>::reg vb)
{
    typedef Vec<Isa, T> W;
    if (incRowC == 1) {
	if (beta != T(0)) {
	    z = W::fma(W::load(c, r), vb, z);
	}
	W::store(c, z, r);
	return;
    }
    T t[W::L];
    W::store(t, z, W::L);
    for (std::ptrdiff_t q = 0; q < r; ++q) {
	T &x = c[q * incRowC];
	x = beta != T(0) ? beta * x + t[q] : t[q];
    }
}

//
//  Tile of C at row i and column j with MV vectors of rows (mr rows, mr >
//  (MV-1)*L) and NC columns.
//
template <typename Isa, typename T, std::ptrdiff_t MV, std::ptrdiff_t NC>
ULMBLAS_ALWAYS_INLINE void
tile(std::ptrdiff_t mr, std::ptrdiff_t i, std::ptrdiff_t j,
     const Operands<T> &op)
{
    typedef Vec<Isa, T> W;
    typename W::reg acc[MV][NC];
    const T *A = op.A + i;
    const T *B = op.B + j * op.incColB;
    //  rows of the last vector
    const std::ptrdiff_t rl = mr - (MV - 1) * W::L;

    ULMBLAS_SIMD_UNROLL
    for (std::ptrdiff_t v = 0; v < MV; ++v) {
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t c = 0; c < NC; ++c) {
	    acc[v][c] = W::zero();
	}
    }
    for (std::ptrdiff_t l = 0; l < op.k; ++l) {
	typename W::reg a[MV];
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t v = 0; v < MV; ++v) {
	    a[v] = W::load(A + v * W::L + l * op.ldA, v < MV - 1 ? W::L : rl);
	}
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t c = 0; c < NC; ++c) {
	    const typename W::reg b =
	        W::broadcast(B[l * op.incRowB + c * op.incColB]);
	    ULMBLAS_SIMD_UNROLL
	    for (std::ptrdiff_t v = 0; v < MV; ++v) {
		acc[v][c] = W::fma(a[v], b, acc[v][c]);
	    }
	}
    }
    const typename W::reg va = W::broadcast(op.alpha);
    const typename W::reg vb = W::broadcast(op.beta);
    ULMBLAS_SIMD_UNROLL
    for (std::ptrdiff_t c = 0; c < NC; ++c) {
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t v = 0; v < MV; ++v) {
	    T *Cvc = op.C + (i + v * W::L) * op.incRowC + (j + c) * op.ldC;
	    update<Isa, T>(Cvc, op.incRowC, v < MV - 1 ? W::L : rl,
	                   W::mul(acc[v][c], va), op.beta, vb);
	}
    }
}

//  the last tile of a column of tiles, mr <= MV*L rows: the fewest vectors
template <typename Isa, typename T, std::ptrdiff_t MV, std::ptrdiff_t NC>
struct RowTail
{
    static ULMBLAS_ALWAYS_INLINE void
    run(std::ptrdiff_t mr, std::ptrdiff_t i, std::ptrdiff_t j,
        const Operands<T> &op)
    {
	if (mr > (MV - 1) * Vec<Isa, T>::L) {
	    tile<Isa, T, MV, NC>(mr, i, j, op);
	} else {
	    RowTail<Isa, T, MV - 1, NC>::run(mr, i, j, op);
	}
    }
};

template <typename Isa, typename T, std::ptrdiff_t NC>
struct RowTail<Isa, T, 0, NC>
{
    static ULMBLAS_ALWAYS_INLINE void
    run(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, const Operands<T> &)
    {
    }
};

//  the m rows of the NC columns from column j
template <typename Isa, typename T, std::ptrdiff_t NC>
ULMBLAS_ALWAYS_INLINE void
columns(std::ptrdiff_t m, std::ptrdiff_t j, const Operands<T> &op)
{
    typedef Vec<Isa, T> W;
    constexpr std::ptrdiff_t R = W::MV * W::L;
    std::ptrdiff_t i = 0;
    for (; i + R <= m; i += R) {
	tile<Isa, T, W::MV, NC>(R, i, j, op);
    }
    if (i < m) {
	RowTail<Isa, T, W::MV, NC>::run(m - i, i, j, op);
    }
}

//  the last nc < NC columns from column j
template <typename Isa, typename T, std::ptrdiff_t NC> struct ColTail
{
    static ULMBLAS_ALWAYS_INLINE void
    run(std::ptrdiff_t m, std::ptrdiff_t nc, std::ptrdiff_t j,
        const Operands<T> &op)
    {
	if (nc == NC) {
	    columns<Isa, T, NC>(m, j, op);
	} else {
	    ColTail<Isa, T, NC - 1>::run(m, nc, j, op);
	}
    }
};

template <typename Isa, typename T> struct ColTail<Isa, T, 0>
{
    static ULMBLAS_ALWAYS_INLINE void
    run(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, const Operands<T> &)
    {
    }
};

//  m x n with A of unit row stride
template <typename Isa, typename T>
void
direct(std::ptrdiff_t m, std::ptrdiff_t n, const Operands<T> &op)
{
    typedef Vec<Isa, T> W;
    std::ptrdiff_t j = 0;
    for (; j + W::NC <= n; j += W::NC) {
	columns<Isa, T, W::NC>(m, j, op);
    }
    if (j < n) {
	ColTail<Isa, T, W::NC - 1>::run(m, n - j, j, op);
    }
}

//
//  Dot products, for A row major and B column major (the rows of A and the
//  columns of B contiguous in l): tile of DR rows and NC columns of C, one
//  vector per element of C along l.  Rows beyond mr: computed from row 0,
//  not stored.
//
template <typename Isa, typename T, std::ptrdiff_t NC>
ULMBLAS_ALWAYS_INLINE void
dotTile(std::ptrdiff_t mr, std::ptrdiff_t k, T alpha, const T *A,
        std::ptrdiff_t incRowA, const T *B, std::ptrdiff_t incColB, T beta,
        T *C, std::ptrdiff_t incRowC, std::ptrdiff_t ldC)
{
    typedef Vec<Isa, T> W;
    constexpr std::ptrdiff_t MR = W::DR;
    typename W::reg acc[MR][NC];
    const T *a[MR];

    ULMBLAS_SIMD_UNROLL
    for (std::ptrdiff_t i = 0; i < MR; ++i) {
	a[i] = A + (i < mr ? i : 0) * incRowA;
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t j = 0; j < NC; ++j) {
	    acc[i][j] = W::zero();
	}
    }
    for (std::ptrdiff_t l = 0; l < k; l += W::L) {
	const std::ptrdiff_t r = k - l < W::L ? k - l : W::L;
	typename W::reg x[MR], y[NC];
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t i = 0; i < MR; ++i) {
	    x[i] = W::load(a[i] + l, r);
	}
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t j = 0; j < NC; ++j) {
	    y[j] = W::load(B + j * incColB + l, r);
	}
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t j = 0; j < NC; ++j) {
	    ULMBLAS_SIMD_UNROLL
	    for (std::ptrdiff_t i = 0; i < MR; ++i) {
		acc[i][j] = W::fma(x[i], y[j], acc[i][j]);
	    }
	}
    }
    ULMBLAS_SIMD_UNROLL
    for (std::ptrdiff_t j = 0; j < NC; ++j) {
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t i = 0; i < MR; ++i) {
	    if (i < mr) {
		T &c = C[i * incRowC + j * ldC];
		const T s = alpha * W::reduce(acc[i][j]);
		c = beta != T(0) ? beta * c + s : s;
	    }
	}
    }
}

//  m x n with A row major, B column major
template <typename Isa, typename T>
void
dot(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k, T alpha, const T *A,
    std::ptrdiff_t incRowA, const T *B, std::ptrdiff_t incColB, T beta, T *C,
    std::ptrdiff_t incRowC, std::ptrdiff_t ldC)
{
    typedef Vec<Isa, T> W;
    static_assert(W::DC <= 4, "column remainders below assume DC <= 4");
    constexpr std::ptrdiff_t NC = W::DC;
    for (std::ptrdiff_t i = 0; i < m; i += W::DR) {
	const std::ptrdiff_t mr = m - i < W::DR ? m - i : W::DR;
	const T *Ai = A + i * incRowA;
	T *Ci = C + i * incRowC;
	std::ptrdiff_t j = 0;
	for (; j + NC <= n; j += NC) {
	    dotTile<Isa, T, NC>(mr, k, alpha, Ai, incRowA, B + j * incColB,
	                        incColB, beta, Ci + j * ldC, incRowC, ldC);
	}
	const T *Bj = B + j * incColB;
	T *Cj = Ci + j * ldC;
	switch (n - j) {
	case 3:
	    dotTile<Isa, T, 3>(mr, k, alpha, Ai, incRowA, Bj, incColB, beta, Cj,
	                       incRowC, ldC);
	    break;
	case 2:
	    dotTile<Isa, T, 2>(mr, k, alpha, Ai, incRowA, Bj, incColB, beta, Cj,
	                       incRowC, ldC);
	    break;
	case 1:
	    dotTile<Isa, T, 1>(mr, k, alpha, Ai, incRowA, Bj, incColB, beta, Cj,
	                       incRowC, ldC);
	    break;
	}
    }
}

//
//  C <- beta C + alpha A B, real T, any strides (m, n, k > 0)
//
template <typename Isa, typename IndexType, typename T>
void
gemm(IndexType m, IndexType n, IndexType k, T alpha, const T *A,
     IndexType incRowA, IndexType incColA, const T *B, IndexType incRowB,
     IndexType incColB, T beta, T *C, IndexType incRowC, IndexType incColC)
{
    static MemoryPool<T> memoryPool;

    if (incRowA != IndexType(1) && incColB == IndexType(1) && k >= n) {
	gemm<Isa>(n, m, k, alpha, B, incColB, incRowB, A, incColA, incRowA,
	          beta, C, incColC, incRowC);
	return;
    }
    if (incRowA != IndexType(1) && incColA == IndexType(1) &&
        incRowB == IndexType(1) && k >= 4 * Vec<Isa, T>::L) {
	dot<Isa, T>(m, n, k, alpha, A, incRowA, B, incColB, beta, C, incRowC,
	            incColC);
	return;
    }
    T *Ab = nullptr;
    std::ptrdiff_t ldA = incColA;
    if (incRowA != IndexType(1)) {
	//  (m may be 0 here: the callers of the plans do not exclude it)
	const IndexType kc = std::max(
	    IndexType(1), IndexType(CopyMax / std::max(m, IndexType(1))));
	if (k > kc) {
	    for (IndexType l = 0; l < k; l += kc) {
		const IndexType kb = std::min(kc, k - l);
		gemm<Isa>(m, n, kb, alpha, &A[l * incColA], incRowA, incColA,
		          &B[l * incRowB], incRowB, incColB,
		          l == 0 ? beta : T(1), C, incRowC, incColC);
	    }
	    return;
	}
	Ab = memoryPool.allocate(std::size_t(m) * std::size_t(k));
	for (IndexType l = 0; l < k; ++l) {
	    for (IndexType i = 0; i < m; ++i) {
		Ab[i + l * m] = A[i * incRowA + l * incColA];
	    }
	}
	A = Ab;
	ldA = m;
    }
    const Operands<T> op = {k,       alpha, A, ldA,     B,      incRowB,
                            incColB, beta,  C, incRowC, incColC};
    direct<Isa, T>(m, n, op);
    if (Ab) {
	memoryPool.release(Ab);
    }
}

//
//  Complex: the operands (T the real type, strides in complex elements)
//  and the signs sR, sI of R and swap(I) for the conjugations.
//
template <typename Isa, typename T> struct COperands
{
    typedef typename Vec<Isa, T>::reg reg;

    std::ptrdiff_t k;
    std::complex<T> alpha;
    const T *A;
    std::ptrdiff_t ldA;
    const T *B;
    std::ptrdiff_t incRowB, incColB;
    std::complex<T> beta;
    T *C;
    std::ptrdiff_t incRowC, ldC;
    reg sR, sI;
};

//  x y for complex x (interleaved) and complex scalar y, given as
//  yr = (Re y, Re y) and yi = (-Im y, Im y) per element: x yr + swap(x) yi
template <typename Isa, typename T>
ULMBLAS_ALWAYS_INLINE typename Vec<Isa, T>::reg
cmul(typename Vec<Isa, T>::reg x, typename Vec<Isa, T>::reg yr,
     typename Vec<Isa, T>::reg yi)
{
    typedef Vec<Isa, T> W;
    return W::fma(CVec<Isa, T>::swap(x), yi, W::mul(x, yr));
}

//
//  Complex tile at row i and column j with MV vectors of rows (mr complex
//  rows, mr > (MV-1)*CL, CL = L / 2) and NC columns.
//
template <typename Isa, typename T, std::ptrdiff_t MV, std::ptrdiff_t NC>
ULMBLAS_ALWAYS_INLINE void
ctile(std::ptrdiff_t mr, std::ptrdiff_t i, std::ptrdiff_t j,
      const COperands<Isa, T> &op)
{
    typedef Vec<Isa, T> W;
    typedef CVec<Isa, T> X;
    constexpr std::ptrdiff_t CL = W::L / 2;
    typename W::reg accR[MV][NC], accI[MV][NC];
    const T *A = op.A + 2 * i;
    const T *B = op.B + 2 * j * op.incColB;
    const std::ptrdiff_t rl = mr - (MV - 1) * CL;

    ULMBLAS_SIMD_UNROLL
    for (std::ptrdiff_t v = 0; v < MV; ++v) {
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t c = 0; c < NC; ++c) {
	    accR[v][c] = W::zero();
	    accI[v][c] = W::zero();
	}
    }
    for (std::ptrdiff_t l = 0; l < op.k; ++l) {
	typename W::reg a[MV];
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t v = 0; v < MV; ++v) {
	    a[v] = W::load(A + 2 * (v * CL + l * op.ldA),
	                   2 * (v < MV - 1 ? CL : rl));
	}
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t c = 0; c < NC; ++c) {
	    const T *b = B + 2 * (l * op.incRowB + c * op.incColB);
	    const typename W::reg br = W::broadcast(b[0]);
	    const typename W::reg bi = W::broadcast(b[1]);
	    ULMBLAS_SIMD_UNROLL
	    for (std::ptrdiff_t v = 0; v < MV; ++v) {
		accR[v][c] = W::fma(a[v], br, accR[v][c]);
		accI[v][c] = W::fma(a[v], bi, accI[v][c]);
	    }
	}
    }
    const typename W::reg ar = W::broadcast(op.alpha.real());
    const typename W::reg ai = X::pair(-op.alpha.imag(), op.alpha.imag());
    const typename W::reg br = W::broadcast(op.beta.real());
    const typename W::reg bi = X::pair(-op.beta.imag(), op.beta.imag());
    const bool beta0 = op.beta == std::complex<T>(0);
    ULMBLAS_SIMD_UNROLL
    for (std::ptrdiff_t c = 0; c < NC; ++c) {
	ULMBLAS_SIMD_UNROLL
	for (std::ptrdiff_t v = 0; v < MV; ++v) {
	    const std::ptrdiff_t r = v < MV - 1 ? CL : rl;
	    typename W::reg z =
	        W::fma(X::swap(accI[v][c]), op.sI, W::mul(accR[v][c], op.sR));
	    z = cmul<Isa, T>(z, ar, ai);
	    T *Cvc = op.C + 2 * ((i + v * CL) * op.incRowC + (j + c) * op.ldC);
	    if (op.incRowC == 1) {
		if (!beta0) {
		    z = W::add(z, cmul<Isa, T>(W::load(Cvc, 2 * r), br, bi));
		}
		W::store(Cvc, z, 2 * r);
	    } else {
		T t[W::L];
		W::store(t, z, W::L);
		for (std::ptrdiff_t q = 0; q < r; ++q) {
		    std::complex<T> *cq = reinterpret_cast<std::complex<T> *>(
		        Cvc + 2 * q * op.incRowC);
		    const std::complex<T> s(t[2 * q], t[2 * q + 1]);
		    *cq = beta0 ? s : op.beta * *cq + s;
		}
	    }
	}
    }
}

template <typename Isa, typename T, std::ptrdiff_t MV, std::ptrdiff_t NC>
struct CRowTail
{
    static ULMBLAS_ALWAYS_INLINE void
    run(std::ptrdiff_t mr, std::ptrdiff_t i, std::ptrdiff_t j,
        const COperands<Isa, T> &op)
    {
	if (mr > (MV - 1) * (Vec<Isa, T>::L / 2)) {
	    ctile<Isa, T, MV, NC>(mr, i, j, op);
	} else {
	    CRowTail<Isa, T, MV - 1, NC>::run(mr, i, j, op);
	}
    }
};

template <typename Isa, typename T, std::ptrdiff_t NC>
struct CRowTail<Isa, T, 0, NC>
{
    static ULMBLAS_ALWAYS_INLINE void
    run(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t,
        const COperands<Isa, T> &)
    {
    }
};

template <typename Isa, typename T, std::ptrdiff_t NC>
ULMBLAS_ALWAYS_INLINE void
ccolumns(std::ptrdiff_t m, std::ptrdiff_t j, const COperands<Isa, T> &op)
{
    typedef CVec<Isa, T> X;
    constexpr std::ptrdiff_t R = X::MV * (Vec<Isa, T>::L / 2);
    std::ptrdiff_t i = 0;
    for (; i + R <= m; i += R) {
	ctile<Isa, T, X::MV, NC>(R, i, j, op);
    }
    if (i < m) {
	CRowTail<Isa, T, X::MV, NC>::run(m - i, i, j, op);
    }
}

template <typename Isa, typename T, std::ptrdiff_t NC> struct CColTail
{
    static ULMBLAS_ALWAYS_INLINE void
    run(std::ptrdiff_t m, std::ptrdiff_t nc, std::ptrdiff_t j,
        const COperands<Isa, T> &op)
    {
	if (nc == NC) {
	    ccolumns<Isa, T, NC>(m, j, op);
	} else {
	    CColTail<Isa, T, NC - 1>::run(m, nc, j, op);
	}
    }
};

template <typename Isa, typename T> struct CColTail<Isa, T, 0>
{
    static ULMBLAS_ALWAYS_INLINE void
    run(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t,
        const COperands<Isa, T> &)
    {
    }
};

//
//  C <- beta C + alpha op(A) op(B), complex, any strides (m, n, k > 0;
//  strides in complex elements), op = conjugation if conjA, conjB
//
template <typename Isa, typename IndexType, typename T>
void
cgemm(IndexType m, IndexType n, IndexType k, const std::complex<T> &alpha,
      bool conjA, const std::complex<T> *A, IndexType incRowA,
      IndexType incColA, bool conjB, const std::complex<T> *B,
      IndexType incRowB, IndexType incColB, const std::complex<T> &beta,
      std::complex<T> *C, IndexType incRowC, IndexType incColC)
{
    typedef CVec<Isa, T> X;
    static MemoryPool<std::complex<T>> memoryPool;

    if (incRowA != IndexType(1) && incColB == IndexType(1) && k >= n) {
	cgemm<Isa>(n, m, k, alpha, conjB, B, incColB, incRowB, conjA, A,
	           incColA, incRowA, beta, C, incColC, incRowC);
	return;
    }
    std::complex<T> *Ab = nullptr;
    std::ptrdiff_t ldA = incColA;
    if (incRowA != IndexType(1)) {
	//  (m may be 0 here: the callers of the plans do not exclude it)
	const IndexType kc = std::max(
	    IndexType(1), IndexType(CopyMax / std::max(m, IndexType(1))));
	if (k > kc) {
	    for (IndexType l = 0; l < k; l += kc) {
		const IndexType kb = std::min(kc, k - l);
		cgemm<Isa>(m, n, kb, alpha, conjA, &A[l * incColA], incRowA,
		           incColA, conjB, &B[l * incRowB], incRowB, incColB,
		           l == 0 ? beta : std::complex<T>(1), C, incRowC,
		           incColC);
	    }
	    return;
	}
	Ab = memoryPool.allocate(std::size_t(m) * std::size_t(k));
	for (IndexType l = 0; l < k; ++l) {
	    for (IndexType i = 0; i < m; ++i) {
		Ab[i + l * m] = A[i * incRowA + l * incColA];
	    }
	}
	A = Ab;
	ldA = m;
    }
    //  signs of R and swap(I): Re = R.re -+ I.im, Im = +-R.im +- I.re
    const T one(1);
    COperands<Isa, T> op;
    op.k = k;
    op.alpha = alpha;
    op.A = reinterpret_cast<const T *>(A);
    op.ldA = ldA;
    op.B = reinterpret_cast<const T *>(B);
    op.incRowB = incRowB;
    op.incColB = incColB;
    op.beta = beta;
    op.C = reinterpret_cast<T *>(C);
    op.incRowC = incRowC;
    op.ldC = incColC;
    op.sR = X::pair(one, conjA ? -one : one);
    op.sI = X::pair(conjA == conjB ? -one : one, conjB ? -one : one);

    std::ptrdiff_t j = 0;
    for (; j + X::NC <= n; j += X::NC) {
	ccolumns<Isa, T, X::NC>(m, j, op);
    }
    if (j < n) {
	CColTail<Isa, T, X::NC - 1>::run(m, n - j, j, op);
    }
    if (Ab) {
	memoryPool.release(Ab);
    }
}

//
//  The kernel as a plan of gemm (true), false for other types (then the
//  caller uses its fallback).  For real types conjugation has no effect.
//
template <typename Isa, typename IndexType, typename Alpha, typename TA,
          typename TB, typename Beta, typename TC>
inline bool
run(IndexType m, IndexType n, IndexType k, const Alpha &alpha, bool conjA,
    const TA *A, IndexType incRowA, IndexType incColA, bool conjB, const TB *B,
    IndexType incRowB, IndexType incColB, const Beta &beta, TC *C,
    IndexType incRowC, IndexType incColC)
{
    return false;
}

#define ULMBLAS_SIMD_SMALL_REAL(T)                                             \
    template <typename Isa, typename IndexType>                                \
    inline bool run(IndexType m, IndexType n, IndexType k, const T &alpha,     \
                    bool conjA, const T *A, IndexType incRowA,                 \
                    IndexType incColA, bool conjB, const T *B,                 \
                    IndexType incRowB, IndexType incColB, const T &beta, T *C, \
                    IndexType incRowC, IndexType incColC)                      \
    {                                                                          \
	if (m > IndexType(CopyMax) || n > IndexType(CopyMax)) {                \
	    return false;                                                      \
	}                                                                      \
	gemm<Isa>(m, n, k, alpha, A, incRowA, incColA, B, incRowB, incColB,    \
	          beta, C, incRowC, incColC);                                  \
	return true;                                                           \
    }

#define ULMBLAS_SIMD_SMALL_COMPLEX(T)                                          \
    template <typename Isa, typename IndexType>                                \
    inline bool run(                                                           \
        IndexType m, IndexType n, IndexType k, const std::complex<T> &alpha,   \
        bool conjA, const std::complex<T> *A, IndexType incRowA,               \
        IndexType incColA, bool conjB, const std::complex<T> *B,               \
        IndexType incRowB, IndexType incColB, const std::complex<T> &beta,     \
        std::complex<T> *C, IndexType incRowC, IndexType incColC)              \
    {                                                                          \
	if (m > IndexType(CopyMax) || n > IndexType(CopyMax)) {                \
	    return false;                                                      \
	}                                                                      \
	cgemm<Isa>(m, n, k, alpha, conjA, A, incRowA, incColA, conjB, B,       \
	           incRowB, incColB, beta, C, incRowC, incColC);               \
	return true;                                                           \
    }

ULMBLAS_SIMD_SMALL_REAL(float)
ULMBLAS_SIMD_SMALL_REAL(double)
ULMBLAS_SIMD_SMALL_COMPLEX(float)
ULMBLAS_SIMD_SMALL_COMPLEX(double)

#undef ULMBLAS_SIMD_SMALL_REAL
#undef ULMBLAS_SIMD_SMALL_COMPLEX

} // namespace small

//
//  The kernel as a plan, for m <= MMax and n <= NMax (0: no bound),
//  otherwise (and for other types) Fallback; also for problems with more
//  than one thread (GemmThreads, ulmblas/level3/gemm/threads.h): the
//  kernel is single threaded, the fallback (the frame algorithm) is not.
//
template <typename Isa, typename Fallback, std::ptrdiff_t MMax = 0,
          std::ptrdiff_t NMax = 0>
struct SmallGemm
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
	typedef decltype(Alpha(0) * TA(0) * TB(0)) T;
	if ((MMax > 0 && m > IndexType(MMax)) ||
	    (NMax > 0 && n > IndexType(NMax)) ||
	    GemmThreads::forProblem<T>(m, n, k) > 1 ||
	    !small::run<Isa>(m, n, k, alpha, conjA, A, incRowA, incColA, conjB,
	                     B, incRowB, incColB, beta, C, incRowC, incColC)) {
	    Fallback::template run<IndexType, Alpha, TA, TB, Beta, TC>(
	        m, n, k, alpha, conjA, A, incRowA, incColA, conjB, B, incRowB,
	        incColB, beta, C, incRowC, incColC);
	}
    }
};

} // namespace simd
} // namespace ulmBLAS

#undef ULMBLAS_SIMD_UNROLL

#endif // ULMBLAS_LEVEL3_UKERNEL_SIMD_SMALL_H
