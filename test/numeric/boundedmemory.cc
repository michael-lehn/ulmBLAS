//
//  Design rule: the size of every buffer of the library is bounded by a
//  constant (block sizes of the configuration), never proportional to the
//  dimensions of a problem (ulmblas/auxiliary/memorypool.h).
//
//  T0385: the routines run with general strides (neither stride 1: the
//  cases in which matrices used to be copied) on problems whose matrices
//  are larger than the bound, on one thread of the thread pool (the
//  buffers of the multithreaded frame grow with the number of threads).
//  The largest block requested from the memory pools (memoryPoolLargest)
//  must stay below the bound, or below the largest block of the same case
//  with column-major matrices (no copies; the selection tables may choose
//  plans with larger block sizes for these sizes).  The bound: the largest
//  block of the level 2 and 3 routines and the integer GEMM (general
//  matrix multiply) on moderate problems (the packing buffers of the
//  configuration), the largest copy of the LAPACK (Linear Algebra PACKage)
//  functions (lapackCopyMax) and of the kernels for small matrices (2^16
//  elements).
//
//  T0386: cblas_gemm_s8u8s32 and cblas_gemm_s16s16s32 with more than one
//  block of rows and columns (1024) against plain loops; they compute in
//  blocks of rows and columns since the buffers are bounded.
//
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include BLAS_HEADER
#include <test/numeric/testing.h>
#include <ulmblas/auxiliary/memorypool.h>
#include <ulmblas/auxiliary/threadpool.h>
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/getri.h>
#include <ulmblas/lapack/potrf.h>
#include <ulmblas/lapack/unitstride.h>
#include <ulmblas/ulmblas.h>

using namespace test;

namespace {

typedef std::ptrdiff_t Index;

//  one thread of the thread pool while an object lives
struct OneThread
{
    OneThread() : size(ulmBLAS::ThreadPool::instance().size())
    {
	ulmBLAS::ThreadPool::instance().setSize(1);
    }
    ~OneThread()
    {
	ulmBLAS::ThreadPool::instance().setSize(size);
    }
    std::ptrdiff_t size;
};

//  m x n with row stride 2 and column stride 2 m + 1 (neither stride 1),
//  or column major (general = false), diagonally dominant (n on the
//  diagonal) and Hermitian if square
template <typename T> struct General
{
    General(Index m_, Index n_, bool general = true)
        : m(m_), n(n_), rs(general ? 2 : 1), cs(general ? 2 * m_ + 1 : m_),
          data(std::size_t(cs * n_ + 1), T(0))
    {
	for (Index j = 0; j < n; ++j) {
	    for (Index i = 0; i < m; ++i) {
		const double x = double((i * 7 + j * 3) % 11) / 11 - 0.5;
		(*this)(i, j) = i == j ? T(double(n)) : T(x);
	    }
	}
    }

    T &
    operator()(Index i, Index j)
    {
	return data[std::size_t(i * rs + j * cs)];
    }

    T *
    ptr()
    {
	return data.data();
    }

    Index m, n, rs, cs;
    std::vector<T> data;
};

template <typename T>
void
level3(Index n)
{
    General<T> A(n, n), B(n, n), C(n, n);
    ulmBLAS::gemm(n, n, n, T(1), false, A.ptr(), A.rs, A.cs, false, B.ptr(),
                  B.rs, B.cs, T(1), C.ptr(), C.rs, C.cs);
    ulmBLAS::trlsm(n, n, T(1), false, false, A.ptr(), A.rs, A.cs, B.ptr(), B.rs,
                   B.cs);
    ulmBLAS::trumm(n, n, T(1), false, false, A.ptr(), A.rs, A.cs, B.ptr(), B.rs,
                   B.cs);
    //  conjugated: other entries of the selection tables
    ulmBLAS::trlsm(n, n, T(1), true, false, A.ptr(), A.rs, A.cs, B.ptr(), B.rs,
                   B.cs);
    ulmBLAS::trusm(n, n, T(1), true, true, A.ptr(), A.rs, A.cs, B.ptr(), B.rs,
                   B.cs);
    ulmBLAS::trlmm(n, n, T(1), true, false, A.ptr(), A.rs, A.cs, B.ptr(), B.rs,
                   B.cs);
    ulmBLAS::sylrk(n, n, T(1), A.ptr(), A.rs, A.cs, T(1), C.ptr(), C.rs, C.cs);
    //  level 2 with packed A (neither stride 1)
    std::vector<T> x(std::size_t(2 * n), T(1)), y(std::size_t(2 * n), T(1));
    ulmBLAS::gemv(n, n, T(1), A.ptr(), A.rs, A.cs, x.data(), Index(2), T(1),
                  y.data(), Index(2));
    ulmBLAS::ger(n, n, T(1), x.data(), Index(2), y.data(), Index(2), C.ptr(),
                 C.rs, C.cs);
}

//  the bound in bytes (see above)
std::size_t
bound()
{
    ulmBLAS::memoryPoolResetLargest();
    level3<float>(300);
    level3<double>(300);
    level3<std::complex<float>>(300);
    level3<std::complex<double>>(300);
    //  integer GEMM (packing buffers of the 8- and 16-bit kernels)
    {
	const Index n = 64;
	std::vector<std::int8_t> A8(std::size_t(n * n), 1);
	std::vector<std::uint8_t> B8(std::size_t(n * n), 1);
	std::vector<std::int16_t> A16(std::size_t(n * n), 1),
	    B16(std::size_t(n * n), 1);
	std::vector<std::int32_t> C(std::size_t(n * n), 0);
	const std::int32_t co = 0;
	cblas_gemm_s8u8s32(CblasColMajor, CblasNoTrans, CblasNoTrans,
	                   CblasFixOffset, n, n, n, 1.0f, A8.data(), n, 1,
	                   B8.data(), n, 1, 0.0f, C.data(), n, &co);
	cblas_gemm_s16s16s32(CblasColMajor, CblasNoTrans, CblasNoTrans,
	                     CblasFixOffset, n, n, n, 1.0f, A16.data(), n, 1,
	                     B16.data(), n, 1, 0.0f, C.data(), n, &co);
    }
    std::size_t b = ulmBLAS::memoryPoolLargest();
    b = std::max(b, ulmBLAS::lapackCopyMax<float>() * sizeof(float));
    b = std::max(b, ulmBLAS::lapackCopyMax<double>() * sizeof(double));
    b = std::max(b, ulmBLAS::lapackCopyMax<std::complex<float>>() *
                        sizeof(std::complex<float>));
    b = std::max(b, ulmBLAS::lapackCopyMax<std::complex<double>>() *
                        sizeof(std::complex<double>));
    b = std::max(b, (std::size_t(1) << 16) * sizeof(std::complex<double>));
    return b;
}

//  elements of a matrix of type T above the bound
template <typename T>
Index
above(std::size_t b)
{
    return Index(b / sizeof(T) / 4 * 5 + 1);
}

//
//  f(general): the case with general strides (true) and column major
//  (false).  With general strides the largest block must not exceed the
//  bound or the largest block with column-major matrices (the packing
//  buffers of the plans chosen for the sizes of the case).
//
template <typename F>
bool
bounded(Context &ctx, const char *what, std::size_t b, F f)
{
    ulmBLAS::memoryPoolResetLargest();
    f(false);
    const std::size_t lu = ulmBLAS::memoryPoolLargest();
    ulmBLAS::memoryPoolResetLargest();
    f(true);
    const std::size_t l = ulmBLAS::memoryPoolLargest();
    return ctx.check(l <= std::max(b, lu),
                     "%s: largest block %zu bytes, bound %zu, column major "
                     "%zu",
                     what, l, b, lu);
}

template <typename T>
bool
factorizations(Context &ctx, std::size_t b, const char *type)
{
    char what[128];
    bool ok = true;
    //  square: n x n above the bound
    const Index n = Index(std::sqrt(double(above<T>(b)))) + 1;
    //  tall: m x 64 above the bound
    const Index m = above<T>(b) / 64 + 1;

    std::snprintf(what, sizeof(what), "%s potrf n=%lld", type, (long long)n);
    ok = ok && bounded(ctx, what, b, [n, &ctx](bool g) {
	     General<T> A(n, n, g);
	     const Index info = ulmBLAS::potrf(true, n, A.ptr(), A.rs, A.cs);
	     ctx.check(info == 0, "potrf: info %lld", (long long)info);
         });
    std::snprintf(what, sizeof(what), "%s getrf m=%lld n=64", type,
                  (long long)m);
    ok = ok && bounded(ctx, what, b, [m, &ctx](bool g) {
	     General<T> A(m, 64, g);
	     std::vector<int> piv(64);
	     const Index info = ulmBLAS::getrf(m, Index(64), A.ptr(), A.rs,
	                                       A.cs, piv.data(), Index(1));
	     ctx.check(info == 0, "getrf: info %lld", (long long)info);
         });
    std::snprintf(what, sizeof(what), "%s getrf, getri n=%lld", type,
                  (long long)n);
    ok = ok && bounded(ctx, what, b, [n, &ctx](bool g) {
	     General<T> A(n, n, g);
	     std::vector<Index> piv(static_cast<std::size_t>(n));
	     Index info = ulmBLAS::getrf(n, n, A.ptr(), A.rs, A.cs, piv.data(),
	                                 Index(1));
	     info +=
	         ulmBLAS::getri(n, A.ptr(), A.rs, A.cs, piv.data(), Index(1));
	     ctx.check(info == 0, "getrf, getri: info %lld", (long long)info);
         });
    return ok;
}

} // namespace

ULMBLAS_TEST(T0385,
             "auxiliary/MemoryPool: buffers bounded by a constant, LAPACK "
             "functions and gemm with general strides on large problems")
{
    OneThread one;
    const std::size_t b = bound();
    bool ok = factorizations<double>(ctx, b, "double") &&
              factorizations<std::complex<double>>(ctx, b, "complex<double>");
    //  tall LU for the other types
    ok = ok && bounded(ctx, "float getrf tall", b, [b](bool g) {
	     const Index m = above<float>(b) / 64 + 1;
	     General<float> A(m, 64, g);
	     std::vector<Index> piv(64);
	     ulmBLAS::getrf(m, Index(64), A.ptr(), A.rs, A.cs, piv.data(),
	                    Index(1));
         });
    ok = ok && bounded(ctx, "complex<float> getrf tall", b, [b](bool g) {
	     const Index m = above<std::complex<float>>(b) / 64 + 1;
	     General<std::complex<float>> A(m, 64, g);
	     std::vector<Index> piv(64);
	     ulmBLAS::getrf(m, Index(64), A.ptr(), A.rs, A.cs, piv.data(),
	                    Index(1));
         });
    //  gemm with small m and n, long k, A with row stride 2 (the kernels
    //  for small matrices copy A)
    ok = ok && bounded(ctx, "dgemm 8 x 8 x k", b, [b](bool g) {
	     const Index k = above<double>(b) / 8 + 1;
	     General<double> A(8, k, g), B(k, 8), C(8, 8);
	     ulmBLAS::gemm(Index(8), Index(8), k, 1.0, false, A.ptr(), A.rs,
	                   A.cs, false, B.ptr(), Index(1), k, 0.0, C.ptr(),
	                   Index(1), Index(8));
         });
    ok = ok && bounded(ctx, "zgemm 8 x 8 x k", b, [b](bool g) {
	     typedef std::complex<double> Z;
	     const Index k = above<Z>(b) / 8 + 1;
	     General<Z> A(8, k, g), B(k, 8), C(8, 8);
	     ulmBLAS::gemm(Index(8), Index(8), k, Z(1), false, A.ptr(), A.rs,
	                   A.cs, false, B.ptr(), Index(1), k, Z(0), C.ptr(),
	                   Index(1), Index(8));
         });
    //  integer GEMM with offsets: C (m x 1024) above the bound
    ok = ok && bounded(ctx, "cblas_gemm_s8u8s32 m x 1024", b, [b](bool g) {
	     const Index m = above<std::int32_t>(b) / 1024 + 1;
	     std::vector<std::int8_t> A(std::size_t(m * 16), 1);
	     std::vector<std::uint8_t> B(16 * 1024, 1);
	     std::vector<std::int32_t> C(std::size_t(m * 1024), 0);
	     const std::int32_t co = 0;
	     cblas_gemm_s8u8s32(CblasColMajor, CblasNoTrans, CblasNoTrans,
	                        CblasFixOffset, m, 1024, 16, 1.0f, A.data(), m,
	                        1, B.data(), 16, 1, 0.0f, C.data(), m, &co);
         });
    ok =
        ok && bounded(ctx, "cblas_gemm_s16s16s32 m x 1024", b, [b](bool g) {
	    const Index m = above<std::int32_t>(b) / 1024 + 1;
	    std::vector<std::int16_t> A(std::size_t(m * 16), 1);
	    std::vector<std::int16_t> B(16 * 1024, 1);
	    std::vector<std::int32_t> C(std::size_t(m * 1024), 0);
	    const std::int32_t co = 0;
	    cblas_gemm_s16s16s32(CblasColMajor, CblasNoTrans, CblasNoTrans,
	                         CblasFixOffset, m, 1024, 16, 1.0f, A.data(), m,
	                         1, B.data(), 16, 1, 0.0f, C.data(), m, &co);
        });
    if (ok) {
	ctx.pass();
    }
}

namespace {

//  C <- alpha (A + ao)(B + bo) + beta C + co (CblasColOffset), column
//  major, plain loops; |values| small, so no wraparound and no rounding
template <typename TA, typename TB>
void
reference(Index m, Index n, Index k, double alpha, const TA *A, int ao,
          const TB *B, int bo, double beta, std::int32_t *C,
          const std::int32_t *co)
{
    for (Index j = 0; j < n; ++j) {
	for (Index i = 0; i < m; ++i) {
	    std::int64_t s = 0;
	    for (Index l = 0; l < k; ++l) {
		s += (std::int64_t(A[i + l * m]) + ao) *
		     (std::int64_t(B[l + j * k]) + bo);
	    }
	    const double x =
	        alpha * double(s) + beta * double(C[i + j * m]) + co[i];
	    C[i + j * m] = std::int32_t(std::lround(x));
	}
    }
}

} // namespace

ULMBLAS_TEST(T0386,
             "interface/cblas_gemm_s8u8s32, cblas_gemm_s16s16s32: several "
             "blocks of 1024 rows and columns, offsets per row")
{
    const Index m = 1100, n = 1030, k = 7;
    std::vector<std::int32_t> co(static_cast<std::size_t>(m));
    for (Index i = 0; i < m; ++i) {
	co[std::size_t(i)] = std::int32_t(i % 13) - 6;
    }
    std::vector<std::int32_t> C0(std::size_t(m * n));
    for (Index q = 0; q < m * n; ++q) {
	C0[std::size_t(q)] = std::int32_t(q % 17) - 8;
    }
    bool ok = true;
    {
	std::vector<std::int8_t> A(std::size_t(m * k));
	std::vector<std::uint8_t> B(std::size_t(k * n));
	for (Index q = 0; q < m * k; ++q) {
	    A[std::size_t(q)] = std::int8_t(q % 19 - 9);
	}
	for (Index q = 0; q < k * n; ++q) {
	    B[std::size_t(q)] = std::uint8_t(q % 23);
	}
	std::vector<std::int32_t> C(C0), R(C0);
	cblas_gemm_s8u8s32(CblasColMajor, CblasNoTrans, CblasNoTrans,
	                   CblasColOffset, m, n, k, 2.0f, A.data(), m, 3,
	                   B.data(), k, -5, 0.5f, C.data(), m, co.data());
	reference(m, n, k, 2.0, A.data(), 3, B.data(), -5, 0.5, R.data(),
	          co.data());
	ok = ctx.check(C == R, "cblas_gemm_s8u8s32: results differ");
    }
    {
	std::vector<std::int16_t> A(std::size_t(m * k));
	std::vector<std::int16_t> B(std::size_t(k * n));
	for (Index q = 0; q < m * k; ++q) {
	    A[std::size_t(q)] = std::int16_t(q % 101 - 50);
	}
	for (Index q = 0; q < k * n; ++q) {
	    B[std::size_t(q)] = std::int16_t(q % 97 - 48);
	}
	std::vector<std::int32_t> C(C0), R(C0);
	cblas_gemm_s16s16s32(CblasColMajor, CblasNoTrans, CblasNoTrans,
	                     CblasColOffset, m, n, k, 2.0f, A.data(), m, 3,
	                     B.data(), k, -5, 0.5f, C.data(), m, co.data());
	reference(m, n, k, 2.0, A.data(), 3, B.data(), -5, 0.5, R.data(),
	          co.data());
	ok = ctx.check(C == R, "cblas_gemm_s16s16s32: results differ") && ok;
    }
    if (ok) {
	ctx.pass();
    }
}
