#ifndef ULMBLAS_TEST_NUMERIC_LAPACKTEST_H
#define ULMBLAS_TEST_NUMERIC_LAPACKTEST_H 1

//
//  Helpers of the tests of the LAPACK templates (factor.cc, solve.cc):
//  strided matrices in a buffer of their own (the other elements hold a
//  marker), storage layouts, machine epsilon of the real type.
//

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#include <test/numeric/reference.h>
#include <test/numeric/testing.h>

namespace test {
namespace lapacktest {

typedef std::int64_t Index;

//
//  A strided matrix in a buffer of its own: element (i,j) is
//  data[off + i*rs + j*cs], the rest of the buffer holds a marker.
//
template <typename T> struct Strided
{
    Strided(Index m_, Index n_, Index rs_, Index cs_)
        : m(m_), n(n_), rs(rs_), cs(cs_)
    {
	const Index lo =
	    std::min<Index>(0, (m - 1) * rs) + std::min<Index>(0, (n - 1) * cs);
	const Index hi =
	    std::max<Index>(0, (m - 1) * rs) + std::max<Index>(0, (n - 1) * cs);
	buf.assign(std::size_t(hi - lo + 1),
	           fromCdouble<T>(cdouble(1234.5, -77.25)));
	off = -lo;
    }

    T &
    operator()(Index i, Index j)
    {
	return buf[std::size_t(off + i * rs + j * cs)];
    }

    T *
    ptr()
    {
	return &buf[std::size_t(off)];
    }

    Index m, n, rs, cs, off;
    std::vector<T> buf;
};

//  layouts: (row stride, column stride) for an m x n matrix
inline std::vector<std::pair<Index, Index>>
layouts(Index m, Index n)
{
    std::vector<std::pair<Index, Index>> l;
    l.push_back(std::make_pair(Index(1), m + 3));     // column major, ld > m
    l.push_back(std::make_pair(n + 2, Index(1)));     // row major, ld > n
    l.push_back(std::make_pair(Index(2), 2 * m + 1)); // every second row
    l.push_back(std::make_pair(3 * n + 1, Index(3))); // neither stride 1
    l.push_back(std::make_pair(Index(-2), -(2 * m + 5))); // negative strides
    return l;
}

template <typename T>
void
fill(Full &A0, Strided<T> &A, Random &random)
{
    for (Index j = 0; j < A.n; ++j) {
	for (Index i = 0; i < A.m; ++i) {
	    const T x = random.value<T>();
	    A(i, j) = x;
	    A0(int(i), int(j)) = toCdouble(x);
	}
    }
}
} // namespace lapacktest
} // namespace test

#endif // ULMBLAS_TEST_NUMERIC_LAPACKTEST_H
