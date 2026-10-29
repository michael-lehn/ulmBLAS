#ifndef ULMBLAS_TEST_NUMERIC_REFERENCE_H
#define ULMBLAS_TEST_NUMERIC_REFERENCE_H 1

//
//  Straightforward reference implementations used by the numerical tests.
//
//  All operands are converted to dense matrices (Full) of complex<double>,
//  with transposition, symmetry or triangular structure already applied.
//  Reference results are computed in complex<double> arithmetic together
//  with a bound sum |terms| for the error check (see test::closeEnough).
//
//  BLAS conventions: if alpha is zero, A and B are not referenced; if beta
//  is zero, C (y) need not be set on input (it may contain NaN).
//

#include <test/numeric/blas.h>
#include <test/numeric/testing.h>

namespace test {

struct Full
{
    Full(int m_=0, int n_=0)
        : m(m_), n(n_), a(std::size_t(std::max(1, m_*n_)), cdouble(0))
    {
    }

    cdouble &
    operator()(int i, int j)
    {
        return a[i+j*m];
    }

    const cdouble &
    operator()(int i, int j) const
    {
        return a[i+j*m];
    }

    int                   m, n;
    std::vector<cdouble>  a;
};

template <typename T>
cdouble
toCdouble(const T &x)
{
    return cdouble(double(std::real(x)), double(std::imag(x)));
}

inline float   fromCdouble(const cdouble &x, float *)   { return float(x.real()); }
inline double  fromCdouble(const cdouble &x, double *)  { return x.real(); }
inline cfloat  fromCdouble(const cdouble &x, cfloat *)
{ return cfloat(float(x.real()), float(x.imag())); }
inline cdouble fromCdouble(const cdouble &x, cdouble *) { return x; }

template <typename T>
T
fromCdouble(const cdouble &x)
{
    return fromCdouble(x, static_cast<T *>(0));
}

inline char
transChar(CBLAS_TRANSPOSE trans)
{
    return trans==CblasNoTrans ? 'N' : trans==CblasTrans ? 'T' : 'C';
}

inline const char *
orderName(CBLAS_ORDER order)
{
    return order==CblasColMajor ? "col" : "row";
}

inline CBLAS_ORDER
order(bool rowMajor)
{
    return rowMajor ? CblasRowMajor : CblasColMajor;
}

//
//  op(A) as dense matrix
//
template <typename T>
Full
full(const Matrix<T> &A, CBLAS_TRANSPOSE trans = CblasNoTrans)
{
    bool  t = trans!=CblasNoTrans;
    Full  F(t ? A.n : A.m, t ? A.m : A.n);

    for (int j=0; j<A.n; ++j) {
        for (int i=0; i<A.m; ++i) {
            cdouble aij = toCdouble(A(i,j));
            if (trans==CblasNoTrans) {
                F(i,j) = aij;
            } else if (trans==CblasTrans) {
                F(j,i) = aij;
            } else {
                F(j,i) = std::conj(aij);
            }
        }
    }
    return F;
}

//
//  Symmetric (hermitian==false) or hermitian matrix of which only the
//  triangle upLo of A is referenced.  For hermitian matrices the imaginary
//  parts of the diagonal are assumed to be zero.
//
template <typename T>
Full
fullSymmetric(const Matrix<T> &A, CBLAS_UPLO upLo, bool hermitian)
{
    int   n = A.n;
    Full  F(n, n);

    for (int j=0; j<n; ++j) {
        for (int i=0; i<n; ++i) {
            bool    stored = (upLo==CblasLower) ? i>=j : i<=j;
            cdouble aij    = stored ? toCdouble(A(i,j)) : toCdouble(A(j,i));
            if (hermitian && !stored) {
                aij = std::conj(aij);
            }
            if (hermitian && i==j) {
                aij = cdouble(aij.real(), 0);
            }
            F(i,j) = aij;
        }
    }
    return F;
}

//
//  op(A) for a triangular matrix A, of which only the triangle upLo (without
//  the diagonal if diag==CblasUnit) is referenced.
//
template <typename T>
Full
fullTriangular(const Matrix<T> &A, CBLAS_UPLO upLo, CBLAS_DIAG diag,
               CBLAS_TRANSPOSE trans)
{
    int   n = A.n;
    Full  F(n, n);

    for (int j=0; j<n; ++j) {
        for (int i=0; i<n; ++i) {
            cdouble aij(0);
            if (i==j) {
                aij = (diag==CblasUnit) ? cdouble(1) : toCdouble(A(i,i));
            } else if ((upLo==CblasLower) ? i>j : i<j) {
                aij = toCdouble(A(i,j));
            }
            if (trans==CblasNoTrans) {
                F(i,j) = aij;
            } else if (trans==CblasTrans) {
                F(j,i) = aij;
            } else {
                F(j,i) = std::conj(aij);
            }
        }
    }
    return F;
}

template <typename T>
Full
fullVector(const Vector<T> &x)
{
    Full F(x.n, 1);
    for (int i=0; i<x.n; ++i) {
        F(i,0) = toCdouble(x(i));
    }
    return F;
}

//
//  Fill the elements of A that must not be referenced with NaN: the strict
//  triangle opposite to upLo and, for unit triangular matrices, the
//  diagonal.
//
template <typename T>
void
poisonTriangle(Matrix<T> &A, CBLAS_UPLO upLo, bool unitDiag)
{
    for (int j=0; j<A.n; ++j) {
        for (int i=0; i<A.m; ++i) {
            bool other = (upLo==CblasLower) ? i<j : i>j;
            if (other || (unitDiag && i==j)) {
                A(i,j) = nan<T>();
            }
        }
    }
}

//
//  R = alpha*A*B + beta*C and the corresponding error bound
//
struct Result
{
    Full  value, bound;
};

inline Result
multiply(cdouble alpha, const Full &A, const Full &B, cdouble beta,
         const Full &C)
{
    Result r;
    r.value = Full(A.m, B.n);
    r.bound = Full(A.m, B.n);

    for (int j=0; j<B.n; ++j) {
        for (int i=0; i<A.m; ++i) {
            cdouble sum(0);
            double  bound = 0;
            if (alpha!=cdouble(0)) {
                for (int l=0; l<A.n; ++l) {
                    sum   += A(i,l)*B(l,j);
                    bound += std::abs(A(i,l))*std::abs(B(l,j));
                }
                sum   *= alpha;
                bound *= std::abs(alpha);
            }
            if (beta!=cdouble(0)) {
                sum   += beta*C(i,j);
                bound += std::abs(beta)*std::abs(C(i,j));
            }
            r.value(i,j) = sum;
            r.bound(i,j) = bound;
        }
    }
    return r;
}

//
//  Compare the m x n matrix C (or one of its triangles: 'L', 'U'; 'A' for
//  all elements) with a reference result.  k is the number of terms of the
//  inner products.  Returns false after the first mismatch.
//
template <typename T>
bool
checkMatrix(Context &ctx, const Matrix<T> &C, const Result &ref, int k,
            char part, const std::string &what)
{
    for (int j=0; j<C.n; ++j) {
        for (int i=0; i<C.m; ++i) {
            if ((part=='L' && i<j) || (part=='U' && i>j)) {
                continue;
            }
            T expected = fromCdouble<T>(ref.value(i,j));
            if (!closeEnough(C(i,j), expected, ref.bound(i,j).real(), k)) {
                ctx.check(false, "%s: element (%d,%d) is (%g,%g), expected "
                          "(%g,%g), bound %g, %s", what.c_str(), i, j,
                          double(std::real(C(i,j))),
                          double(std::imag(C(i,j))),
                          double(std::real(expected)),
                          double(std::imag(expected)),
                          ref.bound(i,j).real(), typeName<T>());
                return false;
            }
        }
    }
    ctx.pass();
    return true;
}

template <typename T>
bool
checkVector(Context &ctx, const Vector<T> &y, const Result &ref, int k,
            const std::string &what)
{
    for (int i=0; i<y.n; ++i) {
        T expected = fromCdouble<T>(ref.value(i,0));
        if (!closeEnough(y(i), expected, ref.bound(i,0).real(), k)) {
            ctx.check(false, "%s: element %d is (%g,%g), expected (%g,%g), "
                      "bound %g, %s", what.c_str(), i,
                      double(std::real(y(i))), double(std::imag(y(i))),
                      double(std::real(expected)),
                      double(std::imag(expected)),
                      ref.bound(i,0).real(), typeName<T>());
            return false;
        }
    }
    ctx.pass();
    return true;
}

//
//  Description of the parameters of a call for failure messages
//
inline std::string
format(const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)));
#endif

inline std::string
format(const char *fmt, ...)
{
    char     buffer[512];
    va_list  ap;
    va_start(ap, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, ap);
    va_end(ap);
    return buffer;
}

template <typename T>
std::string
str(const T &x)
{
    return IsComplex<T>::value
         ? format("(%g,%g)", double(std::real(x)), double(std::imag(x)))
         : format("%g", double(std::real(x)));
}

//
//  Scalars used for alpha and beta: zero, one and a "random" value.
//
template <typename T>
std::vector<T>
scalars()
{
    std::vector<T> s;
    s.push_back(T(0));
    s.push_back(T(1));
    s.push_back(fromCdouble<T>(cdouble(-0.75, 0.375)));
    return s;
}

} // namespace test

#endif // ULMBLAS_TEST_NUMERIC_REFERENCE_H
