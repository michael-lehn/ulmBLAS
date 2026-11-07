//
//  Correctness tests of ulmlapack_dgetrf and ulmlapack_dpotrf against the
//  LAPACK routines of a reference library (OpenBLAS: dgetrf_, dpotrf_).
//
//      test [quick]
//
//  For every matrix the result of ulmlapack is checked by itself (residual
//  of the factorization, |L| <= 1, valid pivot indices, untouched
//  triangle) and compared with the reference library (INFO, pivots where
//  they are determined).  The parameters (tile size, inner block, panel
//  threads, threads) are varied; serial_n = 0 forces the task-based
//  algorithm also for small matrices.
//
//  Every test runs once per storage layout: through the C interface
//  (column-major, ldA) and through the C++ interface (ulmlapack.hpp) with
//  other row and column strides.  For the C++ interface the column-major
//  test matrix is copied into a strided array whose gaps hold a marker
//  value, the result is copied back, and all checks are the same; in
//  addition the gaps must be untouched (bitwise).
//

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <ulmlapack.h>
#include <ulmlapack.hpp>

extern "C" {
void dgetrf_(const int *m, const int *n, double *A, const int *ldA,
             int *ipiv, int *info);
void dpotrf_(const char *uplo, const int *n, double *A, const int *ldA,
             int *info, std::size_t len);
}

namespace {

const double EPS    = 2.220446049250313e-16;
const double THRESH = 30;           // as in LAPACK's test programs

long nPass = 0, nFail = 0, nPivDiff = 0, nPivCmp = 0;
double maxRatio = 0;
bool verbose = false;

std::mt19937_64 rng(4711);

double
urand()
{
    return std::uniform_real_distribution<double>(-1, 1)(rng);
}

void
check(bool ok, const std::string &what)
{
    if (ok) {
        ++nPass;
        if (verbose) {
            std::printf("PASS %s\n", what.c_str());
        }
    } else {
        ++nFail;
        std::printf("FAIL %s\n", what.c_str());
    }
}

//
//  Storage layouts: (incRow, incCol) as functions of (m, n).
//
struct Layout
{
    const char  *name;
    bool        cInterface;
    long        (*incRow)(long m, long n);
    long        (*incCol)(long m, long n);
};

const Layout layouts[] = {
    {"C",       true,  [](long, long) { return 1L; },
                       [](long m, long) { return std::max(1L, m); }},
    {"cm",      false, [](long, long) { return 1L; },
                       [](long m, long) { return std::max(1L, m)+3; }},
    {"inc2",    false, [](long, long) { return 2L; },
                       [](long m, long) { return 2*std::max(1L, m)+1; }},
    {"inc42",   false, [](long, long) { return 42L; },
                       [](long m, long) { return 42*std::max(1L, m); }},
    {"rm",      false, [](long, long n) { return std::max(1L, n); },
                       [](long, long) { return 1L; }},
    {"rm-gap",  false, [](long, long n) { return 3*std::max(1L, n)+1; },
                       [](long, long) { return 3L; }},
};

const Layout *layout = &layouts[0];

//  Variants of the C++ interface: in place, on a column-major copy, on the
//  tile layout (dpotrf only; the TileMatrix API as a variant of its own)
enum Variant { INPLACE, COLMAJOR, TILED, TILEMATRIX };
const char *variantName[] = {"", "/colmajor", "/tiled", "/TileMatrix"};
Variant variant = INPLACE;
bool    prepack = false;        // dpotrf: param prepack = 1 (packed tiles)
int     outer   = 0;            // dpotrf: param outer (outer blocking)
int     window  = 1;            //         param window
bool    strip   = false;        //         param strip

//  the (layout, variant, prepack, outer) combinations that are tested
const struct { const char *layout; Variant variant; bool prepack; int outer;
               int window; bool strip; }
runs[] = {
    {"C", INPLACE, false}, {"cm", INPLACE, false}, {"inc2", INPLACE, false},
    {"inc42", INPLACE, false}, {"rm", INPLACE, false},
    {"rm-gap", INPLACE, false},
    {"inc2", COLMAJOR, false}, {"inc42", COLMAJOR, false},
    {"rm", COLMAJOR, false},
    {"cm", TILED, false}, {"inc42", TILED, false}, {"rm-gap", TILED, false},
    {"cm", TILEMATRIX, false}, {"inc2", TILEMATRIX, false},
    {"C", INPLACE, true}, {"rm-gap", INPLACE, true},
    {"inc42", COLMAJOR, true}, {"cm", TILED, true}, {"inc2", TILEMATRIX, true},
    {"C", INPLACE, false, 2}, {"inc42", INPLACE, false, 3},
    {"rm-gap", INPLACE, false, 4}, {"cm", INPLACE, false, 7},
    {"inc42", INPLACE, false, 3, 1, true}, {"rm-gap", INPLACE, false, 2, 3, false},
    {"inc2", INPLACE, false, 4, 2, true}, {"C", INPLACE, false, 2, 4, true},
};

//  marker for the gaps of a strided array
const double GAP = -1234.5678;

//  copy the column-major m x n matrix A (ldA) into a strided array
std::vector<double>
toStrided(long m, long n, const double *A, long ldA, long ir, long ic)
{
    const long extent = (m>0 && n>0) ? (m-1)*ir+(n-1)*ic+1 : 1;
    std::vector<double> S(extent, GAP);
    for (long j=0; j<n; ++j) {
        for (long i=0; i<m; ++i) {
            S[i*ir+j*ic] = A[i+j*ldA];
        }
    }
    return S;
}

//  copy back; returns false if a gap has been changed
bool
fromStrided(long m, long n, double *A, long ldA, long ir, long ic,
            std::vector<double> &S)
{
    for (long j=0; j<n; ++j) {
        for (long i=0; i<m; ++i) {
            A[i+j*ldA] = S[i*ir+j*ic];
            S[i*ir+j*ic] = GAP;
        }
    }
    for (double x: S) {
        if (std::memcmp(&x, &GAP, sizeof(double))) {
            return false;
        }
    }
    return true;
}

struct Config
{
    ulmlapack_param p;
    std::string     name;
};

std::vector<Config>
configs(bool quick, int maxThreads)
{
    std::vector<Config> c;
    const int nbs[]  = {8, 16, 33, 64, 100};
    const int ibs[]  = {4, 8, 32};
    const int pts[]  = {1, 2, 4, 16};
    std::vector<int> ths = {1, 2, 3, 4, 8, 16, 32};
    ths.erase(std::remove_if(ths.begin(), ths.end(),
                             [=](int t) { return t>maxThreads; }),
              ths.end());
    // default parameters (no override)
    {
        Config d;
        ulmlapack_param_init(&d.p);
        d.name = "default";
        c.push_back(d);
    }
    int idx = 0;
    for (int nb: nbs) {
        for (int th: ths) {
            ++idx;
            if (quick && idx%3) {
                continue;
            }
            Config d;
            ulmlapack_param_init(&d.p);
            d.p.nb            = nb;
            d.p.ib            = ibs[idx%3];
            d.p.panel_threads = pts[idx%4];
            d.p.threads       = th;
            d.p.serial_n      = 0;
            char buf[128];
            std::snprintf(buf, sizeof(buf), "nb=%d ib=%d pt=%d th=%d",
                          nb, d.p.ib, d.p.panel_threads, th);
            d.name = buf;
            c.push_back(d);
        }
    }
    return c;
}

double
norm1(int m, int n, const double *A, int ldA)
{
    double nrm = 0;
    for (int j=0; j<n; ++j) {
        double s = 0;
        for (int i=0; i<m; ++i) {
            s += std::fabs(A[i+j*ldA]);
        }
        nrm = std::max(nrm, s);
    }
    return nrm;
}

//  ||P A - L U||_1 / (||A||_1 max(m,n) eps); LU contains the factors
double
getrfResidual(int m, int n, const double *A0, int ldA0,
              const double *LU, int ldLU, const int *ipiv)
{
    const int k = std::min(m, n);
    // PA: apply interchanges to a copy of A0
    std::vector<double> PA(std::size_t(m)*n);
    for (int j=0; j<n; ++j) {
        for (int i=0; i<m; ++i) {
            PA[i+std::size_t(j)*m] = A0[i+std::size_t(j)*ldA0];
        }
    }
    for (int i=0; i<k; ++i) {
        const int p = ipiv[i]-1;
        if (p!=i) {
            for (int j=0; j<n; ++j) {
                std::swap(PA[i+std::size_t(j)*m], PA[p+std::size_t(j)*m]);
            }
        }
    }
    // PA - L U, L m x k unit lower, U k x n upper
    for (int j=0; j<n; ++j) {
        for (int l=0; l<std::min(j+1, k); ++l) {
            const double u = LU[l+std::size_t(j)*ldLU];
            if (u==0) {
                continue;
            }
            PA[l+std::size_t(j)*m] -= u;                 // L(l,l) = 1
            for (int i=l+1; i<m; ++i) {
                PA[i+std::size_t(j)*m] -= LU[i+std::size_t(l)*ldLU]*u;
            }
        }
    }
    const double nA = norm1(m, n, A0, ldA0);
    const double nR = norm1(m, n, PA.data(), m);
    if (nA==0) {
        return nR==0 ? 0 : 1e300;
    }
    return nR/(nA*std::max(m, n)*EPS);
}

//
//  DGETRF
//
enum GeKind { GE_RANDOM, GE_PLU, GE_ZEROCOL, GE_ZERO, GE_RANKDEF };

const char *geName[] = {"random", "PLU", "zerocol", "zero", "zerocols"};

void
makeGe(GeKind kind, int m, int n, std::vector<double> &A, int ldA,
       std::vector<int> &expectPiv, int &expectInfo)
{
    A.assign(std::size_t(ldA)*n, NAN);         // padding must stay NaN
    expectPiv.clear();
    expectInfo = -1;                           // unknown
    const int k = std::min(m, n);

    if (kind==GE_RANDOM || kind==GE_ZEROCOL || kind==GE_RANKDEF) {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                A[i+std::size_t(j)*ldA] = urand();
            }
        }
        if (kind==GE_ZEROCOL && k>0) {
            // a zero column j: U(j,j) = 0 exactly (unless an earlier pivot
            // is zero), and the factorization continues
            const int j = k/2;
            for (int i=0; i<m; ++i) {
                A[i+std::size_t(j)*ldA] = 0;
            }
            expectInfo = j+1;
        }
        if (kind==GE_RANKDEF && k>0) {
            // first and last column zero: INFO = 1, the factorization
            // continues with the other columns
            for (int i=0; i<m; ++i) {
                A[i] = 0;
                A[i+std::size_t(k-1)*ldA] = 0;
            }
            expectInfo = 1;
        }
    } else if (kind==GE_ZERO) {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                A[i+std::size_t(j)*ldA] = 0;
            }
        }
        expectInfo = (k>0) ? 1 : 0;
        for (int i=0; i<k; ++i) {
            expectPiv.push_back(i+1);
        }
    } else if (kind==GE_PLU) {
        // A = P0 L0 U0 with |L0(i,j)| <= 1/2 (i > j): partial pivoting
        // must find P0 exactly (the pivots are unique with a margin of 2).
        // For larger m the off-diagonal entries of L0 and U0 are scaled
        // down (by 4/sqrt(m)), otherwise L0 and U0 are so ill-conditioned that the rounding errors in
        // the late Schur complements exceed the margin.
        const double ls = 0.5*std::min(1.0, 4/std::sqrt(double(m)));
        std::vector<double> L(std::size_t(m)*k, 0.0), U(std::size_t(k)*n, 0.0);
        for (int j=0; j<k; ++j) {
            L[j+std::size_t(j)*m] = 1;
            for (int i=j+1; i<m; ++i) {
                L[i+std::size_t(j)*m] = ls*urand();
            }
        }
        for (int j=0; j<n; ++j) {
            for (int i=0; i<=std::min(j, k-1); ++i) {
                U[i+std::size_t(j)*k] = 2*ls*urand();
            }
            if (j<k) {
                U[j+std::size_t(j)*k] = (urand()<0 ? -1 : 1)*(1+std::fabs(urand()));
            }
        }
        std::vector<int> perm(m);
        for (int i=0; i<m; ++i) {
            perm[i] = i;
        }
        std::shuffle(perm.begin(), perm.end(), rng);
        // row perm[i] of A is row i of L0 U0
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                double s = 0;
                for (int l=0; l<=std::min(std::min(i, j), k-1); ++l) {
                    s += L[i+std::size_t(l)*m]*U[l+std::size_t(j)*k];
                }
                A[perm[i]+std::size_t(j)*ldA] = s;
            }
        }
        // expected ipiv: simulate the interchanges on the permutation
        std::vector<int> where(m);              // row of A holding LU row i
        std::vector<int> rowOf(m);              // LU row held by row r of A
        for (int i=0; i<m; ++i) {
            where[i] = perm[i];
            rowOf[perm[i]] = i;
        }
        for (int j=0; j<k; ++j) {
            const int p = where[j];             // pivot: the row holding LU row j
            expectPiv.push_back(p+1);
            const int lj = rowOf[j];
            std::swap(rowOf[j], rowOf[p]);
            where[lj] = p;
            where[j]  = j;
        }
        expectInfo = 0;
    }
}

void
testGetrf(const Config &cfg, int m, int n, int pad, GeKind kind)
{
    if (variant==TILED || variant==TILEMATRIX || prepack || outer) {
        return;                 // dgetrf: no tile layout, no prepack (yet)
    }
    const int ldA = std::max(1, m+pad);
    std::vector<double> A0;
    std::vector<int>    expectPiv;
    int                 expectInfo;
    makeGe(kind, m, n, A0, ldA, expectPiv, expectInfo);

    const int k = std::min(m, n);
    std::vector<double> A(A0), R(A0);
    std::vector<int>    ipiv(k+1, -7), rpiv(k+1, -7);

    int  info;
    bool okGaps = true;
    if (layout->cInterface) {
        info = ulmlapack_dgetrf_param(m, n, A.data(), ldA, ipiv.data(), &cfg.p);
    } else {
        const long ir = layout->incRow(m, n), ic = layout->incCol(m, n);
        std::vector<double> S = toStrided(m, n, A.data(), ldA, ir, ic);
        std::vector<ulmlapack::Index> piv(k+1, -8);
        info = (variant==COLMAJOR)
             ? (int) ulmlapack::getrf_colmajor(m, n, S.data(), ir, ic,
                                               piv.data(), 1, &cfg.p)
             : (int) ulmlapack::getrf(m, n, S.data(), ir, ic, piv.data(), 1,
                                      &cfg.p);
        okGaps = fromStrided(m, n, A.data(), ldA, ir, ic, S);
        for (int i=0; i<k; ++i) {
            ipiv[i] = int(piv[i]+1);
        }
        okGaps = okGaps && piv[k]==-8;
    }
    int rinfo;
    dgetrf_(&m, &n, R.data(), &ldA, rpiv.data(), &rinfo);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "dgetrf %s m=%d n=%d ldA=%d [%s] %s%s",
                  geName[kind], m, n, ldA, cfg.name.c_str(), layout->name,
                  variantName[variant]);
    const std::string what(buf);
    check(okGaps, what+" gaps and piv[k] untouched");

    check(info==rinfo, what+" info="+std::to_string(info)
                           +" ref="+std::to_string(rinfo));
    if (expectInfo>=0) {
        check(info==expectInfo, what+" expected info "
                                +std::to_string(expectInfo));
    }
    bool okPiv = true;
    for (int i=0; i<k; ++i) {
        okPiv = okPiv && ipiv[i]>=i+1 && ipiv[i]<=m;
    }
    check(okPiv && ipiv[k]==-7, what+" pivot range");
    if (!expectPiv.empty()) {
        int d = -1;
        for (int i=0; i<k && d<0; ++i) {
            if (expectPiv[i]!=ipiv[i]) {
                d = i;
            }
        }
        check(d<0, what+" pivots = expected"
                   +(d<0 ? std::string()
                         : " (first difference at "+std::to_string(d)
                           +": "+std::to_string(ipiv[d])+" expected "
                           +std::to_string(expectPiv[d])+" reference "
                           +std::to_string(rpiv[d])+")"));
    }
    if (kind==GE_RANDOM || kind==GE_PLU) {
        ++nPivCmp;
        if (!std::equal(ipiv.begin(), ipiv.begin()+k, rpiv.begin())) {
            ++nPivDiff;
            if (kind==GE_PLU) {
                check(false, what+" pivots differ from reference");
            }
        }
    }
    // |L| <= 1
    bool okL = true;
    for (int j=0; j<k; ++j) {
        for (int i=j+1; i<m; ++i) {
            okL = okL && std::fabs(A[i+std::size_t(j)*ldA])<=1;
        }
    }
    check(okL, what+" |L| <= 1");
    // padding untouched
    bool okPad = true;
    for (int j=0; j<n; ++j) {
        for (int i=m; i<ldA; ++i) {
            okPad = okPad && std::isnan(A[i+std::size_t(j)*ldA]);
        }
    }
    check(okPad, what+" padding untouched");

    const double res = getrfResidual(m, n, A0.data(), ldA, A.data(), ldA,
                                     ipiv.data());
    maxRatio = std::max(maxRatio, res);
    std::snprintf(buf, sizeof(buf), " residual %.2f", res);
    check(res<THRESH, what+buf);
}

//
//  DPOTRF
//
//  A = Q diag(lambda) Q^T, Q a product of three Householder reflections,
//  lambda geometrically from 1 to 1/cond.  Optionally the entry (j,j) of
//  an SPD matrix is made very negative: the leading minor of order j+1 is
//  then not positive, those of order <= j are.
//
void
makeSpd(int n, double cond, std::vector<double> &A, int ldA, int negAt)
{
    A.assign(std::size_t(ldA)*n, NAN);
    std::vector<double> M(std::size_t(n)*n, 0.0);
    for (int i=0; i<n; ++i) {
        M[i+std::size_t(i)*n] = (n==1) ? 1 : std::pow(cond, -double(i)/(n-1));
    }
    for (int h=0; h<3; ++h) {
        std::vector<double> v(n);
        double nv = 0;
        for (int i=0; i<n; ++i) {
            v[i] = urand();
            nv += v[i]*v[i];
        }
        nv = std::sqrt(nv);
        for (int i=0; i<n; ++i) {
            v[i] /= nv;
        }
        // M = H M H, H = I - 2 v v^T
        std::vector<double> w(n);
        for (int pass=0; pass<2; ++pass) {
            // M = H M  (columns), then transpose trick: M symmetric
            for (int j=0; j<n; ++j) {
                double s = 0;
                for (int i=0; i<n; ++i) {
                    s += v[i]*M[i+std::size_t(j)*n];
                }
                for (int i=0; i<n; ++i) {
                    M[i+std::size_t(j)*n] -= 2*s*v[i];
                }
            }
            // transpose (H M)^T = M H
            for (int j=0; j<n; ++j) {
                for (int i=j+1; i<n; ++i) {
                    std::swap(M[i+std::size_t(j)*n], M[j+std::size_t(i)*n]);
                }
            }
        }
    }
    // symmetrize exactly
    for (int j=0; j<n; ++j) {
        for (int i=j+1; i<n; ++i) {
            const double s = 0.5*(M[i+std::size_t(j)*n]+M[j+std::size_t(i)*n]);
            M[i+std::size_t(j)*n] = M[j+std::size_t(i)*n] = s;
        }
    }
    if (negAt>=0) {
        M[negAt+std::size_t(negAt)*n] = -1e3;
    }
    for (int j=0; j<n; ++j) {
        for (int i=0; i<n; ++i) {
            A[i+std::size_t(j)*ldA] = M[i+std::size_t(j)*n];
        }
    }
}

//  ||A - L L^T||_1 / (||A||_1 n eps) on the triangle uplo
double
potrfResidual(bool lower, int n, const double *A0, const double *F, int ldA)
{
    std::vector<double> L(std::size_t(n)*n, 0.0), S(std::size_t(n)*n, 0.0);
    for (int j=0; j<n; ++j) {
        for (int i=j; i<n; ++i) {
            L[i+std::size_t(j)*n] = lower ? F[i+std::size_t(j)*ldA]
                                          : F[j+std::size_t(i)*ldA];
        }
    }
    for (int j=0; j<n; ++j) {
        for (int i=j; i<n; ++i) {
            double s = 0;
            for (int l=0; l<=j; ++l) {
                s += L[i+std::size_t(l)*n]*L[j+std::size_t(l)*n];
            }
            const double a = A0[i+std::size_t(j)*ldA];
            S[i+std::size_t(j)*n] = S[j+std::size_t(i)*n] = a-s;
        }
    }
    std::vector<double> Afull(std::size_t(n)*n);
    for (int j=0; j<n; ++j) {
        for (int i=0; i<n; ++i) {
            Afull[i+std::size_t(j)*n] = A0[std::max(i, j)+std::size_t(std::min(i, j))*ldA];
        }
    }
    const double nA = norm1(n, n, Afull.data(), n);
    return norm1(n, n, S.data(), n)/(nA*n*EPS);
}

void
testPotrf(const Config &cfg, bool lower, int n, int pad, double cond,
          int negAt)
{
    const int ldA = std::max(1, n+pad);
    std::vector<double> A0;
    makeSpd(n, cond, A0, ldA, negAt);
    std::vector<double> A(A0), R(A0);
    const char uplo = lower ? 'L' : 'U';

    int  info;
    bool okGaps = true;
    ulmlapack_param pp = cfg.p;
    pp.prepack = prepack;
    pp.outer   = outer;
    pp.window  = window;
    pp.strip   = strip;
    if (layout->cInterface) {
        info = ulmlapack_dpotrf_param(uplo, n, A.data(), ldA, &pp);
    } else {
        const long ir = layout->incRow(n, n), ic = layout->incCol(n, n);
        std::vector<double> S = toStrided(n, n, A.data(), ldA, ir, ic);
        switch (variant) {
        case COLMAJOR:
            info = (int) ulmlapack::potrf_colmajor(lower, n, S.data(), ir, ic,
                                                   &pp);
            break;
        case TILED:
            info = (int) ulmlapack::potrf_tiled(lower, n, S.data(), ir, ic,
                                                &pp);
            break;
        case TILEMATRIX:
            {
                // tile size as the routine would choose it; upper: the
                // transposed view (TileMatrix holds the lower triangle)
                ulmlapack_param p = cfg.p;
                ulmlapack_dpotrf_resolve(std::max(1, n), &p);
                const long tr = lower ? ir : ic, tc = lower ? ic : ir;
                ulmlapack::TileMatrix TM(n, n, p.nb);
                TM.assign(S.data(), tr, tc);
                info = (int) ulmlapack::potrf(TM, &pp);
                TM.copyTo(S.data(), tr, tc);
            }
            break;
        default:
            info = (int) ulmlapack::potrf(lower, n, S.data(), ir, ic, &pp);
        }
        okGaps = fromStrided(n, n, A.data(), ldA, ir, ic, S);
    }
    int rinfo;
    dpotrf_(&uplo, &n, R.data(), &ldA, &rinfo, 1);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "dpotrf %c n=%d ldA=%d cond=%.0e neg=%d [%s] %s%s%s%s",
                  uplo, n, ldA, cond, negAt, cfg.name.c_str(), layout->name,
                  variantName[variant], prepack ? "+prepack" : "",
                  outer ? ("+outer"+std::to_string(outer)
                           +(window>1 ? "+w"+std::to_string(window) : "")
                           +(strip ? "+strip" : "")).c_str() : "");
    const std::string what(buf);
    check(okGaps, what+" gaps untouched");

    check(info==rinfo, what+" info="+std::to_string(info)
                           +" ref="+std::to_string(rinfo));
    if (negAt>=0) {
        check(info==negAt+1, what+" expected info "+std::to_string(negAt+1));
    }
    // other triangle and padding untouched (bitwise; NaN == NaN here)
    bool okOther = true;
    for (int j=0; j<n; ++j) {
        for (int i=0; i<ldA; ++i) {
            const bool inTri = i<n && (lower ? i>=j : i<=j);
            if (!inTri) {
                okOther = okOther
                       && std::memcmp(&A[i+std::size_t(j)*ldA],
                                      &A0[i+std::size_t(j)*ldA],
                                      sizeof(double))==0;
            }
        }
    }
    check(okOther, what+" other triangle untouched");

    if (info==0 && n>0) {
        const double res = potrfResidual(lower, n, A0.data(), A.data(), ldA);
        maxRatio = std::max(maxRatio, res);
        std::snprintf(buf, sizeof(buf), " residual %.2f", res);
        check(res<THRESH, what+buf);
        // compare with the reference factor (well conditioned only)
        if (cond<=1e6) {
            double d = 0, nr = 0;
            for (int j=0; j<n; ++j) {
                for (int i=0; i<n; ++i) {
                    if (lower ? i>=j : i<=j) {
                        d  = std::max(d, std::fabs(A[i+std::size_t(j)*ldA]
                                                   -R[i+std::size_t(j)*ldA]));
                        nr = std::max(nr, std::fabs(R[i+std::size_t(j)*ldA]));
                    }
                }
            }
            std::snprintf(buf, sizeof(buf), " max diff to reference %.1e", d/nr);
            check(d<=nr*1e4*n*EPS*std::sqrt(cond), what+buf);
        }
    }
}

} // namespace

int
main(int argc, char **argv)
{
    bool quick = false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i], "quick")) {
            quick = true;
        } else if (!std::strcmp(argv[i], "-v")) {
            verbose = true;
        }
    }
    const char *mt = std::getenv("TEST_MAX_THREADS");
    const int maxThreads = mt ? std::atoi(mt) : 8;

    const std::vector<Config> cfgs = configs(quick, maxThreads);

    const int sizes[][2] = {
        {0, 0}, {1, 1}, {2, 2}, {7, 7}, {16, 16}, {33, 33}, {64, 64},
        {100, 100}, {129, 129}, {200, 200}, {257, 257}, {400, 400},
        {300, 120}, {500, 64}, {1000, 40},      // tall
        {120, 300}, {64, 500}, {33, 100},       // wide
    };
    for (const auto &run: runs) {
    for (const Layout &lay: layouts) {
        if (!std::strcmp(lay.name, run.layout)) {
            layout = &lay;
        }
    }
    variant = run.variant;
    prepack = run.prepack;
    outer   = run.outer;
    window  = std::max(1, run.window);
    strip   = run.strip;
    for (const Config &c: cfgs) {
        for (auto &mn: sizes) {
            const int m = mn[0], n = mn[1];
            for (int pad: {0, 3}) {
                testGetrf(c, m, n, pad, GE_RANDOM);
                if (m==n && pad==0) {
                    testGetrf(c, m, n, pad, GE_PLU);
                    testGetrf(c, m, n, pad, GE_ZEROCOL);
                    testGetrf(c, m, n, pad, GE_ZERO);
                    testGetrf(c, m, n, pad, GE_RANKDEF);
                }
            }
            if (m!=n) {
                testGetrf(c, m, n, 1, GE_PLU);
                testGetrf(c, m, n, 0, GE_ZEROCOL);
            }
        }
        // larger: only the default parameters and one task configuration
        if (&c==&cfgs[0] || &c==&cfgs[cfgs.size()/2]) {
            testGetrf(c, 700, 700, 5, GE_RANDOM);
            testGetrf(c, 700, 700, 0, GE_PLU);
            testGetrf(c, 650, 400, 0, GE_RANDOM);
        }
        if (&c==&cfgs[0] && !quick) {
            testGetrf(c, 1500, 1500, 1, GE_RANDOM);
            testGetrf(c, 1500, 1500, 0, GE_PLU);
            testPotrf(c, true, 1500, 1, 1e4, -1);
            testPotrf(c, false, 1500, 0, 1e2, 1234);
        }
        for (int n: {0, 1, 2, 7, 16, 33, 64, 100, 129, 200, 257, 400}) {
            for (bool lower: {true, false}) {
                for (double cond: {1e1, 1e6, 1e12}) {
                    testPotrf(c, lower, n, (n%2)*3, cond, -1);
                }
                if (n>0) {
                    testPotrf(c, lower, n, 0, 1e2, n/2);
                    testPotrf(c, lower, n, 1, 1e2, 0);
                    testPotrf(c, lower, n, 2, 1e2, n-1);
                }
            }
        }
        if (&c==&cfgs[0] || &c==&cfgs[cfgs.size()/2]) {
            testPotrf(c, true, 700, 5, 1e8, -1);
            testPotrf(c, false, 700, 0, 1e3, 650);
        }
    }
    }
    // invalid arguments
    {
        double a[4];
        int ip[2];
        check(ulmlapack_dgetrf(-1, 2, a, 2, ip)==-1, "dgetrf m<0");
        check(ulmlapack_dgetrf(2, -1, a, 2, ip)==-2, "dgetrf n<0");
        check(ulmlapack_dgetrf(2, 2, a, 1, ip)==-4, "dgetrf ldA<m");
        check(ulmlapack_dpotrf('X', 2, a, 2)==-1, "dpotrf uplo");
        check(ulmlapack_dpotrf('L', -1, a, 2)==-2, "dpotrf n<0");
        check(ulmlapack_dpotrf('L', 2, a, 1)==-4, "dpotrf ldA<n");
    }
    long panels, teams, members;
    ulmlapack_dgetrf_stats(&panels, &teams, &members);
    std::printf("dgetrf panels in the task algorithm: %ld, by a team: %ld "
                "(average size %.1f)\n", panels, teams,
                teams ? double(members)/teams : 0.0);
    std::printf("layouts x variants %zu, configurations %zu, checks: %ld PASS, %ld FAIL; "
                "max residual ratio %.2f (threshold %.0f); "
                "pivots differing from the reference on random/PLU matrices: "
                "%ld of %ld\n",
                sizeof(runs)/sizeof(runs[0]), cfgs.size(), nPass, nFail, maxRatio, THRESH, nPivDiff, nPivCmp);
    return nFail ? 1 : 0;
}
