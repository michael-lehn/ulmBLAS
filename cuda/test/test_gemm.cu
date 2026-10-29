//
//  test_gemm: correctness tests of the GEMM implementations of cULMblas
//  against cuBLAS (cublasDgemm).
//
//  For every implementation in support::variants() (and cULMblas::gemm)
//  and every case:
//
//  - A, B and C are column major with leading dimensions that are tight or
//    padded; the padding rows and guard zones of 32 elements before and
//    after every matrix contain a NaN with a fixed bit pattern.  A read of
//    padding or guards makes the result NaN, a write is detected by a
//    bitwise comparison afterwards.  A and B must not change at all.
//  - The result is compared elementwise with cuBLAS:
//
//        |C(i,j) - Cref(i,j)| <= 4 (k+2) eps (|alpha| (|A| |B|)(i,j)
//                                             + |beta| |C0(i,j)|)
//
//    with |A| |B| computed by cuBLAS from the absolute values.  The column
//    err/bound is the maximum of the left side divided by the right side;
//    a case passes if it is at most 1.
//  - alpha == 0 or k == 0: A and B are not referenced (NaN in the test), the
//    expected result is beta*C0 (0 for beta == 0), compared exactly.
//  - beta == 0 with C0 = NaN: C must not be read.
//
//  The matrices are pseudo random with a fixed seed (--seed), so every case
//  is reproducible.  Options: --quick (small cases only, e.g. under
//  compute-sanitizer), --variants a,b (implementations), --quiet (only
//  failures and the summary).
//
//  External references (MAGMA, if built with it) run through the same cases;
//  their results that differ from BLAS are listed as DIFF and summarized,
//  but only the implementations of cULMblas decide the exit status.  The
//  check that invalid arguments throw applies to cULMblas only.
//
#include <support/support.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace cULMblas::support;

namespace {

const std::ptrdiff_t Guard = 32;

//  quiet NaN with a payload: padding and guards
double
sentinel()
{
    std::uint64_t bits = 0x7ff8dead0000beefULL;
    double x;
    std::memcpy(&x, &bits, sizeof(x));
    return x;
}

bool
sameBits(double x, double y)
{
    return std::memcmp(&x, &y, sizeof(x)) == 0;
}

enum Init { Random, NanC, NanAB };

struct Case
{
    std::ptrdiff_t  m, n, k, ldA, ldB, ldC;
    double          alpha, beta;
    Init            init;
};

const char *
initName(Init init)
{
    return init == NanC ? "C=NaN" : init == NanAB ? "A,B=NaN" : "random";
}

//
//  A column major matrix with leading dimension ld inside a buffer with
//  guard zones; element (i,j) is buf[Guard + i + j*ld].
//
struct HostMatrix
{
    HostMatrix(std::ptrdiff_t m_, std::ptrdiff_t n_, std::ptrdiff_t ld_)
        : m(m_), n(n_), ld(ld_), buf(2*Guard + ld_*n_, sentinel())
    {
    }

    double &
    operator()(std::ptrdiff_t i, std::ptrdiff_t j)
    {
        return buf[Guard + i + j*ld];
    }

    std::ptrdiff_t       m, n, ld;
    std::vector<double>  buf;
};

void
fill(HostMatrix &X, std::mt19937_64 &gen, bool nan)
{
    std::uniform_real_distribution<double> uniform(-1, 1);
    for (std::ptrdiff_t j = 0; j < X.n; ++j) {
        for (std::ptrdiff_t i = 0; i < X.m; ++i) {
            X(i, j) = nan ? std::numeric_limits<double>::quiet_NaN()
                          : uniform(gen);
        }
    }
}

//  padding and guards unchanged (elements outside the matrix)
bool
outsideUnchanged(const HostMatrix &X, const std::vector<double> &after)
{
    for (std::ptrdiff_t p = 0; p < std::ptrdiff_t(after.size()); ++p) {
        std::ptrdiff_t q = p - Guard;
        bool inside = q >= 0 && q < X.ld*X.n && q % X.ld < X.m;
        if (!inside && !sameBits(after[p], X.buf[p])) {
            return false;
        }
    }
    return true;
}

bool
allUnchanged(const std::vector<double> &before,
             const std::vector<double> &after)
{
    return before.size() == after.size()
        && std::memcmp(before.data(), after.data(),
                       before.size()*sizeof(double)) == 0;
}

struct Result
{
    bool         pass;
    double       ratio;     // max err/bound (0 for exact comparisons)
    std::string  reason;
};

Result
runCase(const Variant &variant, Cublas &cublas, const Case &c,
        std::uint64_t seed)
{
    std::mt19937_64 gen(seed);
    HostMatrix A(c.m, c.k, c.ldA), B(c.k, c.n, c.ldB), C0(c.m, c.n, c.ldC);
    fill(A, gen, c.init == NanAB);
    fill(B, gen, c.init == NanAB);
    fill(C0, gen, c.init == NanC);

    DeviceBuffer dA(A.buf.size()), dB(B.buf.size()), dC(C0.buf.size());
    dA.upload(A.buf);
    dB.upload(B.buf);
    dC.upload(C0.buf);

    Result r = { true, 0, "" };
    try {
        variant.gemm(c.m, c.n, c.k, c.alpha, dA.data + Guard, c.ldA,
                     dB.data + Guard, c.ldB, c.beta, dC.data + Guard, c.ldC,
                     0);
        CULMBLAS_CHECK_CUDA(cudaDeviceSynchronize());
    } catch (const std::exception &e) {
        return Result{ false, 0, e.what() };
    }

    std::vector<double> Cbuf = dC.download();
    if (!allUnchanged(A.buf, dA.download())
     || !allUnchanged(B.buf, dB.download()))
    {
        return Result{ false, 0, "A or B modified" };
    }
    if (!outsideUnchanged(C0, Cbuf)) {
        return Result{ false, 0, "write outside of C (padding or guard)" };
    }
    HostMatrix Cv(c.m, c.n, c.ldC);
    Cv.buf = Cbuf;

    if (c.m == 0 || c.n == 0) {
        return r;
    }

    //  A and B not referenced: C = beta*C0, exactly
    if (c.alpha == 0 || c.k == 0) {
        for (std::ptrdiff_t j = 0; j < c.n; ++j) {
            for (std::ptrdiff_t i = 0; i < c.m; ++i) {
                double expected = c.beta == 0 ? 0 : c.beta*C0(i, j);
                if (!(Cv(i, j) == expected)) {
                    char msg[100];
                    std::snprintf(msg, sizeof(msg), "C(%td,%td) = %g, "
                                  "expected %g", i, j, Cv(i, j), expected);
                    return Result{ false, 0, msg };
                }
            }
        }
        return r;
    }

    //  reference: cuBLAS on the same input, and |A| |B| for the bound
    DeviceBuffer dCref(C0.buf.size());
    dCref.upload(C0.buf);
    cublas.gemm(c.m, c.n, c.k, c.alpha, dA.data + Guard, c.ldA,
                dB.data + Guard, c.ldB, c.beta, dCref.data + Guard, c.ldC);

    HostMatrix absA(c.m, c.k, c.ldA), absB(c.k, c.n, c.ldB);
    std::fill(absA.buf.begin(), absA.buf.end(), 0.0);
    std::fill(absB.buf.begin(), absB.buf.end(), 0.0);
    for (std::ptrdiff_t l = 0; l < c.k; ++l) {
        for (std::ptrdiff_t i = 0; i < c.m; ++i) {
            absA(i, l) = std::fabs(A(i, l));
        }
        for (std::ptrdiff_t j = 0; j < c.n; ++j) {
            absB(l, j) = std::fabs(B(l, j));
        }
    }
    DeviceBuffer dAbsA(absA.buf.size()), dAbsB(absB.buf.size());
    DeviceBuffer dAbsAB(std::size_t(c.m)*c.n);
    dAbsA.upload(absA.buf);
    dAbsB.upload(absB.buf);
    cublas.gemm(c.m, c.n, c.k, 1.0, dAbsA.data + Guard, c.ldA,
                dAbsB.data + Guard, c.ldB, 0.0, dAbsAB.data, c.m);
    CULMBLAS_CHECK_CUDA(cudaDeviceSynchronize());

    HostMatrix Cref(c.m, c.n, c.ldC);
    Cref.buf = dCref.download();
    std::vector<double> absAB = dAbsAB.download();

    const double eps = std::numeric_limits<double>::epsilon();
    for (std::ptrdiff_t j = 0; j < c.n; ++j) {
        for (std::ptrdiff_t i = 0; i < c.m; ++i) {
            double cv = Cv(i, j), cr = Cref(i, j);
            double c0 = c.beta == 0 ? 0 : std::fabs(C0(i, j));
            double bound = 4*(c.k + 2)*eps
                         * (std::fabs(c.alpha)*absAB[i + j*c.m]
                            + std::fabs(c.beta)*c0);
            if (!std::isfinite(cr)) {
                return Result{ false, r.ratio, "cuBLAS result not finite" };
            }
            double err = std::fabs(cv - cr);
            if (!std::isfinite(cv) || (bound == 0 ? err != 0 : err > bound)) {
                char msg[160];
                std::snprintf(msg, sizeof(msg), "C(%td,%td) = %.17g, "
                              "cuBLAS %.17g, bound %.3g", i, j, cv, cr,
                              bound);
                r.pass = false;
                r.ratio = std::isfinite(cv) && bound > 0 ? err/bound
                        : std::numeric_limits<double>::infinity();
                r.reason = msg;
                return r;
            }
            if (bound > 0) {
                r.ratio = std::max(r.ratio, err/bound);
            }
        }
    }
    return r;
}

//
//  The test cases.
//
std::vector<Case>
makeCases(bool quick)
{
    std::vector<Case> cases;
    const double ab[][2] = { {1, 0}, {1, 1}, {0.5, -1.25}, {-2, 0},
                             {0.75, 2} };
    const std::ptrdiff_t pads[] = { 0, 1, 7 };

    //  all combinations of dimensions around the block size 32 (and 1, 2,
    //  64); alpha, beta and the padding of the leading dimensions rotate
    std::vector<std::ptrdiff_t> dims = { 1, 2, 15, 31, 32, 33, 63, 64, 65 };
    if (quick) {
        dims = { 1, 31, 32, 33 };
    }
    int count = 0;
    for (std::ptrdiff_t m : dims) {
        for (std::ptrdiff_t n : dims) {
            for (std::ptrdiff_t k : dims) {
                const double *p = ab[count % 5];
                std::ptrdiff_t pA = pads[count % 3], pB = pads[(count/3) % 3];
                std::ptrdiff_t pC = pads[(count/9) % 3];
                cases.push_back({ m, n, k, m + pA, k + pB, m + pC,
                                  p[0], p[1], Random });
                ++count;
            }
        }
    }

    //  alpha and beta: zero and nonzero, with tight and padded storage
    for (const auto &p : ab) {
        cases.push_back({ 37, 29, 41, 37, 41, 37, p[0], p[1], Random });
        cases.push_back({ 37, 29, 41, 50, 64, 100, p[0], p[1], Random });
    }
    //  beta == 0: C is not read (NaN)
    for (std::ptrdiff_t d : { 1, 33, 100 }) {
        cases.push_back({ d, d + 3, d + 5, d, d + 5, d + 1, 1.5, 0, NanC });
        cases.push_back({ d, d + 3, d + 5, d, d + 5, d + 1, -1, 0, NanC });
    }
    //  alpha == 0: A and B are not read (NaN), C = beta*C0
    for (double beta : { 0.0, 1.0, -0.5 }) {
        cases.push_back({ 45, 31, 17, 45, 17, 46, 0, beta, NanAB });
    }
    cases.push_back({ 45, 31, 17, 45, 17, 46, 0, 0, NanC });
    //  k == 0: C = beta*C0
    for (double beta : { 0.0, 1.0, 3.0 }) {
        cases.push_back({ 33, 34, 0, 33, 1, 33, 2, beta, NanAB });
    }
    //  m == 0 or n == 0: nothing is written
    cases.push_back({ 0, 17, 5, 1, 5, 1, 1, 0.5, Random });
    cases.push_back({ 17, 0, 5, 17, 5, 17, 1, 0.5, Random });
    cases.push_back({ 0, 0, 0, 1, 1, 1, 1, 0.5, Random });
    //  large padding
    cases.push_back({ 70, 50, 60, 1000, 333, 4096, 0.5, 0.5, Random });

    if (!quick) {
        //  medium and large, square and rectangular, not multiples of 32
        const std::ptrdiff_t shapes[][3] = {
            { 100, 100, 100 }, { 127, 129, 131 }, { 255, 256, 257 },
            { 500, 300, 700 }, { 1000, 1000, 1000 }, { 1023, 1025, 1024 },
            { 2048, 2048, 2048 }, { 4097, 129, 1031 }, { 129, 4097, 1031 },
            { 1, 5000, 3000 }, { 5000, 1, 3000 }, { 3000, 3000, 1 },
            { 1999, 777, 3001 }, { 3001, 3003, 515 },
        };
        int s = 0;
        for (const auto &x : shapes) {
            const double *p = ab[s % 5];
            std::ptrdiff_t pad = (s % 2) ? 3 : 0;
            cases.push_back({ x[0], x[1], x[2], x[0] + pad, x[2] + pad,
                              x[0] + pad, p[0], p[1], Random });
            ++s;
        }
    }
    return cases;
}

//  invalid arguments must throw std::invalid_argument
bool
argumentChecks(const Variant &variant)
{
    double *p = reinterpret_cast<double *>(16);    // never dereferenced
    const std::ptrdiff_t bad[][6] = {
        { -1, 1, 1, 1, 1, 1 }, { 1, -1, 1, 1, 1, 1 }, { 1, 1, -1, 1, 1, 1 },
        { 5, 1, 1, 4, 1, 5 }, { 1, 1, 5, 1, 4, 1 }, { 5, 1, 1, 5, 1, 4 },
        { 0, 1, 1, 0, 1, 1 },
    };
    for (const auto &b : bad) {
        try {
            variant.gemm(b[0], b[1], b[2], 1, p, b[3], p, b[4], 1, p, b[5],
                         0);
            return false;
        } catch (const std::invalid_argument &) {
        }
    }
    return true;
}

} // namespace

int
main(int argc, char **argv)
{
    bool quick = false, quiet = false;
    std::uint64_t seed = 2026;
    std::string names;
    for (int a = 1; a < argc; ++a) {
        std::string arg = argv[a];
        if (arg == "--quick") {
            quick = true;
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--seed" && a + 1 < argc) {
            seed = std::stoull(argv[++a]);
        } else if (arg == "--variants" && a + 1 < argc) {
            names = argv[++a];
        } else {
            std::fprintf(stderr, "usage: %s [--quick] [--quiet] [--seed N] "
                         "[--variants a,b]\n", argv[0]);
            return 2;
        }
    }

    try {
        cudaDeviceProp prop = requireDevice();
        Cublas cublas;
        int runtime = 0, driver = 0;
        CULMBLAS_CHECK_CUDA(cudaRuntimeGetVersion(&runtime));
        CULMBLAS_CHECK_CUDA(cudaDriverGetVersion(&driver));
        std::printf("# device %s (compute capability %d.%d), CUDA runtime "
                    "%d, driver %d, cuBLAS %d\n", prop.name, prop.major,
                    prop.minor, runtime, driver, cublas.version());
        std::printf("# seed %llu%s\n", (unsigned long long)seed,
                    quick ? ", quick" : "");
#if defined(CULMBLAS_WITH_MAGMA)
        std::printf("# MAGMA %s\n", Magma::instance().version().c_str());
#endif

        std::vector<Variant> list = selectVariants(names);
        if (names.empty()) {
            list.push_back({ "gemm", "cULMblas::gemm (current default)",
                             &cULMblas::gemm, false });
        }
        std::vector<Case> cases = makeCases(quick);

        int numFailed = 0, numRun = 0;
        std::vector<std::string> external;
        std::printf("# %-6s %5s %5s %5s %5s %5s %5s %6s %6s %-8s %9s\n",
                    "impl", "m", "n", "k", "ldA", "ldB", "ldC", "alpha",
                    "beta", "init", "err/bound");
        for (const Variant &v : list) {
            if (!v.external) {
                bool argsOk = argumentChecks(v);
                ++numRun;
                if (!argsOk) {
                    ++numFailed;
                }
                if (!argsOk || !quiet) {
                    std::printf("  %-6s invalid arguments: %s\n", v.name,
                                argsOk ? "PASS" : "FAIL (no exception)");
                }
            }
            int numDiff = 0;
            for (std::size_t i = 0; i < cases.size(); ++i) {
                const Case &c = cases[i];
                Result r = runCase(v, cublas, c, seed + 1000003*i);
                if (v.external) {
                    numDiff += !r.pass;
                } else {
                    ++numRun;
                    numFailed += !r.pass;
                }
                if (!r.pass || !quiet) {
                    std::printf("  %-6s %5td %5td %5td %5td %5td %5td %6.2f "
                                "%6.2f %-8s %9.2e %s%s%s\n", v.name, c.m,
                                c.n, c.k, c.ldA, c.ldB, c.ldC, c.alpha,
                                c.beta, initName(c.init), r.ratio,
                                r.pass ? "PASS"
                                       : v.external ? "DIFF" : "FAIL",
                                r.reason.empty() ? "" : ": ",
                                r.reason.c_str());
                }
            }
            if (v.external) {
                external.push_back(std::string(v.name) + " (" + v.description
                                   + "): " + std::to_string(numDiff) + " of "
                                   + std::to_string(cases.size())
                                   + " cases differ from BLAS");
            }
        }
        for (const std::string &e : external) {
            std::printf("# external reference %s\n", e.c_str());
        }
        std::printf("# %d checks of cULMblas, %d failed (%zu cases each)\n",
                    numRun, numFailed, cases.size());
        return numFailed == 0 ? 0 : 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "test_gemm: error: %s\n", e.what());
        return 2;
    }
}
