//
//  Independent DGEMM calls from several threads at the same time: each
//  thread multiplies its own matrices (irregular dimensions, leading
//  dimensions larger than needed, both orders, all transpositions, several
//  alpha and beta), compares the result with a reference computed in long
//  double and checks that the padding of C is unchanged.  Half of the
//  threads use the same sequence of dimensions (identical sizes of the
//  packing buffers at the same time), the others different ones.  The
//  threads start together.
//
//  ULMBLAS_TEST_THREAD_ITERATIONS (environment) sets the number of calls
//  per thread (default 150) for longer stress runs.
//
#include <test/numeric/blas.h>
#include <test/numeric/testing.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace test;

struct ThreadResult
{
    long         calls    = 0;
    long         failures = 0;
    std::string  message;
};

void
worker(int id, unsigned seed, long iterations, std::atomic<bool> &start,
       ThreadResult &result)
{
    const int dims[] = { 1, 2, 3, 7, 8, 9, 12, 13, 15, 16, 17, 23, 24, 31,
                         32, 33, 47, 63, 64, 65, 100, 129 };
    const int numDims = sizeof(dims)/sizeof(dims[0]);
    const CBLAS_TRANSPOSE trans[] = { CblasNoTrans, CblasTrans };
    const double scalars[] = { 1.0, 0.0, -1.0, 0.5, 2.25 };

    std::mt19937 shapes(seed);
    std::mt19937 values(1000 + id);

    while (!start.load()) {
    }
    for (long it=0; it<iterations; ++it) {
        int m = dims[shapes() % numDims];
        int n = dims[shapes() % numDims];
        int k = dims[shapes() % numDims];
        bool            rowMajor = shapes() % 2;
        CBLAS_TRANSPOSE tA = trans[shapes() % 2], tB = trans[shapes() % 2];
        double alpha = scalars[shapes() % 5], beta = scalars[shapes() % 5];
        int padA = shapes() % 6, padB = shapes() % 6, padC = shapes() % 6;

        // op(A) is m x k, op(B) k x n; stored rows x cols with leading
        // dimension ld (rows for column major, cols for row major)
        int rA = (tA==CblasNoTrans) ? m : k, cA = (tA==CblasNoTrans) ? k : m;
        int rB = (tB==CblasNoTrans) ? k : n, cB = (tB==CblasNoTrans) ? n : k;
        int ldA = (rowMajor ? cA : rA) + padA;
        int ldB = (rowMajor ? cB : rB) + padB;
        int ldC = (rowMajor ? n : m) + padC;

        std::uniform_real_distribution<double> u(-1.0, 1.0);
        std::vector<double> A(std::size_t(ldA)*(rowMajor ? rA : cA) + 1);
        std::vector<double> B(std::size_t(ldB)*(rowMajor ? rB : cB) + 1);
        std::vector<double> C(std::size_t(ldC)*(rowMajor ? m : n) + 1);
        for (double &x : A) x = u(values);
        for (double &x : B) x = u(values);
        for (double &x : C) x = u(values);
        std::vector<double> C0 = C;

        auto at = [rowMajor](const std::vector<double> &X, int ld, int i,
                             int j) {
            return rowMajor ? X[std::size_t(i)*ld + j]
                            : X[i + std::size_t(j)*ld];
        };
        auto opA = [&](int i, int l) {
            return (tA==CblasNoTrans) ? at(A, ldA, i, l) : at(A, ldA, l, i);
        };
        auto opB = [&](int l, int j) {
            return (tB==CblasNoTrans) ? at(B, ldB, l, j) : at(B, ldB, j, l);
        };

        blas::gemm(rowMajor ? CblasRowMajor : CblasColMajor, tA, tB, m, n, k,
                   alpha, A.data(), ldA, B.data(), ldB, beta, C.data(), ldC);
        ++result.calls;

        const double eps = std::numeric_limits<double>::epsilon();
        bool ok = true;
        for (int i=0; i<m && ok; ++i) {
            for (int j=0; j<n && ok; ++j) {
                long double s = 0, bound = 0;
                for (int l=0; l<k; ++l) {
                    s     += (long double)opA(i, l)*opB(l, j);
                    bound += std::fabs(opA(i, l)*opB(l, j));
                }
                long double c0  = at(C0, ldC, i, j);
                long double ref = alpha*s + (beta!=0 ? beta*c0 : 0);
                bound = std::fabs(alpha)*bound
                      + (beta!=0 ? std::fabs(beta*c0) : 0);
                double c = at(C, ldC, i, j);
                if (!(std::fabs(c - ref) <= 4*(k+2)*eps*bound + 1e-300)) {
                    ok = false;
                    result.message = "thread " + std::to_string(id)
                        + ": dgemm(" + (rowMajor ? "RowMajor" : "ColMajor")
                        + "," + (tA==CblasNoTrans ? "N" : "T") + ","
                        + (tB==CblasNoTrans ? "N" : "T") + ",m="
                        + std::to_string(m) + ",n=" + std::to_string(n)
                        + ",k=" + std::to_string(k) + ",ldC="
                        + std::to_string(ldC) + "): wrong C("
                        + std::to_string(i) + "," + std::to_string(j) + ")";
                }
            }
        }
        // the padding of C and the element after C are unchanged
        for (std::size_t p=0; p<C.size() && ok; ++p) {
            std::size_t ld = ldC, minor = p % ld, major = p / ld;
            bool inside = (p+1<C.size())
                       && minor<std::size_t(rowMajor ? n : m)
                       && major<std::size_t(rowMajor ? m : n);
            if (!inside && C[p]!=C0[p]) {
                ok = false;
                result.message = "thread " + std::to_string(id)
                               + ": padding of C modified";
            }
        }
        if (!ok) {
            ++result.failures;
            return;
        }
    }
}

} // namespace

ULMBLAS_TEST(T0351, "level3/dgemm from 8 threads at the same time, own "
             "matrices, irregular dimensions and leading dimensions")
{
    const int numThreads = 8;
    long      iterations = 150;
    if (const char *s = std::getenv("ULMBLAS_TEST_THREAD_ITERATIONS")) {
        iterations = std::atol(s);
    }

    std::atomic<bool>          start(false);
    std::vector<ThreadResult>  results(numThreads);
    std::vector<std::thread>   threads;
    for (int t=0; t<numThreads; ++t) {
        // threads 0 to 3 share the sequence of dimensions
        unsigned seed = (t<numThreads/2) ? 4711u : 4711u + t;
        threads.emplace_back(worker, t, seed, iterations, std::ref(start),
                             std::ref(results[t]));
    }
    start = true;
    for (std::thread &t : threads) {
        t.join();
    }
    long calls = 0;
    for (const ThreadResult &r : results) {
        calls += r.calls;
        ctx.check(r.failures==0, "%s", r.message.c_str());
    }
    ctx.check(calls==numThreads*iterations, "%ld of %ld calls done", calls,
              numThreads*iterations);
}
