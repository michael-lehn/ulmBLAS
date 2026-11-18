//  GEMM through ulmBLAS::gemm for a type and a list of shapes: GFLOP/s,
//  median of 5 samples of at least 20 ms, each after one untimed call.
//
//      selbench d|s|c|z
//
//  Output: type m n k gflops (one line per shape).
#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <ulmblas/ulmblas.h>

struct Shape
{
    std::ptrdiff_t m, n, k;
};

static std::vector<Shape>
shapes(bool full)
{
    std::vector<Shape> s;
    for (std::ptrdiff_t n : {32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
                             1024, 1100, 1536, 2048, 2300, 3072, 4000}) {
        s.push_back({n, n, n});
    }
    if (full) {
        for (std::ptrdiff_t k : {16, 32, 64, 128, 256, 512}) {
            s.push_back({2048, 2048, k});
        }
        for (std::ptrdiff_t m : {32, 64, 128, 256, 512}) {
            s.push_back({m, 2048, 2048});
        }
        for (std::ptrdiff_t n : {32, 64, 128, 256, 512}) {
            s.push_back({2048, n, 2048});
        }
        for (std::ptrdiff_t n : {300, 700, 1000, 1500}) {
            s.push_back({n, n, 1100});
        }
    }
    return s;
}

static double
now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

template <typename T>
static void
run(const char *type, double opsPerFma, bool full)
{
    std::mt19937 gen(1);
    std::uniform_real_distribution<double> dist(-1, 1);
    for (const Shape &s : shapes(full)) {
        std::vector<T> A(s.m * s.k), B(s.k * s.n), C(s.m * s.n);
        for (T &x : A) x = T(dist(gen));
        for (T &x : B) x = T(dist(gen));
        for (T &x : C) x = T(dist(gen));
        auto call = [&] {
            ulmBLAS::gemm(s.m, s.n, s.k, T(1), false, A.data(),
                          std::ptrdiff_t(1), s.m, false, B.data(),
                          std::ptrdiff_t(1), s.k, T(1), C.data(),
                          std::ptrdiff_t(1), s.m);
        };
        call();
        double t0 = now();
        call();
        double once = now() - t0;
        std::ptrdiff_t reps =
            std::max<std::ptrdiff_t>(1, std::ptrdiff_t(20e-3 / once));
        std::vector<double> g;
        for (int r = 0; r < 5; ++r) {
            call();
            double t = now();
            for (std::ptrdiff_t p = 0; p < reps; ++p) call();
            t = (now() - t) / double(reps);
            g.push_back(opsPerFma * s.m * s.n * s.k / t * 1e-9);
        }
        std::sort(g.begin(), g.end());
        std::printf("%s\t%td\t%td\t%td\t%.1f\n", type, s.m, s.n, s.k, g[2]);
        std::fflush(stdout);
    }
}

int
main(int argc, char **argv)
{
    const char *t = argc > 1 ? argv[1] : "d";
    if (!std::strcmp(t, "d")) run<double>("d", 2, true);
    if (!std::strcmp(t, "s")) run<float>("s", 2, false);
    if (!std::strcmp(t, "c")) run<std::complex<float>>("c", 8, false);
    if (!std::strcmp(t, "z")) run<std::complex<double>>("z", 8, false);
}
