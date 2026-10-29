//
//  clockrun IMPL N REPS: REPS calls of IMPL (cuBLAS or a cULMblas variant)
//  for m = n = k = N, after 5 warm-up calls; prints the wall-clock time of
//  the first and the last timed call (format of nvidia-smi --query-gpu
//  timestamp) and the median time, so that a log of nvidia-smi can be cut
//  into the phases of the implementations.
//
#include <support/support.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <random>
#include <string>
#include <vector>

using namespace cULMblas::support;

static std::string
stamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    int ms = int(std::chrono::duration_cast<std::chrono::milliseconds>(
                 now.time_since_epoch()).count() % 1000);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y/%m/%d %H:%M:%S", std::localtime(&t));
    char out[80];
    std::snprintf(out, sizeof(out), "%s.%03d", buf, ms);
    return out;
}

int
main(int argc, char **argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "usage: clockrun IMPL N REPS\n");
        return 1;
    }
    std::string impl = argv[1];
    std::ptrdiff_t n = std::stol(argv[2]);
    int reps = std::stoi(argv[3]);

    Cublas cublas;
    GemmFunction f = 0;
    for (const Variant &v : variants()) {
        if (impl == v.name) f = v.gemm;
    }
    if (impl != "cuBLAS" && !f) {
        std::fprintf(stderr, "clockrun: unknown implementation %s\n",
                     impl.c_str());
        return 1;
    }
    std::mt19937_64 gen(2026);
    std::uniform_real_distribution<double> uniform(-1, 1);
    std::vector<double> A(n*n), B(n*n), C(n*n);
    for (double &x : A) x = uniform(gen);
    for (double &x : B) x = uniform(gen);
    for (double &x : C) x = uniform(gen);
    DeviceBuffer dA(A.size()), dB(B.size()), dC(C.size());
    dA.upload(A);
    dB.upload(B);
    dC.upload(C);

    auto call = [&] {
        if (f) {
            f(n, n, n, 1.0, dA.data, n, dB.data, n, 1.0, dC.data, n, 0);
        } else {
            cublas.gemm(n, n, n, 1.0, dA.data, n, dB.data, n, 1.0, dC.data, n);
        }
    };
    for (int r = 0; r < 5; ++r) call();
    CULMBLAS_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<cudaEvent_t> ev(reps + 1);
    for (cudaEvent_t &e : ev) CULMBLAS_CHECK_CUDA(cudaEventCreate(&e));
    std::string first = stamp();
    CULMBLAS_CHECK_CUDA(cudaEventRecord(ev[0]));
    for (int r = 0; r < reps; ++r) {
        call();
        CULMBLAS_CHECK_CUDA(cudaEventRecord(ev[r + 1]));
    }
    CULMBLAS_CHECK_CUDA(cudaDeviceSynchronize());
    std::string last = stamp();
    std::vector<float> t(reps);
    for (int r = 0; r < reps; ++r) {
        CULMBLAS_CHECK_CUDA(cudaEventElapsedTime(&t[r], ev[r], ev[r + 1]));
    }
    std::vector<float> s = t;
    std::sort(s.begin(), s.end());
    double med = s[reps/2];
    //  median of the first and of the last tenth of the calls
    auto part = [&](int a, int b) {
        std::vector<float> p(t.begin() + a, t.begin() + b);
        std::sort(p.begin(), p.end());
        return p[p.size()/2];
    };
    std::printf("%-8s n %5td reps %6d first %s last %s median_ms %.4f "
                "GFLOP/s %.1f first10%%_ms %.4f last10%%_ms %.4f\n",
                impl.c_str(), n, reps, first.c_str(), last.c_str(), med,
                2.0*n*n*n/med/1e6, part(0, std::max(1, reps/10)),
                part(reps - std::max(1, reps/10), reps));
    return 0;
}
