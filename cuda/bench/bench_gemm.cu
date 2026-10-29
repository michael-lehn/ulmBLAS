//
//  bench_gemm: GPU time of the GEMM implementations of cULMblas and of
//  cuBLAS (cublasDgemm) on the same device, same dimensions and same
//  operands.
//
//  For every shape m x n x k the operands are created on the host
//  (pseudo random, fixed seed), copied to the device once, and then for
//  every implementation:
//
//  - C is reset to its initial values (not timed),
//  - WARMUP calls (not timed),
//  - REPS calls, each between two CUDA events on the same stream; the time
//    of a call is the time between the events, i.e. the GPU time of the
//    work enqueued between them.  It does not include allocations,
//    host/device transfers or the time the host needs to issue the call.
//
//  Reported: median, minimum and maximum time per call, GFLOP/s of the
//  median (2 m n k flops) and the ratio to cuBLAS (time of cuBLAS / time of
//  the implementation: 1 means as fast as cuBLAS).  There is no correctness
//  check here; that is test_gemm's job.  The output is a table that gnuplot
//  reads directly (comments start with #), see step01/plot.gp.
//
//  Options:
//
//      --shapes "m,n,k m,n,k ..."   shapes (default: m = n = k = 50, 100,
//                                   150, ..., 4000)
//      --reps R                     timed calls per implementation (10)
//      --warmup W                   untimed calls before (2)
//      --variants a,b               cULMblas implementations (all)
//      --pad P                      leading dimensions m+P, k+P, m+P (0)
//      --alpha A --beta B           (1, 1)
//
#include <support/support.h>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace cULMblas::support;

namespace {

struct Shape
{
    std::ptrdiff_t m, n, k;
};

std::vector<Shape>
parseShapes(const std::string &s)
{
    std::vector<Shape> shapes;
    std::istringstream in(s);
    std::string item;
    while (in >> item) {
        Shape x;
        if (std::sscanf(item.c_str(), "%td,%td,%td", &x.m, &x.n, &x.k) != 3
         || x.m < 1 || x.n < 1 || x.k < 1)
        {
            throw std::invalid_argument("bad shape '" + item + "' (m,n,k)");
        }
        shapes.push_back(x);
    }
    return shapes;
}

class Event
{
    public:
        Event()
        {
            CULMBLAS_CHECK_CUDA(cudaEventCreate(&event));
        }

        ~Event()
        {
            cudaEventDestroy(event);
        }

        cudaEvent_t event;
};

struct Timing
{
    double median, min, max;    // milliseconds
};

//  call(): enqueues one GEMM on the stream
template <typename Call>
Timing
measure(Call call, cudaStream_t stream, int warmup, int reps)
{
    for (int r = 0; r < warmup; ++r) {
        call();
    }
    CULMBLAS_CHECK_CUDA(cudaStreamSynchronize(stream));

    Event start, stop;
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        CULMBLAS_CHECK_CUDA(cudaEventRecord(start.event, stream));
        call();
        CULMBLAS_CHECK_CUDA(cudaEventRecord(stop.event, stream));
        CULMBLAS_CHECK_CUDA(cudaEventSynchronize(stop.event));
        float ms = 0;
        CULMBLAS_CHECK_CUDA(cudaEventElapsedTime(&ms, start.event,
                                                 stop.event));
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    std::size_t h = t.size()/2;
    double median = t.size() % 2 ? t[h] : (t[h-1] + t[h])/2;
    return Timing{ median, t.front(), t.back() };
}

} // namespace

int
main(int argc, char **argv)
{
    //  default: m = n = k = 50, 100, 150, ..., 4000
    std::string shapeList;
    for (int s = 50; s <= 4000; s += 50) {
        shapeList += std::to_string(s) + "," + std::to_string(s) + ","
                   + std::to_string(s) + " ";
    }
    int reps = 10, warmup = 2;
    std::ptrdiff_t pad = 0;
    double alpha = 1, beta = 1;
    std::string names;
    for (int a = 1; a < argc; ++a) {
        std::string arg = argv[a];
        if (a + 1 >= argc) {
            arg = "--help";
        }
        if (arg == "--shapes") {
            shapeList = argv[++a];
        } else if (arg == "--reps") {
            reps = std::stoi(argv[++a]);
        } else if (arg == "--warmup") {
            warmup = std::stoi(argv[++a]);
        } else if (arg == "--variants") {
            names = argv[++a];
        } else if (arg == "--pad") {
            pad = std::stol(argv[++a]);
        } else if (arg == "--alpha") {
            alpha = std::stod(argv[++a]);
        } else if (arg == "--beta") {
            beta = std::stod(argv[++a]);
        } else {
            std::fprintf(stderr, "usage: %s [--shapes \"m,n,k ...\"] "
                         "[--reps R] [--warmup W] [--variants a,b] [--pad P] "
                         "[--alpha A] [--beta B]\n", argv[0]);
            return 2;
        }
    }

    try {
        if (reps < 1 || warmup < 0 || pad < 0) {
            throw std::invalid_argument("reps >= 1, warmup >= 0, pad >= 0");
        }
        std::vector<Shape> shapes = parseShapes(shapeList);
        std::vector<Variant> list = selectVariants(names);

        cudaDeviceProp prop = requireDevice();
        Cublas cublas;
        int runtime = 0, driver = 0, clockKHz = 0, memClockKHz = 0;
        CULMBLAS_CHECK_CUDA(cudaRuntimeGetVersion(&runtime));
        CULMBLAS_CHECK_CUDA(cudaDriverGetVersion(&driver));
        CULMBLAS_CHECK_CUDA(cudaDeviceGetAttribute(&clockKHz,
                                                   cudaDevAttrClockRate, 0));
        CULMBLAS_CHECK_CUDA(cudaDeviceGetAttribute(&memClockKHz,
                                                   cudaDevAttrMemoryClockRate,
                                                   0));
        char host[256] = "";
        gethostname(host, sizeof(host) - 1);
        std::time_t now = std::time(0);
        char date[64];
        std::strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S",
                      std::localtime(&now));

        std::printf("# host %s, %s\n", host, date);
        std::printf("# device %s, compute capability %d.%d, %d SMs, "
                    "%.0f MHz (max), memory %.1f GiB at %.0f MHz\n",
                    prop.name, prop.major, prop.minor,
                    prop.multiProcessorCount, clockKHz/1000.0,
                    prop.totalGlobalMem/1073741824.0, memClockKHz/1000.0);
        std::printf("# CUDA runtime %d, driver %d, cuBLAS %d, nvcc %d.%d\n",
                    runtime, driver, cublas.version(), __CUDACC_VER_MAJOR__,
                    __CUDACC_VER_MINOR__);
#if defined(CULMBLAS_BUILD_FLAGS)
        std::printf("# build: %s\n", CULMBLAS_BUILD_FLAGS);
#endif
        std::printf("# DGEMM, column major, alpha %g, beta %g, ld = m+%td, "
                    "k+%td, m+%td; %d warm-up calls, %d timed calls, "
                    "CUDA events\n", alpha, beta, pad, pad, pad, warmup,
                    reps);
#if defined(CULMBLAS_WITH_MAGMA)
        std::printf("# MAGMA %s (magma: magma_dgemm, which calls cublasDgemm; "
                    "magmablas: magmablas_dgemm, MAGMA's own kernel)\n",
                    Magma::instance().version().c_str());
#endif
        std::printf("# %-6s %6s %6s %6s %10s %10s %10s %10s %8s\n", "impl",
                    "m", "n", "k", "median_ms", "min_ms", "max_ms",
                    "GFLOP/s", "/cuBLAS");

        std::mt19937_64 gen(2026);
        std::uniform_real_distribution<double> uniform(-1, 1);

        for (const Shape &s : shapes) {
            std::ptrdiff_t ldA = s.m + pad, ldB = s.k + pad, ldC = s.m + pad;
            std::vector<double> A(std::size_t(ldA)*s.k),
                                B(std::size_t(ldB)*s.n),
                                C0(std::size_t(ldC)*s.n);
            for (double &x : A) x = uniform(gen);
            for (double &x : B) x = uniform(gen);
            for (double &x : C0) x = uniform(gen);

            DeviceBuffer dA(A.size()), dB(B.size()), dC(C0.size());
            dA.upload(A);
            dB.upload(B);

            const cudaStream_t stream = 0;
            double flops = 2.0*s.m*s.n*s.k;

            dC.upload(C0);
            Timing ref = measure([&] {
                cublas.gemm(s.m, s.n, s.k, alpha, dA.data, ldA, dB.data, ldB,
                            beta, dC.data, ldC, stream);
            }, stream, warmup, reps);
            std::printf("%-8s %6td %6td %6td %10.3f %10.3f %10.3f %10.1f "
                        "%8.3f\n", "cuBLAS", s.m, s.n, s.k, ref.median,
                        ref.min, ref.max, flops/ref.median/1e6, 1.0);

            for (const Variant &v : list) {
                dC.upload(C0);
                Timing t = measure([&] {
                    v.gemm(s.m, s.n, s.k, alpha, dA.data, ldA, dB.data, ldB,
                           beta, dC.data, ldC, stream);
                }, stream, warmup, reps);
                std::printf("%-8s %6td %6td %6td %10.3f %10.3f %10.3f %10.1f "
                            "%8.3f\n", v.name, s.m, s.n, s.k, t.median,
                            t.min, t.max, flops/t.median/1e6,
                            ref.median/t.median);
            }
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "bench_gemm: error: %s\n", e.what());
        return 2;
    }
}
