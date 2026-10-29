//
//  ukernel: throughput of the selected ulmBLAS micro kernel for double.
//
//      ukernel KC [KC ...]
//
//  Calls ulmBLAS::ugemm (MR x NR tile, unit row stride of C, beta = 1) for
//  NP x NQ pairs of packed panels of A (MR x KC) and B (KC x NR), the order
//  of the macro kernel (B panel outer, A panels inner).  NP = 8 panels of A
//  fit into the L2 cache; the B panel is reused from L1.  Output per KC:
//
//      kc MR NR median_GFLOPS min max
//
//  On macOS the thread runs with QoS class user-interactive.
//
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ulmblas/level3/level3.h>

static double
now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int
main(int argc, char **argv)
{
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    const long MR = ulmBLAS::BlockSizeUGemm<double>::MR;
    const long NR = ulmBLAS::BlockSizeUGemm<double>::NR;
    const long NP = 8, NQ = 4;
    double     sum = 0;

    for (int a=1; a<argc; ++a) {
        long kc = std::atol(argv[a]);
        std::vector<double> A(NP*MR*kc+MR), B(NQ*NR*kc+NR), C(MR*NR*NP*NQ);
        for (auto &x : A) x = double(std::rand())/RAND_MAX - 0.5;
        for (auto &x : B) x = double(std::rand())/RAND_MAX - 0.5;

        auto sweep = [&]() {
            for (long q=0; q<NQ; ++q) {
                for (long p=0; p<NP; ++p) {
                    ulmBLAS::ugemm(kc, 1.0, &A[p*MR*kc], &B[q*NR*kc], 1.0,
                                   &C[(q*NP+p)*MR*NR], 1L, MR,
                                   &A[((p+1)%NP)*MR*kc], &B[q*NR*kc]);
                }
            }
        };
        double t0 = now(), t1;
        long   reps = 0;
        do { sweep(); ++reps; t1 = now(); } while (t1-t0 < 0.3);
        reps = long(reps*0.2/(t1-t0)) + 1;
        std::vector<double> g;
        for (int s=0; s<7; ++s) {
            t0 = now();
            for (long r=0; r<reps; ++r) sweep();
            g.push_back(2.0*MR*NR*kc*NP*NQ*reps/(now()-t0)*1e-9);
        }
        std::sort(g.begin(), g.end());
        std::printf("%5ld %ld %ld %8.3f %8.3f %8.3f\n", kc, MR, NR, g[3],
                    g[0], g[6]);
        sum += C[0];
    }
    return sum==12345.0;      // C is used
}
