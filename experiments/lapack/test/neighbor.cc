// Lost-update test: while a kernel works on its output view, a second
// thread writes and reads back the cells around the view (8 cells deep).
// A kernel that loads a neighbor cell and stores it back (unchanged)
// overwrites the other thread's write, which the reader detects.
#include <atomic>
#include <cstdio>
#include <random>
#include <cstdlib>
#include <thread>
#include <vector>
#include "kernels.h"
using namespace ulmlapack;

long detect(const char *name, int ld, int rows, int cols, int r0, int c0, std::vector<double> &a,
            void (*kernel)(double *, void *), void *arg, int reps)
{
    std::vector<long> cells;                              // storage indices of the margin
    for (int j=c0-2; j<c0+cols+2; ++j) for (int i=r0-8; i<r0+rows+8; ++i) {
        bool inView = (i>=r0 && i<r0+rows && j>=c0 && j<c0+cols);
        if (!inView && i>=0 && j>=0 && i<ld) cells.push_back(i+long(j)*ld);
    }
    std::atomic<bool> stop(false); std::atomic<long> lost(0);
    volatile double *A = a.data();
    std::thread t([&] {
        double v = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            for (long x: cells) A[x] = v;
            for (long x: cells) if (A[x]!=v) { ++lost; A[x] = v; }
            v += 1;
        }
    });
    for (int r=0; r<reps; ++r) kernel(&a[r0+long(c0)*ld], arg);
    stop = true; t.join();
    if (lost) printf("%s: %ld lost updates in the cells around the view\n", name, lost.load());
    return lost;
}

// kernels on an 8x8 view inside a column-major array with leading dimension LD
const int LD = 64, NB = 8;
std::vector<double> L8(NB*NB), B8(NB*NB), X8(NB*NB);
void k_trlsm_unit(double *C, void *) { ulmBLAS::trlsm(Index(NB), Index(NB), 1.0, false, true,  L8.data(), Index(1), Index(NB), C, Index(1), Index(LD)); }
void k_trlsm_T   (double *C, void *) { ulmBLAS::trlsm(Index(NB), Index(NB), 1.0, false, false, L8.data(), Index(1), Index(NB), C, Index(LD), Index(1)); }
void k_gemm_NN   (double *C, void *) { ulmBLAS::gemm(Index(NB), Index(NB), Index(NB), -1e-9, false, B8.data(), Index(1), Index(NB), false, X8.data(), Index(1), Index(NB), 1.0, C, Index(1), Index(LD)); }
void k_gemm_NT   (double *C, void *) { ulmBLAS::gemm(Index(NB), Index(NB), Index(NB), -1e-9, false, B8.data(), Index(1), Index(NB), false, X8.data(), Index(NB), Index(1), 1.0, C, Index(1), Index(LD)); }
void k_sylrk     (double *C, void *) { ulmBLAS::sylrk(Index(NB), Index(NB), -1e-9, B8.data(), Index(1), Index(NB), 1.0, C, Index(1), Index(LD)); }
void k_potrf     (double *C, void *) { for (int i=0;i<NB;++i) for (int j=0;j<NB;++j) C[i+j*LD] = (i==j) ? 10 : 0.01; potrf_lower_rec(Index(NB), C, Index(1), Index(LD)); }
void k_getrf     (double *C, void *) { std::vector<Index> p(NB); for (int i=0;i<NB;++i) for (int j=0;j<NB;++j) C[i+j*LD] = (i==j) ? 10 : 0.01*(i-j); getrf_rec(Index(NB), Index(NB), C, Index(1), Index(LD), p.data()); }
void k_laswp     (double *C, void *) { Index p[NB] = {3,1,7,3,5,5,6,7}; laswp(Index(NB), C, Index(1), Index(LD), 0, NB, p); }

int main(int argc, char **argv) {
    int reps = argc>1 ? atoi(argv[1]) : 200000;
    std::mt19937_64 g(1); std::uniform_real_distribution<double> u(-1,1);
    for (int i=0;i<NB*NB;++i) { L8[i]=u(g); B8[i]=u(g); X8[i]=u(g); }
    for (int i=0;i<NB;++i) L8[i+i*NB] = 4;
    struct K { const char *n; void (*f)(double*, void*); } ks[] = {
        {"trlsm unit (getrf)", k_trlsm_unit}, {"trlsm transposed B (potrf)", k_trlsm_T},
        {"gemm NN (getrf)", k_gemm_NN}, {"gemm NT (potrf)", k_gemm_NT}, {"sylrk (potrf)", k_sylrk},
        {"potrf_lower_rec", k_potrf}, {"getrf_rec", k_getrf}, {"laswp", k_laswp}};
    long total = 0;
    for (auto &k: ks) for (int off: {0, 1, 3, 8}) {      // view offsets: aligned and not
        std::vector<double> a(LD*40, 0.0);
        for (int i=0;i<LD*40;++i) a[i] = (i%7)*0.1+1;
        long l = detect(k.n, LD, NB, NB, 16+off, 10, a, k.f, 0, reps);
        total += l;
    }
    printf("neighbor test: %ld lost updates\n", total);
    return total!=0;
}
