// Bitwise determinism stress: for fixed (routine, n, nb, threads) the
// result must be identical in every repetition (all updates of a tile are
// ordered by the dependencies).  OpenBLAS calls are interleaved as in the
// test program.  Runs for SECONDS seconds.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <tuple>
#include <vector>
#include <ulmlapack.h>
extern "C" void dpotrf_(const char*, const int*, double*, const int*, int*, size_t);
extern "C" void dgetrf_(const int*, const int*, double*, const int*, int*, int*);
int main(int argc, char **argv) {
    double seconds = atof(argv[1]);
    std::mt19937_64 rng(7); std::uniform_real_distribution<double> u(-1,1);
    std::map<std::tuple<int,int,int,int>, std::vector<double>> ref;
    long runs = 0, mism = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count() < seconds) {
        int r = rng()%3, n = (rng()%2) ? 200 : 400, nbs[] = {8, 16, 33}, nb = nbs[rng()%3], th = 1 + rng()%8;
        std::mt19937_64 g(n); std::vector<double> A(size_t(n)*n);
        for (double &x: A) x = u(g);
        if (r<2) { for (int j=0;j<n;++j) for (int i=j+1;i<n;++i) A[j+i*n]=A[i+j*n]; for (int i=0;i<n;++i) A[i+i*n]+=n; }
        ulmlapack_param p; ulmlapack_param_init(&p); p.nb=nb; p.threads=th; p.serial_n=0; p.panel_threads = 1;   // teams form dynamically and change the rounding
        std::vector<int> ip(n);
        std::vector<double> B(A);
        if (r<2) ulmlapack_dpotrf_param(r ? 'U' : 'L', n, B.data(), n, &p);
        else     ulmlapack_dgetrf_param(n, n, B.data(), n, ip.data(), &p);
        // LU uses serial panels (panel_threads = 1): a panel team forms
        // dynamically, and the team and the serial panel round differently,
        // so with teams the LU result is not bitwise reproducible.
        auto key = std::make_tuple(r, n, nb, th);
        if (r==2) for (int i=0;i<n;++i) B.push_back(ip[i]);
        auto it = ref.find(key);
        if (it==ref.end()) ref[key] = B;
        else if (std::memcmp(it->second.data(), B.data(), B.size()*8)) {
            if (++mism<40) printf("MISMATCH r=%d n=%d nb=%d th=%d pt=%d (run %ld)\n", r, n, nb, th, p.panel_threads, runs);
        }
        ++runs;
        { std::vector<double> Q(A); int info; if (r<2) dpotrf_("L", &n, Q.data(), &n, &info, 1); else dgetrf_(&n, &n, Q.data(), &n, ip.data(), &info); }
    }
    printf("determinism: %ld runs, %ld mismatches\n", runs, mism);
    return mism!=0;
}
