//
//  The gemv kernels of ulmBLAS directly (header only), for the parameters
//  of ulmblas/config/level2.h set with -D:
//
//      l2kern OP TYPE N...      OP: n (A column major), t (row major),
//                               c (row major, conj), ger, pool (empty
//                               parallel region, microseconds), pool0
//                               (the same without pinning the caller)
//                               TYPE: s d c z
//
//  GFLOP/s, best of 5 samples of at least 20 ms, one thread (gemvSerial)
//  unless THREADS > 1 (gemv, ger with the pool).  ROWS=r: gemv with the
//  first r rows of y only (A n x n).
//
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ulmblas/ulmblas.h>
#include <vector>

template <typename T>
T
rnd()
{
    return T(double(std::rand()) / RAND_MAX - 0.5);
}

template <typename T> struct Rnd
{
    static T
    get()
    {
	return rnd<T>();
    }
};

template <typename T> struct Rnd<std::complex<T>>
{
    static std::complex<T>
    get()
    {
	return std::complex<T>(rnd<T>(), rnd<T>());
    }
};

static double
now()
{
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

template <typename T>
void
run(const char *op, std::ptrdiff_t n, bool threads)
{
    const bool cplx =
        sizeof(T) == 8 && !std::is_same<T, double>::value || sizeof(T) == 16;
    std::vector<T> A(n * n), x(n), y(n);
    for (auto &a : A) {
	a = Rnd<T>::get();
    }
    for (auto &a : x) {
	a = Rnd<T>::get();
    }
    for (auto &a : y) {
	a = Rnd<T>::get();
    }
    const std::ptrdiff_t one = 1;
    //  ROWS: only the first ROWS rows of y (the part of one thread)
    const char *rows = std::getenv("ROWS");
    const std::ptrdiff_t m = rows ? std::atol(rows) : n;
    double best = 1e30;
    long reps = 1;
    for (int s = 0; s < 6; ++s) {
	double t0 = now();
	for (long r = 0; r < reps; ++r) {
	    if (!std::strcmp(op, "n") || !std::strcmp(op, "t") ||
	        !std::strcmp(op, "c")) {
		const bool colMajor = op[0] == 'n';
		const bool conj = op[0] == 'c';
		if (threads) {
		    ulmBLAS::gemv(m, n, T(1), conj, A.data(),
		                  colMajor ? one : n, colMajor ? n : one,
		                  x.data(), one, T(1), y.data(), one);
		} else {
		    ulmBLAS::gemvSerial(m, n, T(1), conj, A.data(),
		                        colMajor ? one : n, colMajor ? n : one,
		                        x.data(), one, T(1), y.data(), one);
		}
	    } else if (!std::strcmp(op, "ger")) {
		if (threads) {
		    ulmBLAS::ger(n, n, T(1e-9), x.data(), one, y.data(), one,
		                 A.data(), one, n);
		} else {
		    ulmBLAS::gerSerial(n, n, T(1e-9), x.data(), one, y.data(),
		                       one, A.data(), one, n);
		}
	    }
	}
	double t = (now() - t0) / reps;
	if (s == 0) {
	    reps = long(0.02 / t) + 1;
	    continue;
	}
	if (t < best) {
	    best = t;
	}
    }
    std::printf("%s %ld %.2f\n", op, long(n),
                (cplx ? 8.0 : 2.0) * m * n / best * 1e-9);
}

int
main(int argc, char **argv)
{
    if (argc < 4) {
	std::fprintf(stderr, "usage: l2kern OP TYPE N...\n");
	return 1;
    }
    const char *t = std::getenv("THREADS");
    const bool threads = t && std::atoi(t) > 1;
    if (threads) {
	ulmBLAS::ThreadPool::instance().setSize(std::atoi(t));
    }
    if (!std::strncmp(argv[1], "pool", 4)) {
	const bool pinCaller = argv[1][4] != '0';
	ulmBLAS::ThreadPool &pool = ulmBLAS::ThreadPool::instance();
	auto body = [](std::ptrdiff_t, std::ptrdiff_t) {};
	for (int s = 0; s < 5; ++s) {
	    const long reps = 100000;
	    double t0 = now();
	    for (long r = 0; r < reps; ++r) {
		pool.run(pool.size(), body, pinCaller);
	    }
	    std::printf("pool %ld threads: %.3f us\n", long(pool.size()),
	                (now() - t0) / reps * 1e6);
	}
	return 0;
    }
    for (int a = 3; a < argc; ++a) {
	const std::ptrdiff_t n = std::atol(argv[a]);
	switch (argv[2][0]) {
	case 's':
	    run<float>(argv[1], n, threads);
	    break;
	case 'd':
	    run<double>(argv[1], n, threads);
	    break;
	case 'c':
	    run<std::complex<float>>(argv[1], n, threads);
	    break;
	case 'z':
	    run<std::complex<double>>(argv[1], n, threads);
	    break;
	}
    }
    return 0;
}
