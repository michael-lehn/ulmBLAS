//
//  potrf, getrf (and getri) through the C++ template interface of ulmBLAS
//  with general strides, and OpenBLAS (dlopen, column major only).
//
//      bench routine n s reps [openblas.so]
//
//  routine: potrf, getrf or getri (double).  s: row stride of A, column
//  stride s*n (s = 1: column major).  Prints the best time of reps runs
//  and GFLOPS.  With an OpenBLAS library the same with OpenBLAS (s must
//  be 1).  Threads: ULMBLAS_NUM_THREADS, OMP_NUM_THREADS and
//  OPENBLAS_NUM_THREADS of the environment.
//
//  Built with the library of the build (libcblas.a, libclapack.a), see
//  run.sh.
//
#include <ulmblas/lapack/getrf.h>
#include <ulmblas/lapack/getri.h>
#include <ulmblas/lapack/potrf.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <dlfcn.h>

typedef std::ptrdiff_t Idx;

namespace {

double
now()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

typedef void (*Potrf)(const char *, const int *, double *, const int *, int *,
                      std::size_t);
typedef void (*Getrf)(const int *, const int *, double *, const int *, int *,
                      int *);
typedef void (*Getri)(const int *, double *, const int *, const int *, double *,
                      const int *, int *);

} // namespace

int
main(int argc, char **argv)
{
    if (argc < 5) {
	std::fprintf(stderr, "usage: bench potrf|getrf|getri n s reps "
	                     "[openblas.so]\n");
	return 2;
    }
    const char *routine = argv[1];
    const Idx n = std::atol(argv[2]);
    const Idx s = std::atol(argv[3]);
    const int reps = std::atoi(argv[4]);
    const char *ob = argc > 5 ? argv[5] : nullptr;
    const bool chol = std::strcmp(routine, "potrf") == 0;
    const bool inv = std::strcmp(routine, "getri") == 0;

    //  the matrix (column major), diagonally dominant
    std::mt19937_64 gen(1);
    std::uniform_real_distribution<double> dist(-1, 1);
    std::vector<double> A0(std::size_t(n * n));
    for (Idx j = 0; j < n; ++j) {
	for (Idx i = 0; i < n; ++i) {
	    A0[i + j * n] = dist(gen);
	}
    }
    if (chol) {
	for (Idx j = 0; j < n; ++j) {
	    for (Idx i = 0; i < j; ++i) {
		A0[i + j * n] = A0[j + i * n];
	    }
	}
    }
    for (Idx i = 0; i < n; ++i) {
	A0[i + i * n] += double(n);
    }

    const Idx rs = s, cs = s * n;
    std::vector<double> A(std::size_t(s * n * n), 0.0);
    std::vector<Idx> piv(static_cast<std::size_t>(n));
    std::vector<int> ipiv(static_cast<std::size_t>(n));
    std::vector<double> work(std::size_t(64 * n));

    Potrf obPotrf = nullptr;
    Getrf obGetrf = nullptr;
    Getri obGetri = nullptr;
    if (ob) {
	void *h = dlopen(ob, RTLD_NOW | RTLD_LOCAL);
	if (!h) {
	    std::fprintf(stderr, "%s\n", dlerror());
	    return 1;
	}
	obPotrf = reinterpret_cast<Potrf>(dlsym(h, "dpotrf_"));
	obGetrf = reinterpret_cast<Getrf>(dlsym(h, "dgetrf_"));
	obGetri = reinterpret_cast<Getri>(dlsym(h, "dgetri_"));
    }

    const double flops =
        chol ? double(n) * n * n / 3 : (inv ? 4.0 / 3 : 2.0 / 3) * n * n * n;
    double best = 1e30;
    Idx info = 0;
    for (int r = 0; r < reps; ++r) {
	for (Idx j = 0; j < n; ++j) {
	    for (Idx i = 0; i < n; ++i) {
		A[std::size_t(i * rs + j * cs)] = A0[std::size_t(i + j * n)];
	    }
	}
	if (inv) {
	    //  getri times only the inverse (from the LU factorization)
	    if (ob) {
		int ni = int(n), lda = int(n), inf = 0;
		obGetrf(&ni, &ni, A.data(), &lda, ipiv.data(), &inf);
	    } else {
		ulmBLAS::getrf(n, n, A.data(), rs, cs, piv.data(), Idx(1));
	    }
	}
	const double t0 = now();
	if (ob) {
	    int ni = int(n), lda = int(n), inf = 0;
	    if (chol) {
		obPotrf("L", &ni, A.data(), &lda, &inf, 1);
	    } else if (inv) {
		int lw = int(work.size());
		obGetri(&ni, A.data(), &lda, ipiv.data(), work.data(), &lw,
		        &inf);
	    } else {
		obGetrf(&ni, &ni, A.data(), &lda, ipiv.data(), &inf);
	    }
	    info = inf;
	} else if (chol) {
	    info = ulmBLAS::potrf(true, n, A.data(), rs, cs);
	} else if (inv) {
	    info = ulmBLAS::getri(n, A.data(), rs, cs, piv.data(), Idx(1));
	} else {
	    info = ulmBLAS::getrf(n, n, A.data(), rs, cs, piv.data(), Idx(1));
	}
	const double t = now() - t0;
	if (t < best) {
	    best = t;
	}
    }
    std::printf("%s %s n=%ld s=%ld time=%.4f gflops=%.2f info=%ld\n",
                ob ? "openblas" : "ulmblas", routine, long(n), long(s), best,
                flops / best * 1e-9, long(info));
    return 0;
}
