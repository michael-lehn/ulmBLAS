/*
 *  Level 2 benchmark through CBLAS, library loaded with dlopen:
 *
 *      l2bench LIB OP N...
 *
 *  OP: dgemv_n dgemv_t sgemv_n sgemv_t zgemv_n zgemv_t zgemv_c dger zgeru
 *      dtrsv_ln dtrsv_lt dtrsv_un dtrsv_ut ztrsv_ln ztrsv_lt ztrsv_un ztrsv_ut
 *      dgemv_rn dgemv_rt (row major)
 *
 *  Column major, lda = m = n, unit increments, alpha = 1, beta = 1 (gemv),
 *  alpha = 1e-9 (ger).  Operands stay in place between calls (in cache if
 *  they fit).  The number of threads comes from the environment
 *  (OPENBLAS_NUM_THREADS, ULMBLAS_NUM_THREADS).
 *
 *  Small problems depend much on where A, x, y lie relative to each other
 *  (4K aliasing: zgemv_t n = 100 on heim, OpenBLAS 13 to 58 GFLOP/s for
 *  different offsets of y in its page).  So each size is measured with
 *  TRIALS (default 5) layouts: A, x, y at offsets (multiples of 64 bytes)
 *  from page boundaries, pseudo-random from n and the trial (the same for
 *  all libraries).  Per layout the best of 5 samples of at least 20 ms;
 *  output: the median over the layouts, GFLOP/s.
 *
 *  ROWS=r, COLS=c: dgemv, zgemv with the first r rows, c columns of A only
 *  (e.g. the part of one thread).
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum
{
    RowMajor = 102,
    ColMajor = 101
};
enum
{
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113
};
enum
{
    Upper = 121,
    Lower = 122
};
enum
{
    NonUnit = 131
};

typedef void (*dgemv_t)(int, int, int, int, double, const double *, int,
                        const double *, int, double, double *, int);
typedef void (*sgemv_t)(int, int, int, int, float, const float *, int,
                        const float *, int, float, float *, int);
typedef void (*zgemv_t)(int, int, int, int, const void *, const void *, int,
                        const void *, int, const void *, void *, int);
typedef void (*dger_t)(int, int, int, double, const double *, int,
                       const double *, int, double *, int);
typedef void (*zger_t)(int, int, int, const void *, const void *, int,
                       const void *, int, void *, int);
typedef void (*dtrsv_t)(int, int, int, int, int, const double *, int, double *,
                        int);
typedef void (*ztrsv_t)(int, int, int, int, int, const void *, int, void *,
                        int);

static dgemv_t dgemv;
static sgemv_t sgemv;
static zgemv_t zgemv;
static dger_t dger;
static zger_t zgeru;
static dtrsv_t dtrsv;
static ztrsv_t ztrsv;

static double
now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void *lib;

static void *
sym(const char *name)
{
    void *f = dlsym(lib, name);
    if (!f) {
	fprintf(stderr, "missing %s\n", name);
	exit(1);
    }
    return f;
}

static uint64_t state;

static double
rnd(void)
{
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return (double)(state >> 11) / 9007199254740992.0 - 0.5;
}

/* len elements of size bytes at offset off from a page boundary */
static void *
alloc(size_t len, size_t size, size_t off, void **base)
{
    *base = aligned_alloc(4096, (len * size + off + 4095) / 4096 * 4096);
    return (char *)*base + off;
}

static size_t
offset(void)
{
    return (size_t)((rnd() + 0.5) * 64) % 64 * 64;
}

static double
measure(const char *op, int n, int trial)
{
    const size_t nn = (size_t)n * n;
    const int cplx = op[0] == 'z';
    const int sgl = op[0] == 's';
    const int tr = strstr(op, "trsv") != 0;
    const int c = cplx ? 2 : 1;
    const size_t sz = sgl ? sizeof(float) : sizeof(double);
    void *bA, *bx, *by, *bx0;

    state = 1000003ull * n + 7919ull * trial + 1;
    void *A = alloc(c * nn, sz, offset(), &bA);
    void *x = alloc(c * n, sz, offset(), &bx);
    void *y = alloc(c * n, sz, offset(), &by);
    void *x0 = alloc(c * n, sz, offset(), &bx0);
    for (size_t i = 0; i < c * nn; ++i) {
	if (sgl)
	    ((float *)A)[i] = rnd();
	else
	    ((double *)A)[i] = rnd();
    }
    for (int i = 0; i < c * n; ++i) {
	if (sgl)
	    ((float *)x)[i] = rnd(), ((float *)y)[i] = rnd();
	else
	    ((double *)x)[i] = rnd(), ((double *)y)[i] = rnd();
    }
    memcpy(x0, x, c * n * sz);
    if (tr) {
	/* well conditioned: diagonal n, the rest in [-0.5, 0.5] */
	for (int i = 0; i < n; ++i) {
	    ((double *)A)[c * ((size_t)i * n + i)] = n;
	}
    }
    const double one[2] = {1, 0}, small[2] = {1e-9, 0};
    const int rows = getenv("ROWS") ? atoi(getenv("ROWS")) : n;
    const int cols = getenv("COLS") ? atoi(getenv("COLS")) : n;
    double flops = (cplx ? 8.0 : 2.0) * nn * rows / n * cols / n;
    if (tr)
	flops /= 2;

    double best = 1e30;
    long reps = 1;
    for (int s = 0; s < 6; ++s) {
	double t0 = now();
	for (long r = 0; r < reps; ++r) {
	    if (!strcmp(op, "dgemv_n") || !strcmp(op, "dgemv_t")) {
		dgemv(ColMajor, op[6] == 'n' ? NoTrans : Trans, rows, cols, 1.0,
		      A, n, x, 1, 1.0, y, 1);
	    } else if (!strcmp(op, "dgemv_rn") || !strcmp(op, "dgemv_rt")) {
		dgemv(RowMajor, op[7] == 'n' ? NoTrans : Trans, n, n, 1.0, A, n,
		      x, 1, 1.0, y, 1);
	    } else if (!strcmp(op, "sgemv_n") || !strcmp(op, "sgemv_t")) {
		sgemv(ColMajor, op[6] == 'n' ? NoTrans : Trans, n, n, 1.0f, A,
		      n, x, 1, 1.0f, y, 1);
	    } else if (!strncmp(op, "zgemv_", 6)) {
		int t = op[6] == 'n'   ? NoTrans
		        : op[6] == 't' ? Trans
		                       : ConjTrans;
		zgemv(ColMajor, t, rows, cols, one, A, n, x, 1, one, y, 1);
	    } else if (!strcmp(op, "dger")) {
		dger(ColMajor, n, n, 1e-9, x, 1, y, 1, A, n);
	    } else if (!strcmp(op, "zgeru")) {
		zgeru(ColMajor, n, n, small, x, 1, y, 1, A, n);
	    } else if (tr) {
		int ul = op[6] == 'l' ? Lower : Upper;
		int t = op[7] == 'n' ? NoTrans : Trans;
		memcpy(x, x0, c * n * sz);
		if (cplx) {
		    ztrsv(ColMajor, ul, t, NonUnit, n, A, n, x, 1);
		} else {
		    dtrsv(ColMajor, ul, t, NonUnit, n, A, n, x, 1);
		}
	    } else {
		fprintf(stderr, "unknown op %s\n", op);
		exit(1);
	    }
	}
	double t = (now() - t0) / reps;
	if (s == 0) {
	    /* the first sample: calibration (and warm up) */
	    reps = (long)(0.02 / t) + 1;
	    continue;
	}
	if (t < best)
	    best = t;
    }
    free(bA), free(bx), free(by), free(bx0);
    return flops / best * 1e-9;
}

static int
cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int
main(int argc, char **argv)
{
    if (argc < 4) {
	fprintf(stderr, "usage: l2bench LIB OP N...\n");
	return 1;
    }
    lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
	fprintf(stderr, "%s\n", dlerror());
	return 1;
    }
    dgemv = (dgemv_t)sym("cblas_dgemv");
    sgemv = (sgemv_t)sym("cblas_sgemv");
    zgemv = (zgemv_t)sym("cblas_zgemv");
    dger = (dger_t)sym("cblas_dger");
    zgeru = (zger_t)sym("cblas_zgeru");
    dtrsv = (dtrsv_t)sym("cblas_dtrsv");
    ztrsv = (ztrsv_t)sym("cblas_ztrsv");
    const int trials = getenv("TRIALS") ? atoi(getenv("TRIALS")) : 5;
    for (int a = 3; a < argc; ++a) {
	const int n = atoi(argv[a]);
	double g[64];
	for (int t = 0; t < trials && t < 64; ++t) {
	    g[t] = measure(argv[2], n, t);
	}
	qsort(g, trials, sizeof(double), cmp);
	printf("%s %d %.2f\n", argv[2], n, g[trials / 2]);
	fflush(stdout);
    }
    return 0;
}
