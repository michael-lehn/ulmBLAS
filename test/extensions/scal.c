/*
 *  ?scal: the result of an element does not depend on its position in x
 *  (as test/numeric/level1.cc, T0122, for the shared library and, with
 *  the dispatching library, for every backend: check-shared.sh runs the
 *  program once per backend with ULMBLAS_BACKEND).
 *
 *  x holds n copies of one value; after x = alpha*x all elements must be
 *  equal bit for bit.  Offsets 0 ... 15 elements (all positions relative to
 *  a 64-byte boundary), lengths beyond the peel threshold (256) and the
 *  increments 1 and 2; the Fortran 77 interface (sscal_, dscal_, cscal_,
 *  zscal_) and CBLAS (cblas_?scal).  LAPACK's eigenvalue tests compare
 *  results of different paths bit for bit (test/known_issues.md, #0033).
 *  Exit status 1 on failure.
 */
#include <cblas.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void
sscal_(const ULMBLAS_INDEX_TYPE *n, const float *alpha, float *x,
       const ULMBLAS_INDEX_TYPE *incX);
void
dscal_(const ULMBLAS_INDEX_TYPE *n, const double *alpha, double *x,
       const ULMBLAS_INDEX_TYPE *incX);
void
cscal_(const ULMBLAS_INDEX_TYPE *n, const float *alpha, float *x,
       const ULMBLAS_INDEX_TYPE *incX);
void
zscal_(const ULMBLAS_INDEX_TYPE *n, const double *alpha, double *x,
       const ULMBLAS_INDEX_TYPE *incX);

enum
{
    MAX_N = 1000,
    MAX_OFFSET = 16,
    MAX_INC = 2
};

/*  xorshift64*, uniform in [-1, 1] */
static uint64_t state = 0x2545F4914F6CDD1DULL;

static double
uniform(void)
{
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return 2.0 * (double)((state * 0x2545F4914F6CDD1DULL) >> 11) /
               (double)(1ULL << 53) -
           1.0;
}

/*  x(0), x(inc), ..., x((n-1)*inc) equal bit for bit (elements of size
 *  bytes) */
static ptrdiff_t
allEqual(ptrdiff_t n, const char *x, ptrdiff_t inc, size_t size)
{
    for (ptrdiff_t i = 1; i < n; ++i) {
	if (memcmp(x, x + (size_t)(i * inc) * size, size) != 0) {
	    return 0;
	}
    }
    return 1;
}

/*  type: 0 s, 1 d, 2 c, 3 z; cblas: 0 Fortran 77, 1 CBLAS */
static ptrdiff_t
check(ptrdiff_t type, ptrdiff_t cblas, double ar, double ai, double zr,
      double zi, ptrdiff_t n, ptrdiff_t inc, ptrdiff_t offset)
{
    static double buffer[2 * (MAX_OFFSET + MAX_INC * MAX_N)];
    const size_t size = type == 0 ? 4 : type == 3 ? 16 : 8;
    const ptrdiff_t len = offset + 1 + (n - 1) * inc;
    float *xs = (float *)buffer;
    ULMBLAS_INDEX_TYPE n_ = (ULMBLAS_INDEX_TYPE)n;
    ULMBLAS_INDEX_TYPE inc_ = (ULMBLAS_INDEX_TYPE)inc;
    const float as[2] = {(float)ar, (float)ai};
    const double ad[2] = {ar, ai};

    for (ptrdiff_t i = 0; i < len; ++i) {
	switch (type) {
	case 0:
	    xs[i] = (float)zr;
	    break;
	case 1:
	    buffer[i] = zr;
	    break;
	case 2:
	    xs[2 * i] = (float)zr;
	    xs[2 * i + 1] = (float)zi;
	    break;
	default:
	    buffer[2 * i] = zr;
	    buffer[2 * i + 1] = zi;
	    break;
	}
    }
    char *x = (char *)buffer + (size_t)offset * size;
    switch (type * 2 + cblas) {
    case 0:
	sscal_(&n_, as, (float *)x, &inc_);
	break;
    case 1:
	cblas_sscal(n_, as[0], (float *)x, inc_);
	break;
    case 2:
	dscal_(&n_, ad, (double *)x, &inc_);
	break;
    case 3:
	cblas_dscal(n_, ad[0], (double *)x, inc_);
	break;
    case 4:
	cscal_(&n_, as, (float *)x, &inc_);
	break;
    case 5:
	cblas_cscal(n_, as, x, inc_);
	break;
    case 6:
	zscal_(&n_, ad, (double *)x, &inc_);
	break;
    default:
	cblas_zscal(n_, ad, x, inc_);
	break;
    }
    return allEqual(n, x, inc, size);
}

int
main(void)
{
    static const ptrdiff_t sizes[] = {1,   2,   3,   5,   7,   8,   13,
                                      16,  31,  33,  64,  70,  255, 256,
                                      257, 271, 300, 512, 1000};
    static const char *names[] = {"sscal", "dscal", "cscal", "zscal"};
    const ptrdiff_t numSizes = sizeof(sizes) / sizeof(sizes[0]);
    ptrdiff_t failures = 0, calls = 0;

    for (ptrdiff_t trial = 0; trial < 20; ++trial) {
	const double ar = uniform(), ai = uniform();
	const double zr = uniform(), zi = uniform();
	for (ptrdiff_t type = 0; type < 4; ++type) {
	    for (ptrdiff_t cblas = 0; cblas < 2; ++cblas) {
		for (ptrdiff_t inc = 1; inc <= MAX_INC; ++inc) {
		    for (ptrdiff_t s = 0; s < numSizes; ++s) {
			for (ptrdiff_t off = 0; off < MAX_OFFSET; ++off) {
			    ++calls;
			    if (check(type, cblas, ar, ai, zr, zi, sizes[s],
			              inc, off)) {
				continue;
			    }
			    if (++failures <= 5) {
				printf("%s%s: n = %td, inc = %td, offset "
				       "%td: equal elements of x give "
				       "different results\n",
				       cblas ? "cblas_" : "", names[type],
				       sizes[s], inc, off);
			    }
			}
		    }
		}
	    }
	}
    }
    if (failures) {
	printf("%td of %td calls position-dependent\n", failures, calls);
	return 1;
    }
    printf("PASS (%td calls)\n", calls);
    return 0;
}
