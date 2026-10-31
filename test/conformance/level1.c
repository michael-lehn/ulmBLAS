/*
 * Differential test of the level 1 BLAS: the same calls against ulmBLAS and
 * against the reference BLAS/CBLAS of LAPACK 3.12.1 (two executables, the
 * outputs are compared by compare.cc; see conformance.sh).  Every call gets
 * buffers with guard regions; the results and the complete buffers are
 * printed.
 */
#include <complex.h>
#include <stdio.h>
#include <string.h>

typedef float complex  fc;
typedef double complex dc;

/* Fortran 77 interface */
float  sasum_(const int *, const float *, const int *);
double dasum_(const int *, const double *, const int *);
float  scasum_(const int *, const fc *, const int *);
double dzasum_(const int *, const dc *, const int *);
float  snrm2_(const int *, const float *, const int *);
double dnrm2_(const int *, const double *, const int *);
float  scnrm2_(const int *, const fc *, const int *);
double dznrm2_(const int *, const dc *, const int *);
int    isamax_(const int *, const float *, const int *);
int    idamax_(const int *, const double *, const int *);
int    icamax_(const int *, const fc *, const int *);
int    izamax_(const int *, const dc *, const int *);
void   sscal_(const int *, const float *, float *, const int *);
void   dscal_(const int *, const double *, double *, const int *);
void   cscal_(const int *, const fc *, fc *, const int *);
void   zscal_(const int *, const dc *, dc *, const int *);
void   csscal_(const int *, const float *, fc *, const int *);
void   zdscal_(const int *, const double *, dc *, const int *);
void   saxpy_(const int *, const float *, const float *, const int *, float *, const int *);
void   daxpy_(const int *, const double *, const double *, const int *, double *, const int *);
void   caxpy_(const int *, const fc *, const fc *, const int *, fc *, const int *);
void   zaxpy_(const int *, const dc *, const dc *, const int *, dc *, const int *);
void   scopy_(const int *, const float *, const int *, float *, const int *);
void   dcopy_(const int *, const double *, const int *, double *, const int *);
void   ccopy_(const int *, const fc *, const int *, fc *, const int *);
void   zcopy_(const int *, const dc *, const int *, dc *, const int *);
void   sswap_(const int *, float *, const int *, float *, const int *);
void   dswap_(const int *, double *, const int *, double *, const int *);
void   cswap_(const int *, fc *, const int *, fc *, const int *);
void   zswap_(const int *, dc *, const int *, dc *, const int *);
float  sdot_(const int *, const float *, const int *, const float *, const int *);
double ddot_(const int *, const double *, const int *, const double *, const int *);
float  sdsdot_(const int *, const float *, const float *, const int *, const float *, const int *);
double dsdot_(const int *, const float *, const int *, const float *, const int *);
fc     cdotu_(const int *, const fc *, const int *, const fc *, const int *);
fc     cdotc_(const int *, const fc *, const int *, const fc *, const int *);
dc     zdotu_(const int *, const dc *, const int *, const dc *, const int *);
dc     zdotc_(const int *, const dc *, const int *, const dc *, const int *);
void   srot_(const int *, float *, const int *, float *, const int *, const float *, const float *);
void   drot_(const int *, double *, const int *, double *, const int *, const double *, const double *);
void   csrot_(const int *, fc *, const int *, fc *, const int *, const float *, const float *);
void   zdrot_(const int *, dc *, const int *, dc *, const int *, const double *, const double *);
void   srotm_(const int *, float *, const int *, float *, const int *, const float *);
void   drotm_(const int *, double *, const int *, double *, const int *, const double *);

/* CBLAS (index results as int: the low 32 bits of size_t or int) */
float  cblas_sasum(int, const float *, int);
double cblas_dasum(int, const double *, int);
float  cblas_scasum(int, const void *, int);
double cblas_dzasum(int, const void *, int);
float  cblas_snrm2(int, const float *, int);
double cblas_dnrm2(int, const double *, int);
float  cblas_scnrm2(int, const void *, int);
double cblas_dznrm2(int, const void *, int);
int    cblas_isamax(int, const float *, int);
int    cblas_idamax(int, const double *, int);
int    cblas_icamax(int, const void *, int);
int    cblas_izamax(int, const void *, int);
void   cblas_sscal(int, float, float *, int);
void   cblas_dscal(int, double, double *, int);
void   cblas_cscal(int, const void *, void *, int);
void   cblas_zscal(int, const void *, void *, int);
void   cblas_csscal(int, float, void *, int);
void   cblas_zdscal(int, double, void *, int);
void   cblas_saxpy(int, float, const float *, int, float *, int);
void   cblas_daxpy(int, double, const double *, int, double *, int);
void   cblas_caxpy(int, const void *, const void *, int, void *, int);
void   cblas_zaxpy(int, const void *, const void *, int, void *, int);
void   cblas_scopy(int, const float *, int, float *, int);
void   cblas_dcopy(int, const double *, int, double *, int);
void   cblas_ccopy(int, const void *, int, void *, int);
void   cblas_zcopy(int, const void *, int, void *, int);
void   cblas_sswap(int, float *, int, float *, int);
void   cblas_dswap(int, double *, int, double *, int);
void   cblas_cswap(int, void *, int, void *, int);
void   cblas_zswap(int, void *, int, void *, int);
float  cblas_sdot(int, const float *, int, const float *, int);
double cblas_ddot(int, const double *, int, const double *, int);
float  cblas_sdsdot(int, float, const float *, int, const float *, int);
double cblas_dsdot(int, const float *, int, const float *, int);
void   cblas_cdotu_sub(int, const void *, int, const void *, int, void *);
void   cblas_cdotc_sub(int, const void *, int, const void *, int, void *);
void   cblas_zdotu_sub(int, const void *, int, const void *, int, void *);
void   cblas_zdotc_sub(int, const void *, int, const void *, int, void *);
void   cblas_srot(int, float *, int, float *, int, float, float);
void   cblas_drot(int, double *, int, double *, int, double, double);
void   cblas_csrot(int, void *, int, void *, int, float, float);
void   cblas_zdrot(int, void *, int, void *, int, double, double);
void   cblas_srotm(int, float *, int, float *, int, const float *);
void   cblas_drotm(int, double *, int, double *, int, const double *);

#define LEN   24      /* buffer length */
#define OFS   8       /* x and y point to buffer + OFS (start of storage) */

static const int ns[]   = { 0, 1, 3 };
static const int incs[] = { -2, -1, 0, 1, 2 };

/* data: 1, -2, 3, -4, ... in the storage, guard values 1000 + i */
#define FILL(T, b, base)                                                     \
    for (int i_=0; i_<LEN; ++i_) {                                           \
        b[i_] = (T) ((i_>=OFS && i_<OFS+8)                                   \
                     ? ((i_-OFS+base)%2 ? -(i_-OFS+base) : (i_-OFS+base))    \
                     : 1000+i_);                                             \
    }

static void pr(double v)            { printf(" %.6g", v); }
static void prc(double complex v)   { printf(" (%.6g,%.6g)", creal(v), cimag(v)); }

#define PRBUF_R(b)  do { printf(" |"); for (int i_=0; i_<LEN; ++i_) pr(b[i_]); } while (0)
#define PRBUF_C(b)  do { printf(" |"); for (int i_=0; i_<LEN; ++i_) prc(b[i_]); } while (0)

/* complex data: real part as above, imaginary part = 1/2 of it */
#define FILLC(T, R, b, base)                                                 \
    do { R tmp_[LEN]; FILL(R, tmp_, base);                                   \
         for (int i_=0; i_<LEN; ++i_) b[i_] = tmp_[i_] + tmp_[i_]/2*I; } while (0)

#define HEAD(name, n, ix, iy) printf("%-14s n=%d incX=%2d incY=%2d:", name, n, ix, iy)

int
main(void)
{
    for (int a=0; a<3; ++a)
    for (int b=0; b<5; ++b)
    for (int c=0; c<5; ++c) {
        const int n = ns[a], ix = incs[b], iy = incs[c];
        float  xs[LEN], ys[LEN];
        double xd[LEN], yd[LEN];
        fc     xc[LEN], yc[LEN];
        dc     xz[LEN], yz[LEN];

        /* ---------------- one vector: only for c == 0 ---------------- */
        if (c==0) {
#define ONE_R(T, xb, F77, CB, PRR)                                           \
            FILL(T, xb, 1); HEAD(#F77, n, ix, 0); PRR(F77(&n, xb+OFS, &ix)); \
            PRBUF_R(xb); printf("\n");                                       \
            FILL(T, xb, 1); HEAD(#CB, n, ix, 0); PRR(CB(n, xb+OFS, ix));     \
            PRBUF_R(xb); printf("\n");
#define ONE_C(T, R, xb, F77, CB, PRR)                                        \
            FILLC(T, R, xb, 1); HEAD(#F77, n, ix, 0);                        \
            PRR(F77(&n, xb+OFS, &ix)); PRBUF_C(xb); printf("\n");            \
            FILLC(T, R, xb, 1); HEAD(#CB, n, ix, 0);                         \
            PRR(CB(n, xb+OFS, ix)); PRBUF_C(xb); printf("\n");
            ONE_R(float,  xs, sasum_,  cblas_sasum,  pr)
            ONE_R(double, xd, dasum_,  cblas_dasum,  pr)
            ONE_C(fc, float,  xc, scasum_, cblas_scasum, pr)
            ONE_C(dc, double, xz, dzasum_, cblas_dzasum, pr)
            ONE_R(float,  xs, snrm2_,  cblas_snrm2,  pr)
            ONE_R(double, xd, dnrm2_,  cblas_dnrm2,  pr)
            ONE_C(fc, float,  xc, scnrm2_, cblas_scnrm2, pr)
            ONE_C(dc, double, xz, dznrm2_, cblas_dznrm2, pr)
            ONE_R(float,  xs, isamax_, cblas_isamax, pr)
            ONE_R(double, xd, idamax_, cblas_idamax, pr)
            ONE_C(fc, float,  xc, icamax_, cblas_icamax, pr)
            ONE_C(dc, double, xz, izamax_, cblas_izamax, pr)

            { float  al = 3;  FILL(float, xs, 1);  HEAD("sscal_", n, ix, 0);
              sscal_(&n, &al, xs+OFS, &ix); PRBUF_R(xs); printf("\n");
              FILL(float, xs, 1);  HEAD("cblas_sscal", n, ix, 0);
              cblas_sscal(n, al, xs+OFS, ix); PRBUF_R(xs); printf("\n"); }
            { double al = 3;  FILL(double, xd, 1); HEAD("dscal_", n, ix, 0);
              dscal_(&n, &al, xd+OFS, &ix); PRBUF_R(xd); printf("\n");
              FILL(double, xd, 1); HEAD("cblas_dscal", n, ix, 0);
              cblas_dscal(n, al, xd+OFS, ix); PRBUF_R(xd); printf("\n"); }
            { fc al = 3 + 1*I; FILLC(fc, float, xc, 1); HEAD("cscal_", n, ix, 0);
              cscal_(&n, &al, xc+OFS, &ix); PRBUF_C(xc); printf("\n");
              FILLC(fc, float, xc, 1); HEAD("cblas_cscal", n, ix, 0);
              cblas_cscal(n, &al, xc+OFS, ix); PRBUF_C(xc); printf("\n"); }
            { dc al = 3 + 1*I; FILLC(dc, double, xz, 1); HEAD("zscal_", n, ix, 0);
              zscal_(&n, &al, xz+OFS, &ix); PRBUF_C(xz); printf("\n");
              FILLC(dc, double, xz, 1); HEAD("cblas_zscal", n, ix, 0);
              cblas_zscal(n, &al, xz+OFS, ix); PRBUF_C(xz); printf("\n"); }
            { float al = 3; FILLC(fc, float, xc, 1); HEAD("csscal_", n, ix, 0);
              csscal_(&n, &al, xc+OFS, &ix); PRBUF_C(xc); printf("\n");
              FILLC(fc, float, xc, 1); HEAD("cblas_csscal", n, ix, 0);
              cblas_csscal(n, al, xc+OFS, ix); PRBUF_C(xc); printf("\n"); }
            { double al = 3; FILLC(dc, double, xz, 1); HEAD("zdscal_", n, ix, 0);
              zdscal_(&n, &al, xz+OFS, &ix); PRBUF_C(xz); printf("\n");
              FILLC(dc, double, xz, 1); HEAD("cblas_zdscal", n, ix, 0);
              cblas_zdscal(n, al, xz+OFS, ix); PRBUF_C(xz); printf("\n"); }
        }

        /* ---------------- two vectors ---------------- */
#define TWO_R(T, xb, yb, NAME, CALL)                                         \
        FILL(T, xb, 1); FILL(T, yb, 2); HEAD(NAME, n, ix, iy); CALL;         \
        PRBUF_R(xb); PRBUF_R(yb); printf("\n");
#define TWO_C(T, R, xb, yb, NAME, CALL)                                      \
        FILLC(T, R, xb, 1); FILLC(T, R, yb, 2); HEAD(NAME, n, ix, iy); CALL; \
        PRBUF_C(xb); PRBUF_C(yb); printf("\n");

        { float al = 3;
          TWO_R(float, xs, ys, "saxpy_", saxpy_(&n, &al, xs+OFS, &ix, ys+OFS, &iy))
          TWO_R(float, xs, ys, "cblas_saxpy", cblas_saxpy(n, al, xs+OFS, ix, ys+OFS, iy)) }
        { double al = 3;
          TWO_R(double, xd, yd, "daxpy_", daxpy_(&n, &al, xd+OFS, &ix, yd+OFS, &iy))
          TWO_R(double, xd, yd, "cblas_daxpy", cblas_daxpy(n, al, xd+OFS, ix, yd+OFS, iy)) }
        { fc al = 3 + 1*I;
          TWO_C(fc, float, xc, yc, "caxpy_", caxpy_(&n, &al, xc+OFS, &ix, yc+OFS, &iy))
          TWO_C(fc, float, xc, yc, "cblas_caxpy", cblas_caxpy(n, &al, xc+OFS, ix, yc+OFS, iy)) }
        { dc al = 3 + 1*I;
          TWO_C(dc, double, xz, yz, "zaxpy_", zaxpy_(&n, &al, xz+OFS, &ix, yz+OFS, &iy))
          TWO_C(dc, double, xz, yz, "cblas_zaxpy", cblas_zaxpy(n, &al, xz+OFS, ix, yz+OFS, iy)) }

        TWO_R(float, xs, ys, "scopy_", scopy_(&n, xs+OFS, &ix, ys+OFS, &iy))
        TWO_R(float, xs, ys, "cblas_scopy", cblas_scopy(n, xs+OFS, ix, ys+OFS, iy))
        TWO_R(double, xd, yd, "dcopy_", dcopy_(&n, xd+OFS, &ix, yd+OFS, &iy))
        TWO_R(double, xd, yd, "cblas_dcopy", cblas_dcopy(n, xd+OFS, ix, yd+OFS, iy))
        TWO_C(fc, float, xc, yc, "ccopy_", ccopy_(&n, xc+OFS, &ix, yc+OFS, &iy))
        TWO_C(fc, float, xc, yc, "cblas_ccopy", cblas_ccopy(n, xc+OFS, ix, yc+OFS, iy))
        TWO_C(dc, double, xz, yz, "zcopy_", zcopy_(&n, xz+OFS, &ix, yz+OFS, &iy))
        TWO_C(dc, double, xz, yz, "cblas_zcopy", cblas_zcopy(n, xz+OFS, ix, yz+OFS, iy))

        TWO_R(float, xs, ys, "sswap_", sswap_(&n, xs+OFS, &ix, ys+OFS, &iy))
        TWO_R(float, xs, ys, "cblas_sswap", cblas_sswap(n, xs+OFS, ix, ys+OFS, iy))
        TWO_R(double, xd, yd, "dswap_", dswap_(&n, xd+OFS, &ix, yd+OFS, &iy))
        TWO_R(double, xd, yd, "cblas_dswap", cblas_dswap(n, xd+OFS, ix, yd+OFS, iy))
        TWO_C(fc, float, xc, yc, "cswap_", cswap_(&n, xc+OFS, &ix, yc+OFS, &iy))
        TWO_C(fc, float, xc, yc, "cblas_cswap", cblas_cswap(n, xc+OFS, ix, yc+OFS, iy))
        TWO_C(dc, double, xz, yz, "zswap_", zswap_(&n, xz+OFS, &ix, yz+OFS, &iy))
        TWO_C(dc, double, xz, yz, "cblas_zswap", cblas_zswap(n, xz+OFS, ix, yz+OFS, iy))

        TWO_R(float, xs, ys, "sdot_", pr(sdot_(&n, xs+OFS, &ix, ys+OFS, &iy)))
        TWO_R(float, xs, ys, "cblas_sdot", pr(cblas_sdot(n, xs+OFS, ix, ys+OFS, iy)))
        TWO_R(double, xd, yd, "ddot_", pr(ddot_(&n, xd+OFS, &ix, yd+OFS, &iy)))
        TWO_R(double, xd, yd, "cblas_ddot", pr(cblas_ddot(n, xd+OFS, ix, yd+OFS, iy)))
        { float sb = 5;
          TWO_R(float, xs, ys, "sdsdot_", pr(sdsdot_(&n, &sb, xs+OFS, &ix, ys+OFS, &iy)))
          TWO_R(float, xs, ys, "cblas_sdsdot", pr(cblas_sdsdot(n, sb, xs+OFS, ix, ys+OFS, iy))) }
        TWO_R(float, xs, ys, "dsdot_", pr(dsdot_(&n, xs+OFS, &ix, ys+OFS, &iy)))
        TWO_R(float, xs, ys, "cblas_dsdot", pr(cblas_dsdot(n, xs+OFS, ix, ys+OFS, iy)))
        TWO_C(fc, float, xc, yc, "cdotu_", prc(cdotu_(&n, xc+OFS, &ix, yc+OFS, &iy)))
        TWO_C(fc, float, xc, yc, "cdotc_", prc(cdotc_(&n, xc+OFS, &ix, yc+OFS, &iy)))
        TWO_C(dc, double, xz, yz, "zdotu_", prc(zdotu_(&n, xz+OFS, &ix, yz+OFS, &iy)))
        TWO_C(dc, double, xz, yz, "zdotc_", prc(zdotc_(&n, xz+OFS, &ix, yz+OFS, &iy)))
        { fc r; dc s;
          TWO_C(fc, float, xc, yc, "cblas_cdotu_sub", (cblas_cdotu_sub(n, xc+OFS, ix, yc+OFS, iy, &r), prc(r)))
          TWO_C(fc, float, xc, yc, "cblas_cdotc_sub", (cblas_cdotc_sub(n, xc+OFS, ix, yc+OFS, iy, &r), prc(r)))
          TWO_C(dc, double, xz, yz, "cblas_zdotu_sub", (cblas_zdotu_sub(n, xz+OFS, ix, yz+OFS, iy, &s), prc(s)))
          TWO_C(dc, double, xz, yz, "cblas_zdotc_sub", (cblas_zdotc_sub(n, xz+OFS, ix, yz+OFS, iy, &s), prc(s))) }

        { float cs = 1, sn = 2; double cd = 1, sd = 2;
          TWO_R(float, xs, ys, "srot_", srot_(&n, xs+OFS, &ix, ys+OFS, &iy, &cs, &sn))
          TWO_R(float, xs, ys, "cblas_srot", cblas_srot(n, xs+OFS, ix, ys+OFS, iy, cs, sn))
          TWO_R(double, xd, yd, "drot_", drot_(&n, xd+OFS, &ix, yd+OFS, &iy, &cd, &sd))
          TWO_R(double, xd, yd, "cblas_drot", cblas_drot(n, xd+OFS, ix, yd+OFS, iy, cd, sd))
          TWO_C(fc, float, xc, yc, "csrot_", csrot_(&n, xc+OFS, &ix, yc+OFS, &iy, &cs, &sn))
          TWO_C(fc, float, xc, yc, "cblas_csrot", cblas_csrot(n, xc+OFS, ix, yc+OFS, iy, cs, sn))
          TWO_C(dc, double, xz, yz, "zdrot_", zdrot_(&n, xz+OFS, &ix, yz+OFS, &iy, &cd, &sd))
          TWO_C(dc, double, xz, yz, "cblas_zdrot", cblas_zdrot(n, xz+OFS, ix, yz+OFS, iy, cd, sd)) }

        { float  ps[5] = { -1, 1, 2, 3, 4 };
          double pd[5] = { -1, 1, 2, 3, 4 };
          TWO_R(float, xs, ys, "srotm_", srotm_(&n, xs+OFS, &ix, ys+OFS, &iy, ps))
          TWO_R(float, xs, ys, "cblas_srotm", cblas_srotm(n, xs+OFS, ix, ys+OFS, iy, ps))
          TWO_R(double, xd, yd, "drotm_", drotm_(&n, xd+OFS, &ix, yd+OFS, &iy, pd))
          TWO_R(double, xd, yd, "cblas_drotm", cblas_drotm(n, xd+OFS, ix, yd+OFS, iy, pd)) }
    }
    return 0;
}
