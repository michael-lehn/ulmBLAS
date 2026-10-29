/*
 *  blis-probe: what the BLIS build of the benchmarks (PR #921, m4sme_p)
 *  does at run time for cblas_dgemm (column major, NoTrans, alpha = beta =
 *  1, one thread), without modifying BLIS:
 *
 *  - the configuration selected at run time (bli_arch_query_id) and the
 *    default and maximum block sizes MR, NR, MC, KC, NC and the thresholds
 *    of the small-matrix (sup) path in the context of that configuration;
 *  - for each shape M_N_K, every call of the double precision gemm micro
 *    kernel: the micro kernel registered in the context is replaced by a
 *    wrapper that records (m, n, k) of the call and calls the original
 *    kernel.  Printed: the number of calls per (m, n, k) and the k blocks
 *    (k of the calls in order of first appearance), and a check of C
 *    against a naive product.
 *
 *      blis-probe M_N_K [M_N_K ...]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "blis.h"

typedef void (*dgemm_ukr_t)(dim_t, dim_t, dim_t, const void *, const void *,
                            const void *, const void *, void *, inc_t, inc_t,
                            const auxinfo_t *, const cntx_t *);

static dgemm_ukr_t real;

enum { MAXK = 64 };
static struct { dim_t m, n, k; long calls; } hist[MAXK];
static int nhist;

static void
wrap(dim_t m, dim_t n, dim_t k, const void *alpha, const void *a,
     const void *b, const void *beta, void *c, inc_t rs_c, inc_t cs_c,
     const auxinfo_t *data, const cntx_t *cntx)
{
    int i;
    for (i=0; i<nhist; ++i) {
        if (hist[i].m==m && hist[i].n==n && hist[i].k==k) break;
    }
    if (i==nhist && nhist<MAXK) {
        hist[nhist].m = m; hist[nhist].n = n; hist[nhist].k = k; ++nhist;
    }
    if (i<MAXK) ++hist[i].calls;
    real(m, n, k, alpha, a, b, beta, c, rs_c, cs_c, data, cntx);
}

int
main(int argc, char **argv)
{
    bli_init();
    cntx_t *cntx = (cntx_t *)bli_gks_query_cntx();
    printf("BLIS %s, configuration at run time: %s\n",
           bli_info_get_version_str(), bli_arch_string(bli_arch_query_id()));
    const char *name[] = { "MR", "NR", "MC", "KC", "NC", "MT", "NT", "KT" };
    kerid_t id[] = { BLIS_MR, BLIS_NR, BLIS_MC, BLIS_KC, BLIS_NC,
                     BLIS_MT, BLIS_NT, BLIS_KT };
    for (int i=0; i<8; ++i) {
        printf("double %s: default %ld, max %ld\n", name[i],
               (long)bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, id[i], cntx),
               (long)bli_cntx_get_blksz_max_dt(BLIS_DOUBLE, id[i], cntx));
    }
    real = (dgemm_ukr_t)bli_cntx_get_ukr_dt(BLIS_DOUBLE, BLIS_GEMM_UKR, cntx);
    bli_cntx_set_ukrs(cntx, BLIS_GEMM_UKR, BLIS_DOUBLE, (void_fp)wrap,
                      BLIS_VA_END);

    srand(2026);
    int bad = 0;
    for (int s=1; s<argc; ++s) {
        int m, n, k;
        if (sscanf(argv[s], "%d_%d_%d", &m, &n, &k)!=3) return 2;
        double *A = malloc(sizeof(double)*m*k), *B = malloc(sizeof(double)*k*n);
        double *C = malloc(sizeof(double)*m*n), *R = malloc(sizeof(double)*m*n);
        for (long i=0; i<(long)m*k; ++i) A[i] = 2.0*rand()/RAND_MAX-1;
        for (long i=0; i<(long)k*n; ++i) B[i] = 2.0*rand()/RAND_MAX-1;
        for (long i=0; i<(long)m*n; ++i) C[i] = R[i] = 2.0*rand()/RAND_MAX-1;
        nhist = 0; memset(hist, 0, sizeof(hist));
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                    A, m, B, k, 1.0, C, m);
        for (int j=0; j<n; ++j)
            for (int l=0; l<k; ++l)
                for (int i=0; i<m; ++i)
                    R[i+(long)j*m] += A[i+(long)l*m]*B[l+(long)j*k];
        double err = 0;
        for (long i=0; i<(long)m*n; ++i) err = fmax(err, fabs(C[i]-R[i]));
        int ok = err <= 1e-10*k;
        bad |= !ok;
        printf("%d %d %d: max |error| %.3g %s; micro kernel calls (m n k: calls):",
               m, n, k, err, ok ? "ok" : "WRONG");
        long total = 0;
        for (int i=0; i<nhist; ++i) {
            printf(" %ld %ld %ld: %ld,", (long)hist[i].m, (long)hist[i].n,
                   (long)hist[i].k, hist[i].calls);
            total += hist[i].calls;
        }
        printf(" total %ld; k blocks:", total);
        long seen[MAXK]; int ns = 0;
        for (int i=0; i<nhist; ++i) {
            int dup = 0;
            for (int j=0; j<ns; ++j) dup |= seen[j]==hist[i].k;
            if (!dup) { seen[ns++] = hist[i].k; printf(" %ld", (long)hist[i].k); }
        }
        printf("\n");
        free(A); free(B); free(C); free(R);
    }
    return bad;
}
