/*
 *  blis-info: which configuration and DGEMM micro kernel a BLIS library
 *  selects at run time on this machine (upstream BLIS 2.0 and AOCL-BLIS).
 *  Prints the sub-configuration, the double precision block sizes and the
 *  address of the native dgemm micro kernel; link statically with -no-pie
 *  and look the address up with nm to get the kernel's name.
 */
#include <stdio.h>
#include "blis.h"

int
main(void)
{
    bli_init();
    arch_t id = bli_arch_query_id();
    const cntx_t *cntx = bli_gks_query_cntx();
    printf("blis_version %s\n", bli_info_get_version_str());
    printf("arch %s\n", bli_arch_string(id));
    printf("MR %ld NR %ld MC %ld KC %ld NC %ld\n",
           (long)bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx),
           (long)bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx),
           (long)bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MC, cntx),
           (long)bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_KC, cntx),
           (long)bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NC, cntx));
#ifndef BLIS_OLD_CNTX    /* BLIS 1.0 and newer; -DBLIS_OLD_CNTX for 0.9 */
    printf("dgemm_ukr %p row_pref %d\n",
           (void *)bli_cntx_get_ukr_dt(BLIS_DOUBLE, BLIS_GEMM_UKR, cntx),
           (int)bli_cntx_get_ukr_prefs_dt(BLIS_DOUBLE, BLIS_GEMM_UKR_ROW_PREF,
                                          cntx));
#else
    printf("dgemm_ukr %p row_pref %d\n",
           (void *)bli_cntx_get_l3_nat_ukr_dt(BLIS_DOUBLE, BLIS_GEMM_UKR,
                                              (cntx_t *)cntx),
           (int)bli_cntx_l3_nat_ukr_prefers_rows_dt(BLIS_DOUBLE,
                                                    BLIS_GEMM_UKR,
                                                    (cntx_t *)cntx));
#endif
    printf("threads %ld\n", (long)bli_thread_get_num_threads());
    return 0;
}
