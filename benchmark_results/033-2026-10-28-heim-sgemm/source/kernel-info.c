/*
 *  kernel-info: which GEMM micro kernel a native library uses for one data
 *  type (-DPREC=s, d, c or z) on this CPU.
 *
 *  -DOPENBLAS: core name and configuration of OpenBLAS, and its block
 *  sizes P, Q, R for the type (globals of non-DYNAMIC_ARCH builds).
 *  -DBLIS: architecture of BLIS, the implementation of its GEMM micro kernel
 *  for the type, the kernel function (compared with the known x86-64
 *  kernels), the block sizes MR, NR, MC, KC, NC, and the method of the
 *  GEMM operation (native, or induced: 1m computes complex GEMM with the
 *  real micro kernel, which is then reported as well).  -DBLIS_0X for the
 *  context API of BLIS 0.x (AOCL-BLAS 5.0).
 */
#include <stdio.h>

#define CAT_(a, b) a##b
#define CAT(a, b)  CAT_(a, b)

#if defined(OPENBLAS)

char *openblas_get_corename(void);
char *openblas_get_config(void);
/*  -DOB_PARAMS if the library defines the variables (non-DYNAMIC_ARCH
 *  builds; xbench-*.sh checks with nm): strong references, so that the
 *  linker takes them from the static library (a weak reference would not
 *  extract the archive member) */
#if defined(OB_PARAMS)
extern long CAT(PREC, gemm_p), CAT(PREC, gemm_q), CAT(PREC, gemm_r);
#endif

#define STR_(x) #x
#define STR(x)  STR_(x)

int
main(void)
{
    printf("openblas corename %s\nopenblas config %s\n",
           openblas_get_corename(), openblas_get_config());
#if defined(OB_PARAMS)
    printf("openblas %sgemm_p %ld %sgemm_q %ld %sgemm_r %ld\n",
           STR(PREC), CAT(PREC, gemm_p), STR(PREC), CAT(PREC, gemm_q),
           STR(PREC), CAT(PREC, gemm_r));
#else
    printf("openblas %sgemm_p/q/r: not variables of this build\n", STR(PREC));
#endif
    return 0;
}

#elif defined(BLIS)

#include <blis.h>

#define P_s BLIS_FLOAT
#define P_d BLIS_DOUBLE
#define P_c BLIS_SCOMPLEX
#define P_z BLIS_DCOMPLEX

#define KERNELS(X) \
    X(bli_sgemm_haswell_asm_6x16) X(bli_sgemm_haswell_asm_16x6) \
    X(bli_dgemm_haswell_asm_6x8) X(bli_dgemm_haswell_asm_8x6) \
    X(bli_cgemm_haswell_asm_3x8) X(bli_cgemm_haswell_asm_8x3) \
    X(bli_zgemm_haswell_asm_3x4) X(bli_zgemm_haswell_asm_4x3) \
    X(bli_sgemm_sandybridge_asm_8x8) X(bli_dgemm_sandybridge_asm_8x4) \
    X(bli_cgemm_sandybridge_asm_8x4) X(bli_zgemm_sandybridge_asm_4x4) \
    X(bli_sgemm_skx_asm_32x12_l2) X(bli_dgemm_skx_asm_16x14) \
    X(bli_sgemm_penryn_asm_8x4) X(bli_dgemm_penryn_asm_4x4) \
    X(bli_dgemm_zen4_asm_32x6) X(bli_dgemm_zen4_asm_8x24) \
    X(bli_zgemm_zen4_asm_12x4) X(bli_zgemm_zen4_asm_4x12)

/*  weak references bound to the kernel symbols by asm labels (blis.h
 *  declares the kernels with their own types) */
#define DECL(f) extern char CAT(k_, f) __asm__(#f) __attribute__((weak));
KERNELS(DECL)

#if defined(BLIS_0X)
#   define UKR(dt, cntx) bli_cntx_get_l3_nat_ukr_dt(dt, BLIS_GEMM_UKR, (cntx_t *)(cntx))
#else
#   define UKR(dt, cntx) bli_cntx_get_ukr_dt(dt, BLIS_GEMM_UKR, cntx)
#endif

static void
ukernel(num_t dt, const cntx_t *cntx, const char *what)
{
    void_fp     ukr  = UKR(dt, cntx);
    const char *name = "unknown";
#define CMP(f) if (&CAT(k_, f) && (void_fp)&CAT(k_, f)==ukr) name = #f;
    KERNELS(CMP)
    printf("blis %sgemm ukr impl %s\n", what,
           bli_info_get_gemm_ukr_impl_string(BLIS_NAT, dt));
    printf("blis %sgemm ukr %s (%p)\n", what, name, (void *)ukr);
    printf("blis %sMR %ld NR %ld MC %ld KC %ld NC %ld\n", what,
           (long)bli_cntx_get_blksz_def_dt(dt, BLIS_MR, (cntx_t *)cntx),
           (long)bli_cntx_get_blksz_def_dt(dt, BLIS_NR, (cntx_t *)cntx),
           (long)bli_cntx_get_blksz_def_dt(dt, BLIS_MC, (cntx_t *)cntx),
           (long)bli_cntx_get_blksz_def_dt(dt, BLIS_KC, (cntx_t *)cntx),
           (long)bli_cntx_get_blksz_def_dt(dt, BLIS_NC, (cntx_t *)cntx));
}

int
main(void)
{
    num_t        dt   = CAT(P_, PREC);
    const cntx_t *cntx = bli_gks_query_cntx();

    printf("blis arch %s\n", bli_arch_string(bli_arch_query_id()));
    ukernel(dt, cntx, "");
    printf("blis gemm method %s\n", bli_info_get_gemm_impl_string(dt));
    if (bli_is_complex(dt) && bli_ind_oper_find_avail(BLIS_GEMM, dt)!=BLIS_NAT) {
        ukernel(bli_dt_proj_to_real(dt), cntx, "real ");
    }
    return 0;
}

#endif
