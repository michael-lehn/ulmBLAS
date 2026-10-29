#!/bin/bash
#  CGEMM campaign on an exclusive Ice Lake node, after the test jobs of the
#  CGEMM kernels (DEP: their job ids, afterok)
U=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench/up/xbench
cd $U || exit 1
A512="-mavx -mfma -mavx512f"
ONCE=ulmblas-ref sbatch -p cpu_il -J xbench-c-icelake --dependency=afterok:$DEP \
    xbench-uc3.sbatch c icelake $U/ct.tgz \
    ulmblas-ref="-mavx -mfma" \
    ulmblas-haswell_3x8="-mavx -mfma -DULMBLAS_CGEMM_UKERNEL_blis_haswell_3x8" \
    ulmblas-haswell_8x3="-mavx -mfma -DULMBLAS_CGEMM_UKERNEL_blis_haswell_8x3" \
    ulmblas-sandybridge_8x4="-mavx -DULMBLAS_CGEMM_UKERNEL_blis_sandybridge_8x4" \
    ulmblas-aocl_haswell_3x8="-mavx -mfma -DULMBLAS_CGEMM_UKERNEL_blis_aocl_haswell_3x8" \
    ulmblas-openblas-haswell_8x2="-mavx -mavx2 -mfma -DULMBLAS_CGEMM_UKERNEL_openblas_haswell_8x2" \
    ulmblas-openblas-skylakex_8x2="$A512 -DULMBLAS_CGEMM_UKERNEL_openblas_skylakex_8x2"
squeue --me
