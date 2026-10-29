#!/bin/bash
#  ZGEMM campaign on an exclusive Ice Lake node, after the sanitizer run
U=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench/up/xbench
cd $U || exit 1
A512="-mavx -mfma -mavx512f -mavx512dq"
ONCE=ulmblas-ref sbatch -p cpu_il -J xbench-z-icelake --dependency=afterok:8028075 \
    xbench-uc3.sbatch z icelake $U/zt2.tgz \
    ulmblas-ref="-mavx -mfma" \
    ulmblas-zen4_12x4="$A512 -DULMBLAS_ZGEMM_UKERNEL_blis_zen4_12x4" \
    ulmblas-zen4_4x12="$A512 -DULMBLAS_ZGEMM_UKERNEL_blis_zen4_4x12" \
    ulmblas-openblas-skylakex_4x2="$A512 -DULMBLAS_ZGEMM_UKERNEL_openblas_skylakex_4x2" \
    ulmblas-haswell_3x4="-mavx -mfma -DULMBLAS_ZGEMM_UKERNEL_blis_haswell_3x4" \
    ulmblas-openblas-haswell_4x2="-mavx -mavx2 -mfma -DULMBLAS_ZGEMM_UKERNEL_openblas_haswell_4x2"
squeue --me
