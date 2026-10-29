#!/bin/bash
#  SGEMM campaign on an exclusive Ice Lake node: the 10 vendored SGEMM
#  kernels, the reference kernel (first round only) and the native libraries
U=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench/up/xbench
cd $U || exit 1
A512="-mavx -mfma -mavx512f"
ONCE=ulmblas-ref sbatch -p cpu_il -J xbench-s-icelake \
    xbench-uc3.sbatch s icelake $U/ct.tgz \
    ulmblas-ref="-mavx -mfma" \
    ulmblas-haswell_6x16="-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_6x16" \
    ulmblas-haswell_16x6="-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_16x6" \
    ulmblas-haswell_24x4="-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_24x4" \
    ulmblas-haswell_4x24="-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_4x24" \
    ulmblas-sandybridge_8x8="-mavx -DULMBLAS_SGEMM_UKERNEL_blis_sandybridge_8x8" \
    ulmblas-penryn_8x4="-msse3 -DULMBLAS_SGEMM_UKERNEL_blis_penryn_8x4" \
    ulmblas-aocl_haswell_6x16="-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_aocl_haswell_6x16" \
    ulmblas-openblas-haswell_8x4="-mavx -mavx2 -mfma -DULMBLAS_SGEMM_UKERNEL_openblas_haswell_8x4" \
    ulmblas-skx_32x12_l2="$A512 -DULMBLAS_SGEMM_UKERNEL_blis_skx_32x12_l2" \
    ulmblas-openblas-skylakex_16x4="$A512 -DULMBLAS_SGEMM_UKERNEL_openblas_skylakex_16x4"
squeue --me
