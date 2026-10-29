#!/bin/sh
#  SGEMM, then CGEMM campaign on heim: the vendored AVX/AVX2/SSE3 kernels,
#  the reference build (first round only), native OpenBLAS and BLIS; core 3,
#  3 rounds
W=$(cd $(dirname $0) && pwd)
cd $W || exit 1
ONCE=ulmblas-ref sh $W/xbench/xbench-heim.sh s $W/src $W/srun \
    ulmblas-ref="-mavx -mfma" \
    ulmblas-haswell_6x16="-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_6x16" \
    ulmblas-haswell_16x6="-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_16x6" \
    ulmblas-haswell_24x4="-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_24x4" \
    ulmblas-haswell_4x24="-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_haswell_4x24" \
    ulmblas-sandybridge_8x8="-mavx -DULMBLAS_SGEMM_UKERNEL_blis_sandybridge_8x8" \
    ulmblas-penryn_8x4="-msse3 -DULMBLAS_SGEMM_UKERNEL_blis_penryn_8x4" \
    ulmblas-aocl_haswell_6x16="-mavx -mfma -DULMBLAS_SGEMM_UKERNEL_blis_aocl_haswell_6x16" \
    ulmblas-openblas-haswell_8x4="-mavx -mavx2 -mfma -DULMBLAS_SGEMM_UKERNEL_openblas_haswell_8x4" \
    > $W/srun.out 2>&1
ONCE=ulmblas-ref sh $W/xbench/xbench-heim.sh c $W/src $W/crun \
    ulmblas-ref="-mavx -mfma" \
    ulmblas-haswell_3x8="-mavx -mfma -DULMBLAS_CGEMM_UKERNEL_blis_haswell_3x8" \
    ulmblas-haswell_8x3="-mavx -mfma -DULMBLAS_CGEMM_UKERNEL_blis_haswell_8x3" \
    ulmblas-sandybridge_8x4="-mavx -DULMBLAS_CGEMM_UKERNEL_blis_sandybridge_8x4" \
    ulmblas-aocl_haswell_3x8="-mavx -mfma -DULMBLAS_CGEMM_UKERNEL_blis_aocl_haswell_3x8" \
    ulmblas-openblas-haswell_8x2="-mavx -mavx2 -mfma -DULMBLAS_CGEMM_UKERNEL_openblas_haswell_8x2" \
    > $W/crun.out 2>&1
echo "done $(date -Iseconds)" > $W/done.txt
