#!/bin/sh
#  ZGEMM campaign on heim: 5 ulmBLAS kernels, the reference build (first
#  round only), native OpenBLAS and BLIS; core 3, 3 rounds
W=/tmp/ulmblas-zgemm.zSIuAh
cd $W || exit 1
ONCE=ulmblas-ref nohup sh $W/xbench/xbench-heim.sh z $W/src2 $W/zrun \
    ulmblas-ref="-mavx -mfma" \
    ulmblas-haswell_3x4="-mavx -mfma -DULMBLAS_ZGEMM_UKERNEL_blis_haswell_3x4" \
    ulmblas-haswell_4x3="-mavx -mfma -DULMBLAS_ZGEMM_UKERNEL_blis_haswell_4x3" \
    ulmblas-sandybridge_4x4="-mavx -mfma -DULMBLAS_ZGEMM_UKERNEL_blis_sandybridge_4x4" \
    ulmblas-aocl_haswell_3x4="-mavx -mfma -DULMBLAS_ZGEMM_UKERNEL_blis_aocl_haswell_3x4" \
    ulmblas-openblas-haswell_4x2="-mavx -mavx2 -mfma -DULMBLAS_ZGEMM_UKERNEL_openblas_haswell_4x2" \
    > $W/zrun.out 2>&1 &
