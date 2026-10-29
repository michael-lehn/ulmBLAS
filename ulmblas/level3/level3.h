#ifndef ULMBLAS_LEVEL3_LEVEL3_H
#define ULMBLAS_LEVEL3_LEVEL3_H 1

#include <ulmblas/config/simd.h>
#include <ulmblas/level3/gemm.h>
#include <ulmblas/level3/helmm.h>
#include <ulmblas/level3/helr2k.h>
#include <ulmblas/level3/helrk.h>
#include <ulmblas/level3/heumm.h>
#include <ulmblas/level3/heur2k.h>
#include <ulmblas/level3/heurk.h>
#include <ulmblas/level3/level3.h>
#include <ulmblas/level3/mkernel/mgemm.h>
#include <ulmblas/level3/mkernel/msylrk.h>
#include <ulmblas/level3/mkernel/msyurk.h>
#include <ulmblas/level3/mkernel/mtrlmm.h>
#include <ulmblas/level3/mkernel/mtrlsm.h>
#include <ulmblas/level3/mkernel/mtrumm.h>
#include <ulmblas/level3/mkernel/mtrusm.h>
#include <ulmblas/level3/pack/gepack.h>
#include <ulmblas/level3/pack/helpack.h>
#include <ulmblas/level3/pack/heupack.h>
#include <ulmblas/level3/pack/sylpack.h>
#include <ulmblas/level3/pack/syupack.h>
#include <ulmblas/level3/pack/trlpack.h>
#include <ulmblas/level3/pack/trlspack.h>
#include <ulmblas/level3/pack/trupack.h>
#include <ulmblas/level3/pack/truspack.h>
#include <ulmblas/level3/sylmm.h>
#include <ulmblas/level3/sylr2k.h>
#include <ulmblas/level3/sylrk.h>
#include <ulmblas/level3/syumm.h>
#include <ulmblas/level3/syur2k.h>
#include <ulmblas/level3/syurk.h>
#include <ulmblas/level3/trlmm.h>
#include <ulmblas/level3/trlsm.h>
#include <ulmblas/level3/trumm.h>
#include <ulmblas/level3/trusm.h>
#include <ulmblas/level3/ukernel/ref/ugemm.h>
#include <ulmblas/level3/ukernel/ref/utrlsm.h>
#include <ulmblas/level3/ukernel/ref/utrusm.h>
#if defined(USE_SSE)
#   include <ulmblas/level3/ukernel/sse/ugemm.h>
#endif
#if defined(USE_SSE) && !defined(USE_AVX) && !defined(USE_BLIS_DGEMM)
#   include <ulmblas/level3/ukernel/sse/utrlsm.h>
#   include <ulmblas/level3/ukernel/sse/utrusm.h>
#endif
#if defined(USE_AVX) && !defined(USE_BLIS_DGEMM)
#   include <ulmblas/level3/ukernel/avx/ugemm.h>
#endif
#if defined(USE_NEON) && !defined(USE_SME)
#   include <ulmblas/level3/ukernel/neon/ugemm.h>
#endif
#if defined(USE_SME)
#   include <ulmblas/level3/ukernel/sme/ugemm.h>
#endif
#include <ulmblas/level3/ukernel/ugemm.h>
#include <ulmblas/level3/ukernel/usylrk.h>
#include <ulmblas/level3/ukernel/usyurk.h>
#include <ulmblas/level3/ukernel/utrlsm.h>
#include <ulmblas/level3/ukernel/utrusm.h>

#endif // ULMBLAS_LEVEL3_LEVEL3_H
