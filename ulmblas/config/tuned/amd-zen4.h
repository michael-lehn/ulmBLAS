//
//  Parameters of ulmBLAS for amd-zen4 (ulmblas/config/tuned.h): measured
//  by make tune-params (tune/params/params.py), last on 2026-11-17, do not
//  edit.  The report: tune/results/amd-zen4/params.md.
//  Included by ulmblas/config/tuned/tuned.h (after the definition of Tuned).
//
#ifndef ULMBLAS_CONFIG_TUNED_AMD_ZEN4_H
#define ULMBLAS_CONFIG_TUNED_AMD_ZEN4_H 1

#if defined(__AVX512F__) && defined(__x86_64__) && !defined(USE_TESTPARAM)

#define ULMBLAS_TUNED_AMD_ZEN4 1

namespace ulmBLAS {
namespace tunedparam {

inline void
amd_zen4(Tuned &p)
{
    // clang-format off
    p.gemmThreads[tunedC] = {1, 170267, 32, 481589, 48, 8289721, 64, 246573711, 0};
    p.gemmThreads[tunedD] = {1, 481589, 32, 1482910, 48, 8289721, 64, 11863283, 0};
    p.gemmThreads[tunedS] = {1, 963178, 24, 1482910, 32, 5448560, 48, 8289721, 64, 11863283, 0};
    p.gemmThreads[tunedZ] = {1, 170267, 32, 481589, 64, 11863283, 0};
    p.gemvNThreads[tunedC] = {1, 196608, 24, 393216, 48, 786432, 64, 1572864, 0};
    p.gemvNThreads[tunedD] = {1, 98304, 24, 196608, 32, 1572864, 48, 8388608, 64};
    p.gemvNThreads[tunedS] = {1, 98304, 8, 196608, 16, 1572864, 32};
    p.gemvNThreads[tunedZ] = {1, 6144, 2, 49152, 48, 98304, 0};
    p.gemvTThreads[tunedC] = {1, 24576, 12, 98304, 24, 196608, 32, 393216, 64, 3145728, 0};
    p.gemvTThreads[tunedD] = {1, 49152, 8, 196608, 24, 786432, 48, 1572864, 64, 3145728, 0};
    p.gemvTThreads[tunedS] = {1, 98304, 8, 196608, 12, 393216, 24, 1572864, 48, 3145728, 64, 8388608, 0};
    p.gemvTThreads[tunedZ] = {1, 6144, 2, 12288, 6, 49152, 12, 98304, 32, 196608, 64, 1572864, 0};
    p.gerThreads[tunedC] = {1, 24576, 12, 98304, 16, 196608, 24, 393216, 48, 1572864, 64, 3145728, 0};
    p.gerThreads[tunedD] = {1, 12288, 2, 24576, 6, 98304, 16, 196608, 24, 393216, 48, 786432, 64, 1572864, 0};
    p.gerThreads[tunedS] = {1, 49152, 4, 98304, 8, 196608, 16, 393216, 24, 786432, 48, 3145728, 0};
    p.gerThreads[tunedZ] = {1, 24576, 8, 49152, 12, 98304, 32, 196608, 48, 786432, 64, 1572864, 0};
    p.potrfNb = {192, 387, 128, 2449, 192};
    p.potrfSerialN = 244;
    p.trmm.kb[tunedC] = {192, 3000, 256};
    p.trmm.kb[tunedS] = {192, 3000, 256};
    p.trmm.minPanels[tunedC] = 2;
    p.trmm.minPanels[tunedZ] = 1;
    p.trmm.sharedBlocks[tunedC] = 1;
    p.trmm.sharedBlocks[tunedZ] = 4;
    p.trmm.sharedPanels[tunedC] = 2;
    p.trmm.sharedPanels[tunedZ] = 32;
    p.trsm.kb[tunedC] = {256};
    p.trsm.kb[tunedZ] = {256};
    p.trsm.minPanels[tunedC] = 1;
    p.trsm.minPanels[tunedZ] = 1;
    p.trsm.sharedBlocks[tunedC] = 1;
    p.trsm.sharedBlocks[tunedZ] = 4;
    p.trsm.sharedPanels[tunedC] = 32;
    p.trsm.sharedPanels[tunedZ] = 32;
    p.trsmInvert[tunedC] = 1;
    p.trsmInvert[tunedZ] = 0;
    // clang-format on
}

} // namespace tunedparam
} // namespace ulmBLAS

#endif // defined(__AVX512F__) && defined(__x86_64__) && !defined(USE_TESTPARAM)

#endif // ULMBLAS_CONFIG_TUNED_AMD_ZEN4_H
