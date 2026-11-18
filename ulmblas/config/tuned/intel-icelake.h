//
//  Parameters of ulmBLAS for intel-icelake (ulmblas/config/tuned.h): measured
//  by make tune-params (tune/params/params.py), last on 2026-11-17, do not
//  edit.  The report: tune/results/intel-icelake/params.md.
//  Included by ulmblas/config/tuned/tuned.h (after the definition of Tuned).
//
#ifndef ULMBLAS_CONFIG_TUNED_INTEL_ICELAKE_H
#define ULMBLAS_CONFIG_TUNED_INTEL_ICELAKE_H 1

#if defined(__AVX512F__) && defined(__x86_64__) && !defined(USE_TESTPARAM)

#define ULMBLAS_TUNED_INTEL_ICELAKE 1

namespace ulmBLAS {
namespace tunedparam {

inline void
intel_icelake(Tuned &p)
{
    // clang-format off
    p.gemmMinWork[tunedC] = 24576;
    p.gemmThreads[tunedD] = {1, 963178, 0};
    p.gemmThreads[tunedZ] = {1, 170267, 24, 481589, 32, 963178, 0};
    p.gemvNThreads[tunedC] = {1, 24576, 2, 98304, 16, 786432, 32, 1572864, 48, 3145728, 0};
    p.gemvNThreads[tunedD] = {1, 24576, 3, 49152, 16, 98304, 32, 196608, 48, 393216, 0};
    p.gemvNThreads[tunedS] = {1, 98304, 2, 196608, 16, 393216, 48, 1572864, 0};
    p.gemvNThreads[tunedZ] = {1, 6144, 8, 49152, 12, 98304, 48, 196608, 0};
    p.gemvTThreads[tunedC] = {1, 12288, 4, 24576, 16, 98304, 24, 196608, 32, 786432, 48, 1572864, 0};
    p.gemvTThreads[tunedD] = {1, 12288, 8, 24576, 12, 49152, 16, 98304, 32, 196608, 48, 393216, 0};
    p.gemvTThreads[tunedS] = {1, 24576, 8, 49152, 12, 98304, 16, 196608, 24, 393216, 32, 1572864, 0};
    p.gemvTThreads[tunedZ] = {1, 6144, 4, 24576, 16, 98304, 32, 196608, 48, 393216, 0};
    p.gerThreads[tunedC] = {1, 6144, 8, 12288, 12, 24576, 16, 49152, 32, 786432, 0};
    p.gerThreads[tunedD] = {1, 6144, 2, 12288, 12, 24576, 16, 49152, 24, 98304, 32, 393216, 0};
    p.gerThreads[tunedS] = {1, 12288, 6, 24576, 16, 98304, 24, 393216, 32, 786432, 0};
    p.gerThreads[tunedZ] = {1, 6144, 4, 12288, 24, 49152, 32, 393216, 0};
    p.potrfNb = {48, 387, 96, 1549, 128, 3872, 192};
    p.potrfSerialN = 244;
    p.trmm.kb[tunedC] = {96, 3000, 256};
    p.trmm.kb[tunedZ] = {192, 3000, 256};
    p.trmm.minPanels[tunedS] = 1;
    p.trmm.minPanels[tunedZ] = 1;
    p.trmm.sharedBlocks[tunedS] = 2;
    p.trmm.sharedBlocks[tunedZ] = 6;
    p.trmm.sharedPanels[tunedS] = 4;
    p.trmm.sharedPanels[tunedZ] = 16;
    p.trsm.kb[tunedC] = {96, 3000, 256};
    p.trsm.kb[tunedZ] = {96, 3000, 256};
    p.trsm.minPanels[tunedS] = 1;
    p.trsm.sharedBlocks[tunedS] = 4;
    p.trsm.sharedPanels[tunedS] = 8;
    p.trsmInvert[tunedC] = 1;
    p.trsmInvert[tunedZ] = 1;
    // clang-format on
}

} // namespace tunedparam
} // namespace ulmBLAS

#endif // defined(__AVX512F__) && defined(__x86_64__) && !defined(USE_TESTPARAM)

#endif // ULMBLAS_CONFIG_TUNED_INTEL_ICELAKE_H
