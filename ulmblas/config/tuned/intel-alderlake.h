//
//  Parameters of ulmBLAS for intel-alderlake (ulmblas/config/tuned.h): measured
//  by make tune-params (tune/params/params.py), last on 2026-11-18, do not
//  edit.  The report: tune/results/intel-alderlake/params.md.
//  Included by ulmblas/config/tuned/tuned.h (after the definition of Tuned).
//
#ifndef ULMBLAS_CONFIG_TUNED_INTEL_ALDERLAKE_H
#define ULMBLAS_CONFIG_TUNED_INTEL_ALDERLAKE_H 1

#if defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512F__) &&          \
    defined(__x86_64__) && !defined(USE_TESTPARAM)

#define ULMBLAS_TUNED_INTEL_ALDERLAKE 1

namespace ulmBLAS {
namespace tunedparam {

inline void
intel_alderlake(Tuned &p)
{
    // clang-format off
    p.gemmMinWork[tunedD] = 196608;
    p.gemmMinWork[tunedS] = 196608;
    p.gemvNThreads[tunedC] = {1, 24576, 0};
    p.gemvNThreads[tunedD] = {1, 12288, 4, 24576, 0};
    p.gemvNThreads[tunedS] = {1, 98304, 0};
    p.gemvNThreads[tunedZ] = {1, 24576, 0};
    p.gemvTThreads[tunedC] = {1, 6144, 4, 12288, 0};
    p.gemvTThreads[tunedD] = {1, 24576, 0};
    p.gemvTThreads[tunedS] = {1, 24576, 0};
    p.gerThreads[tunedC] = {1, 6144, 0};
    p.gerThreads[tunedD] = {1, 6144, 0};
    p.gerThreads[tunedS] = {1, 12288, 0};
    p.gerThreads[tunedZ] = {1, 3072, 0};
    p.trmm.kb[tunedC] = {96, 3000, 256};
    p.trmm.kb[tunedD] = {128, 3000, 256};
    p.trmm.kb[tunedS] = {128, 3000, 256};
    p.trmm.kb[tunedZ] = {96, 3000, 256};
    p.trmm.minPanels[tunedC] = 2;
    p.trmm.minPanels[tunedZ] = 1;
    p.trmm.sharedBlocks[tunedC] = 6;
    p.trmm.sharedBlocks[tunedZ] = 3;
    p.trmm.sharedPanels[tunedC] = 64;
    p.trmm.sharedPanels[tunedZ] = 64;
    p.trsm.kb[tunedC] = {96, 3000, 256};
    p.trsm.kb[tunedD] = {192, 3000, 256};
    p.trsm.kb[tunedS] = {128, 3000, 256};
    p.trsm.kb[tunedZ] = {96, 3000, 256};
    p.trsm.minPanels[tunedC] = 4;
    p.trsm.minPanels[tunedZ] = 2;
    p.trsm.sharedBlocks[tunedC] = 6;
    p.trsm.sharedBlocks[tunedZ] = 3;
    p.trsm.sharedPanels[tunedC] = 64;
    p.trsm.sharedPanels[tunedZ] = 64;
    p.trsmInvert[tunedC] = 1;
    p.trsmInvert[tunedD] = 1;
    p.trsmInvert[tunedS] = 1;
    p.trsmInvert[tunedZ] = 0;
    // clang-format on
}

} // namespace tunedparam
} // namespace ulmBLAS

#endif // defined(__AVX2__) && defined(__FMA__) && !defined(__AVX512F__) &&
       // defined(__x86_64__) && !defined(USE_TESTPARAM)

#endif // ULMBLAS_CONFIG_TUNED_INTEL_ALDERLAKE_H
