//
//  Parameters of ulmBLAS for apple-m5 (ulmblas/config/tuned.h): measured
//  by make tune-params (tune/params/params.py), last on 2026-11-17, do not
//  edit.  The report: tune/results/apple-m5/params.md.
//  Included by ulmblas/config/tuned/tuned.h (after the definition of Tuned).
//
#ifndef ULMBLAS_CONFIG_TUNED_APPLE_M5_H
#define ULMBLAS_CONFIG_TUNED_APPLE_M5_H 1

#if defined(USE_SME) && !defined(USE_TESTPARAM)

#define ULMBLAS_TUNED_APPLE_M5 1

namespace ulmBLAS {
namespace tunedparam {

inline void
apple_m5(Tuned &p)
{
    // clang-format off
    p.gemvNThreads[tunedD] = {1, 12288, 0};
    p.gemvNThreads[tunedS] = {1, 24576, 0};
    p.gemvTThreads[tunedD] = {1, 6144, 3, 12288, 0};
    p.gemvTThreads[tunedS] = {1, 6144, 3, 12288, 0};
    p.gerThreads[tunedD] = {1, 3072, 0};
    p.gerThreads[tunedS] = {1, 6144, 0};
    p.gerThreads[tunedZ] = {0};
    p.trsm.kb[tunedC] = {512, 3000, 256};
    p.trsm.kb[tunedZ] = {192, 3000, 256};
    p.trsmInvert[tunedC] = 1;
    p.trsmInvert[tunedZ] = 1;
    // clang-format on
}

} // namespace tunedparam
} // namespace ulmBLAS

#endif // defined(USE_SME) && !defined(USE_TESTPARAM)

#endif // ULMBLAS_CONFIG_TUNED_APPLE_M5_H
