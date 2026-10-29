//
//  probe.sh cpu: exit status 0 if the CPU and the operating system support
//  all features given as arguments (sse3, avx, fma, avx2, avx512f, avx512dq on x86,
//  neon and sme on AArch64), 1 if one of them is not supported, 2 for an unknown feature.
//  neon: an AArch64 (LP64) target, which always has AdvSIMD.  sme: the
//  experimental SME2 micro kernel can be used (see ulmblas/config/simd.h:
//  SME2 enabled for clang) and the CPU has SME2 and FEAT_SME_F64F64 with a
//  streaming vector length of 512 bits (checked on macOS).
//
#include <cstring>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SME2) && defined(__clang__)
#   define PROBE_SME
#   if defined(__APPLE__)
#       include <sys/sysctl.h>
#   endif
#endif

#if defined(PROBE_SME)
static bool
sysctlIs(const char *name, long value)
{
#   if defined(__APPLE__)
    long   x = 0;
    size_t n = sizeof(x);
    return sysctlbyname(name, &x, &n, 0, 0)==0 && x==value;
#   else
    (void)name; (void)value;
    return false;
#   endif
}
#endif

int
main(int argc, char **argv)
{
    static const char *known[] = { "sse3", "avx", "fma", "avx2", "avx512f",
                                   "avx512dq", "neon", "sme" };
    bool supported = true;

    for (int i=1; i<argc; ++i) {
        bool isKnown = false;
        for (unsigned k=0; k<sizeof(known)/sizeof(known[0]); ++k) {
            isKnown = isKnown || !std::strcmp(argv[i], known[k]);
        }
        if (!isKnown) {
            return 2;
        }
        if (!std::strcmp(argv[i], "neon")) {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__LP64__)
            continue;
#else
            supported = false;
            continue;
#endif
        }
        if (!std::strcmp(argv[i], "sme")) {
#if defined(PROBE_SME)
            supported = supported
                     && sysctlIs("hw.optional.arm.FEAT_SME2", 1)
                     && sysctlIs("hw.optional.arm.FEAT_SME_F64F64", 1)
                     && sysctlIs("hw.optional.arm.sme_max_svl_b", 64);
#else
            supported = false;
#endif
            continue;
        }
#if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_init();
        supported = supported
                 && (std::strcmp(argv[i], "avx")  || __builtin_cpu_supports("avx"))
                 && (std::strcmp(argv[i], "fma")  || __builtin_cpu_supports("fma"))
                 && (std::strcmp(argv[i], "avx2") || __builtin_cpu_supports("avx2"))
                 && (std::strcmp(argv[i], "sse3") || __builtin_cpu_supports("sse3"))
                 && (std::strcmp(argv[i], "avx512f")
                     || __builtin_cpu_supports("avx512f"))
                 && (std::strcmp(argv[i], "avx512dq")
                     || __builtin_cpu_supports("avx512dq"));
#else
        supported = false;
#endif
    }
    return supported ? 0 : 1;
}
