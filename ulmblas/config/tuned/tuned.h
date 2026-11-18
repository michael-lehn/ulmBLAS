//
//  Index of the parameters measured by make tune per platform: generated
//  by make tune (tune/params/params.py), do not edit.
//
//  tunedForPlatform(platform, p) sets the measured parameters of the
//  platform in p (see ulmblas/config/tuned.h); the others keep their
//  defaults.
//
#ifndef ULMBLAS_CONFIG_TUNED_TUNED_H
#define ULMBLAS_CONFIG_TUNED_TUNED_H 1

#include <ulmblas/config/platform.h>
#include <ulmblas/config/tuned/amd-zen4.h>
#include <ulmblas/config/tuned/apple-m5.h>
#include <ulmblas/config/tuned/intel-alderlake.h>
#include <ulmblas/config/tuned/intel-icelake.h>

namespace ulmBLAS {

struct Tuned;

inline void
tunedForPlatform(Platform platform, Tuned &p)
{
    (void)platform;
    (void)p;
#if defined(ULMBLAS_TUNED_AMD_ZEN4)
    if (platform == platformFromName("amd-zen4")) {
	tunedparam::amd_zen4(p);
    }
#endif
#if defined(ULMBLAS_TUNED_APPLE_M5)
    if (platform == platformFromName("apple-m5")) {
	tunedparam::apple_m5(p);
    }
#endif
#if defined(ULMBLAS_TUNED_INTEL_ALDERLAKE)
    if (platform == platformFromName("intel-alderlake")) {
	tunedparam::intel_alderlake(p);
    }
#endif
#if defined(ULMBLAS_TUNED_INTEL_ICELAKE)
    if (platform == platformFromName("intel-icelake")) {
	tunedparam::intel_icelake(p);
    }
#endif
}

} // namespace ulmBLAS

#endif // ULMBLAS_CONFIG_TUNED_TUNED_H
