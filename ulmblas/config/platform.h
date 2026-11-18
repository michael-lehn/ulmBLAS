#ifndef ULMBLAS_CONFIG_PLATFORM_H
#define ULMBLAS_CONFIG_PLATFORM_H 1

#include <cstdlib>
#include <cstring>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if (defined(__x86_64__) || defined(__i386__)) &&                              \
    (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

//
//  The platform (processor) that ulmBLAS runs on, detected once at run
//  time.  It selects the tables of the run-time selection of the
//  implementations (e.g. ulmblas/level3/gemm/select.h): a table measured
//  on a platform (make tune) is used on that platform, other processors
//  get the default table of the build.
//
//      Platform::Generic        not one of the following
//      Platform::AppleM4        Apple M4 (macOS: brand string "Apple M4...")
//      Platform::AppleM5        Apple M5
//      Platform::IntelIceLake   Intel Ice Lake server (family 6, model
//                               0x6a, 0x6c)
//      Platform::AmdZen4        AMD Zen 4 (family 0x19, models 0x10 to
//                               0x1f, 0x60 to 0x7f, 0xa0 to 0xaf)
//      Platform::IntelAlderLake Intel Alder Lake and Raptor Lake desktop
//                               (family 6, models 0x97, 0x9a, 0xb7, 0xba,
//                               0xbf; AVX2, no AVX-512)
//
//  The environment variable ULMBLAS_PLATFORM (generic, apple-m4, apple-m5,
//  intel-icelake, amd-zen4, intel-alderlake) overrides the detection, e.g. to
//  use or test the table of another platform.
//
namespace ulmBLAS {

enum class Platform
{
    Generic,
    AppleM4,
    AppleM5,
    IntelIceLake,
    AmdZen4,
    IntelAlderLake
};

inline const char *
platformName(Platform platform)
{
    switch (platform) {
    case Platform::AppleM4:
	return "apple-m4";
    case Platform::AppleM5:
	return "apple-m5";
    case Platform::IntelIceLake:
	return "intel-icelake";
    case Platform::AmdZen4:
	return "amd-zen4";
    case Platform::IntelAlderLake:
	return "intel-alderlake";
    default:
	return "generic";
    }
}

//
//  The platform with the name, Platform::Generic for unknown names.
//
inline Platform
platformFromName(const char *name)
{
    const Platform platforms[] = {Platform::AppleM4, Platform::AppleM5,
                                  Platform::IntelIceLake, Platform::AmdZen4,
                                  Platform::IntelAlderLake};
    for (Platform platform : platforms) {
	if (std::strcmp(name, platformName(platform)) == 0) {
	    return platform;
	}
    }
    return Platform::Generic;
}

//
//  Detection from the processor (without ULMBLAS_PLATFORM).
//
inline Platform
detectPlatform()
{
#if defined(__APPLE__)
    char brand[128] = {0};
    std::size_t size = sizeof(brand) - 1;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) ==
        0) {
	if (std::strncmp(brand, "Apple M4", 8) == 0) {
	    return Platform::AppleM4;
	}
	if (std::strncmp(brand, "Apple M5", 8) == 0) {
	    return Platform::AppleM5;
	}
    }
#elif (defined(__x86_64__) || defined(__i386__)) &&                            \
    (defined(__GNUC__) || defined(__clang__))
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
	char vendor[13];
	std::memcpy(vendor, &ebx, 4);
	std::memcpy(vendor + 4, &edx, 4);
	std::memcpy(vendor + 8, &ecx, 4);
	vendor[12] = 0;

	__get_cpuid(1, &eax, &ebx, &ecx, &edx);
	unsigned family = (eax >> 8) & 0xf;
	unsigned model = (eax >> 4) & 0xf;
	if (family == 0xf) {
	    family += (eax >> 20) & 0xff;
	}
	if (family == 0x6 || family >= 0xf) {
	    model += ((eax >> 16) & 0xf) << 4;
	}
	if (std::strcmp(vendor, "GenuineIntel") == 0 && family == 0x6 &&
	    (model == 0x6a || model == 0x6c)) {
	    return Platform::IntelIceLake;
	}
	if (std::strcmp(vendor, "GenuineIntel") == 0 && family == 0x6 &&
	    (model == 0x97 || model == 0x9a || model == 0xb7 || model == 0xba ||
	     model == 0xbf)) {
	    return Platform::IntelAlderLake;
	}
	if (std::strcmp(vendor, "AuthenticAMD") == 0 && family == 0x19 &&
	    ((model >= 0x10 && model <= 0x1f) ||
	     (model >= 0x60 && model <= 0x7f) ||
	     (model >= 0xa0 && model <= 0xaf))) {
	    return Platform::AmdZen4;
	}
    }
#endif
    return Platform::Generic;
}

//
//  The platform: ULMBLAS_PLATFORM if set, otherwise detected.  Determined
//  at the first call (thread-safe).
//
inline Platform
platform()
{
    static const Platform platform_ = []() {
	const char *name = std::getenv("ULMBLAS_PLATFORM");
	return name ? platformFromName(name) : detectPlatform();
    }();
    return platform_;
}

} // namespace ulmBLAS

#endif // ULMBLAS_CONFIG_PLATFORM_H
