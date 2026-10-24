//
//  probe.sh cpu: exit status 0 if the CPU and the operating system support
//  all features given as arguments (avx, fma, avx2), 1 if one of them is
//  not supported, 2 for an unknown feature.
//
#include <cstring>

int
main(int argc, char **argv)
{
    static const char *known[] = { "avx", "fma", "avx2" };
    bool supported = true;

    for (int i=1; i<argc; ++i) {
        bool isKnown = false;
        for (unsigned k=0; k<sizeof(known)/sizeof(known[0]); ++k) {
            isKnown = isKnown || !std::strcmp(argv[i], known[k]);
        }
        if (!isKnown) {
            return 2;
        }
#if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_init();
        supported = supported
                 && (std::strcmp(argv[i], "avx")  || __builtin_cpu_supports("avx"))
                 && (std::strcmp(argv[i], "fma")  || __builtin_cpu_supports("fma"))
                 && (std::strcmp(argv[i], "avx2") || __builtin_cpu_supports("avx2"));
#else
        supported = false;
#endif
    }
    return supported ? 0 : 1;
}
