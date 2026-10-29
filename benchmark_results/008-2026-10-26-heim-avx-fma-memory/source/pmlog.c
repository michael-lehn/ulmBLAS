#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stddef.h>
/* logs posix_memalign calls of more than 64 KiB (diagnosis only) */
int posix_memalign(void **p, size_t a, size_t n)
{
    static int (*real)(void **, size_t, size_t);
    if (!real) real = dlsym(RTLD_NEXT, "posix_memalign");
    int r = real(p, a, n);
    if (n > 65536) fprintf(stderr, "posix_memalign %zu -> %p\n", n, *p);
    return r;
}
