/*
 *  The dispatching library libulmblas.dylib (macOS on Apple Silicon): at
 *  load time it selects a backend, libulmblas_sme.dylib if the CPU has the
 *  SME2 features the SME kernels require (FEAT_SME2, FEAT_SME_F64F64:
 *  Apple M4 and later), otherwise libulmblas_neon.dylib (every Apple
 *  Silicon CPU).  Both backends are the shared library of ../shared, built
 *  with different ARCHFLAGS, in the directory of this library.
 *
 *  Each exported function of this library is a trampoline (trampolines.S,
 *  generated): it loads the address of the backend's function from
 *  ulmblas_dispatch_table and branches to it, all arguments unchanged.
 *  The table is filled here, before any other code can call the library
 *  (the initializers of a library run before those of the libraries and
 *  the program that depend on it).
 *
 *  The error handlers xerbla_ and cblas_xerbla are not forwarded: if the
 *  program defines its own, it is passed to the backend (see below);
 *  otherwise the backend uses its own.
 *
 *  ULMBLAS_BACKEND=neon or ULMBLAS_BACKEND=sme overrides the selection
 *  (sme on a CPU without SME2 is an error).  Any failure (backend not
 *  found, a function missing) terminates the program with a message: a
 *  BLAS that silently does nothing would be worse.
 */
#include <dlfcn.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

#include "names.h"

/*  filled below, read by the trampolines (not exported) */
__attribute__((
    visibility("hidden"))) void *ulmblas_dispatch_table[ULMBLAS_DISPATCH_COUNT];

static int
has_feature(const char *name)
{
    int value = 0;
    size_t size = sizeof(value);

    if (sysctlbyname(name, &value, &size, NULL, 0) != 0) {
	return 0;
    }
    return value != 0;
}

static void
fail(const char *what, const char *detail)
{
    fprintf(stderr, "ulmBLAS: %s%s%s\n", what, detail ? ": " : "",
            detail ? detail : "");
    abort();
}

__attribute__((constructor)) static void
ulmblas_dispatch_init(void)
{
    const int sme = has_feature("hw.optional.arm.FEAT_SME2") &&
                    has_feature("hw.optional.arm.FEAT_SME_F64F64");
    const char *backend = sme ? "sme" : "neon";
    const char *env = getenv("ULMBLAS_BACKEND");

    if (env && *env) {
	if (strcmp(env, "neon") != 0 && strcmp(env, "sme") != 0) {
	    fail("ULMBLAS_BACKEND must be neon or sme, not", env);
	}
	if (strcmp(env, "sme") == 0 && !sme) {
	    fail("ULMBLAS_BACKEND=sme, but the CPU has no SME2 "
	         "with FEAT_SME_F64F64",
	         NULL);
	}
	backend = env;
    }

    /*  the backend is in the directory of this library */
    Dl_info info;
    if (!dladdr((const void *)&ulmblas_dispatch_init, &info) ||
        !info.dli_fname) {
	fail("cannot determine the path of libulmblas", NULL);
    }
    char self[PATH_MAX], path[PATH_MAX];
    strncpy(self, info.dli_fname, sizeof(self) - 1);
    self[sizeof(self) - 1] = 0;
    snprintf(path, sizeof(path), "%s/libulmblas_%s.dylib", dirname(self),
             backend);

    /*  Error handlers defined by the program (or a library loaded before
     *  this one), searched before the backend is loaded; this library
     *  does not export the handlers, so it does not find itself.  The
     *  backend calls them instead of its own (weak definitions of a
     *  library loaded with dlopen are not replaced by the program's).  */
    void *f77_xerbla = dlsym(RTLD_DEFAULT, "xerbla_");
    void *cblas_xerbla = dlsym(RTLD_DEFAULT, "cblas_xerbla");

    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
	fail("cannot load the backend", dlerror());
    }
    if (f77_xerbla) {
	void (*set)(void *) =
	    (void (*)(void *))dlsym(handle, "ulmblas_set_f77_xerbla");
	if (!set) {
	    fail("function missing in the backend", "ulmblas_set_f77_xerbla");
	}
	set(f77_xerbla);
    }
    if (cblas_xerbla) {
	void (*set)(void *) =
	    (void (*)(void *))dlsym(handle, "ulmblas_set_cblas_xerbla");
	if (!set) {
	    fail("function missing in the backend", "ulmblas_set_cblas_xerbla");
	}
	set(cblas_xerbla);
    }
    for (int i = 0; i < ULMBLAS_DISPATCH_COUNT; ++i) {
	void *f = dlsym(handle, ulmblas_dispatch_names[i]);
	if (!f) {
	    fail("function missing in the backend", ulmblas_dispatch_names[i]);
	}
	ulmblas_dispatch_table[i] = f;
    }
}

/*  for tests and diagnostics: the selected backend, "neon" or "sme" */
const char *
ulmblas_backend(void)
{
    Dl_info info;
    if (dladdr(ulmblas_dispatch_table[0], &info) && info.dli_fname) {
	return strstr(info.dli_fname, "libulmblas_sme") ? "sme" : "neon";
    }
    return "unknown";
}
