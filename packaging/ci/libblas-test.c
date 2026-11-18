/*
 *  libblas-test.c: libblas.so.3 of the system (alternatives of Debian and
 *  openSUSE, provider packages of Arch) must be ulmBLAS; if
 *  liblapack.so.3 exists, dgesv through it must run on ulmBLAS too.  Part
 *  of the packaging dry run (test-installed.sh).
 */
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef const char *(*BackendFn)(void);
typedef void (*DgesvFn)(const int *n, const int *nrhs, double *A,
                        const int *ldA, int *ipiv, double *B, const int *ldB,
                        int *info);

static void
show_maps(void)
{
    FILE *maps = fopen("/proc/self/maps", "r");
    char line[4096];
    if (!maps) {
	return;
    }
    while (fgets(line, sizeof(line), maps)) {
	if (strstr(line, "ulmblas") || strstr(line, "blas.so") ||
	    strstr(line, "lapack.so")) {
	    const char *path = strchr(line, '/');
	    if (path) {
		printf("    %s", path);
	    }
	}
    }
    fclose(maps);
}

int
main(void)
{
    void *blas = dlopen("libblas.so.3", RTLD_NOW | RTLD_GLOBAL);
    if (!blas) {
	printf("libblas.so.3: %s\n", dlerror());
	return 1;
    }
    BackendFn backend = (BackendFn)dlsym(blas, "ulmblas_backend");
    if (!backend) {
	printf("libblas.so.3 is not ulmBLAS (no ulmblas_backend)\n");
	show_maps();
	return 1;
    }
    printf("libblas.so.3: ulmBLAS, backend %s\n", backend());

    void *lapack = dlopen("liblapack.so.3", RTLD_NOW);
    if (lapack) {
	DgesvFn dgesv = (DgesvFn)dlsym(lapack, "dgesv_");
	double A[9] = {4, 1, 0, 1, 4, 1, 0, 1, 4};
	double B[3] = {5, 6, 5};
	int n = 3, one = 1, ipiv[3], info = -1;
	if (!dgesv) {
	    printf("liblapack.so.3: no dgesv_\n");
	    return 1;
	}
	dgesv(&n, &one, A, &n, ipiv, B, &n, &info);
	printf("liblapack.so.3: dgesv info %d, x = %g %g %g\n", info, B[0],
	       B[1], B[2]);
	if (info != 0 || B[0] < 0.999 || B[0] > 1.001 || B[1] < 0.999 ||
	    B[1] > 1.001) {
	    return 1;
	}
    } else {
	/*  not installed, or not loadable (e.g. a symbol missing in BLAS) */
	const char *error = dlerror();
	printf("liblapack.so.3: %s\n", error ? error : "?");
	if (error && !strstr(error, "No such file")) {
	    return 1;
	}
    }
    show_maps();
    return 0;
}
