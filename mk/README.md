# The build of ulmBLAS

The whole build is one GNU (GNU's Not Unix) make (3.81 or newer) that
reads `GNUmakefile`; no make calls another.  The same sources are compiled
several times with other flags (the test configurations, the shared library
and its interfaces with 64-bit integers, the backends of the dispatching
library), each build in its own object directory.  The rules of these
builds are generated with `$(eval $(call ...))` from variables that describe
them, so make sees one dependency graph: `make -j` runs everything in
parallel, and nothing is rebuilt that is up to date.

| File | Contents |
| --- | --- |
| `make.inc` | compilers, flags and their defaults, helper functions |
| `mk/tools.mk` | checks of the tools (compilers, archiver) before a build |
| `mk/lib.mk` | the libraries of the interfaces: `cblas_lib`, `f77blas_lib`, `clapack_lib`, `f77lapack_lib` |
| `mk/shared.mk` | a shared library (`shared_lib`): `libulmblas`, `libulmblas64`, a backend |
| `mk/dispatch.mk` | the dispatching library and its backends (`dispatch_lib`) |
| `mk/lbt.mk` | `libulmblas_lbt` for Julia |
| `mk/install.mk` | `build-shared`, `install-shared`, `check-package` |
| `mk/test.mk` | the tests (`make check` and the test configurations of `test/config/`) |
| `mk/bench.mk` | `check-large` and the ATLAS timing programs (`make bench`) |

## A build

A build `N` is a set of variables `N.*` and a call that makes its rules,
e.g. the C interface in the top directory (`GNUmakefile`):

```make
top.cblas.out   := .
top.cblas.obj   := interfaces/blas/C/build.cblas
top.cblas.asm   := yes
top.cblas.tools := tool-ar
$(eval $(call cblas_lib,top.cblas,cblas))
```

The call defines `top.cblas.lib` (`./libcblas.a`), `top.cblas.header`
(`./cblas.h`), `top.cblas.all` and the rules of the objects.  The settings
of a build (`N.index`, `N.arch`, `N.config`, `N.cxxflags`, `N.fflags`) are
listed at the top of `mk/lib.mk`; every compile command is recorded in
`N.obj/compile.cmd`, so a changed command rebuilds the objects of that
build only.

A shared library is made of such builds (`mk/shared.mk`: the Fortran 77,
C and LAPACK interfaces with 32-bit integers and, with `ILP64=1`, those with
64-bit integers), and the dispatching library of shared libraries
(`mk/dispatch.mk`: one per backend, with its `ARCHFLAGS`).

## Outside of make

Scripts are only run for what is not a build: probing tools
(`mk/probe.sh`), running and evaluating tests (`test/run-tests.sh`,
`test/check-shared.sh`, `packaging/check.sh`), generating the trampolines
of the dispatching library (`interfaces/blas/dispatch/gen.sh`).
Reference-LAPACK (the reference implementation of LAPACK, Linear Algebra
PACKage; `libulmlapack`, `interfaces/lapack/reference/build.sh`) is built
with its own makefiles.  `interfaces/install-shared.sh` only runs `make
install-shared` (or `build-shared`) for the recipes of packages that call
it.
