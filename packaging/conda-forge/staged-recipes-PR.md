ulmBLAS implements all 148 routines of the reference BLAS (single, double,
complex) with the Fortran 77 (LP64) and the CBLAS interface.  Output
`libulmblas`.

`libulmblas` is a small front library that selects, when it is loaded, a
backend built for the CPU (all in `lib/`, or `Library/bin` on Windows):

- osx-arm64: `sme` (SME2 with FP64, Apple M4 and later), `neon`
- linux-64, win-64: `avx512`, `avx2` (AVX2 + FMA), `sse3`
- linux-aarch64: one library (NEON)

`ULMBLAS_BACKEND` overrides the choice, `ulmblas_backend()` reports it.
osx-64 is skipped (not supported upstream).  On Windows it is built with
the MinGW-w64 compilers (`compiler('m2w64_c')`); the library exports a C
interface only and depends only on system DLLs; an MSVC import library
`ulmblas.lib` is installed as well.  The header is installed as
`include/ulmblas/cblas.h` so that it does not clash with `blas-devel`.

No LAPACK in this package.  The plan is a later PR to blas-feedstock that
adds `ulmblas` as `blas_impl`, handled like `blis` (Netlib `liblapack` on
top).  This recipe is built with rattler-build and conda-forge's pinning
in the upstream CI on all four platforms; on linux-64, linux-aarch64 and
osx-arm64 conda-forge's NumPy runs its test suite on it.  Locally
(osx-arm64) the blas-feedstock with the `ulmblas` variant passes its tests
(107/107), and NumPy and SciPy pass their test suites on it.

Checklist

- [x] Title of this PR is meaningful: e.g. "Adding my_nifty_package", not "updated recipe".
- [x] Recipe uses the v1 `recipe.yaml` format, or this PR explains the exceptional requirement for deprecated v0.
- [x] License file is packaged (see the [v1 example recipe](https://github.com/conda-forge/staged-recipes/blob/main/recipes/example-v1/recipe.yaml) for an example).
- [x] Source is from official source.
- [x] Package does not vendor other packages. (If a package uses the source of another package, they should be separate packages or the licenses of all packages need to be packaged). The license file packages the licenses of the BLIS and OpenBLAS micro kernels that are part of the sources.
- [x] If static libraries are linked in, the license of the static library is packaged.
- [x] Package does not ship static libraries. If static libraries are needed, [follow CFEP-18](https://github.com/conda-forge/cfep/blob/main/cfep-18.md).
- [x] Build number is 0.
- [x] A tarball (`url`) rather than a repo (e.g. `git_url`) is used in your recipe (see [here](https://conda-forge.org/docs/maintainer/adding_pkgs.html#build-from-tarballs-not-repos) for more details).
- [ ] GitHub users listed in the maintainer section have posted a comment confirming they are willing to be listed there.
- [x] When in trouble, please check our [knowledge base documentation](https://conda-forge.org/docs/maintainer/knowledge_base.html) before pinging a team.
