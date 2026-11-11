# Packaging

| Directory | For |
| --- | --- |
| [conda-forge/ulmblas](conda-forge/ulmblas) | conda-forge package `libulmblas` (recipe for staged-recipes, later the feedstock `ulmblas-feedstock`) |
| [conda-forge/blas-feedstock](conda-forge/blas-feedstock) | patches for [conda-forge/blas-feedstock](https://github.com/conda-forge/blas-feedstock) (on top of `BASE_COMMIT`): the variant `ulmblas` of `libblas`/`libcblas` (`conda install numpy "libblas=*=*ulmblas"`), build number 13, and the re-render (conda-smithy) that adds the CI jobs of the variant |
| [homebrew](homebrew) | formula for the tap `michael-lehn/homebrew-ulmblas` |

All of them build with `interfaces/install-shared.sh` (`make
install-shared`), see [INSTALL.md](../INSTALL.md#shared-libraries-for-numpy-and-other-programs).

## Platforms

| conda-forge | Library | Backends | Built with |
| --- | --- | --- | --- |
| linux-64 | `libulmblas.so.0` | `avx512`, `avx2`, `sse3` | GCC |
| linux-aarch64 | `libulmblas.so.0` | (one library, NEON) | GCC |
| osx-arm64 | `libulmblas.0.dylib` | `sme`, `neon` | clang |
| win-64 | `libulmblas.dll`, `ulmblas.lib` | `avx512`, `avx2`, `sse3` | MinGW-w64 GCC (UCRT), `compiler('m2w64_c')` |

No Intel Macs.  No LAPACK in the conda package: conda-forge's Netlib
`liblapack` runs on top of `libblas`, as for BLIS.

## Local test builds

conda package (rattler-build, conda-forge's pinning from the package
`conda-forge-pinning`, file renamed to `conda_build_config.yaml`, no `-c`
option since the pinning sets the channels):

```sh
cp packaging/conda-forge/ulmblas/* recipe/
# in recipe/recipe.yaml: source: {path: /path/to/ulmBLAS, use_gitignore: true}
rattler-build build --recipe recipe/recipe.yaml \
    --variant-config pin/conda_build_config.yaml --output-dir out
```

blas-feedstock with the patch (conda-build; its build script calls conda,
so `CONDARC` must name a file with `channels: [conda-forge]`):

```sh
git clone https://github.com/conda-forge/blas-feedstock && cd blas-feedstock
git am .../packaging/conda-forge/blas-feedstock/000*.patch
conda-build recipe -m .../conda_build_config.yaml -c file://.../out -c conda-forge \
    --override-channels --output-folder cb-out \
    --variants "{'blas_impl': ['ulmblas'], 'blas_impl_lib': ['libulmblas.0.dylib'],
                 'lapack_impl_lib': ['notapplicable'], 'MACOSX_SDK_VERSION': ['11.0']}"
micromamba create -p env -c file://cb-out -c file://out -c conda-forge \
    numpy scipy "libblas=*=*ulmblas"
```

Homebrew: a local tap (`brew tap-new --no-git local/test`), the formula
with `url "file://..."` of a `git archive` tarball, then `brew install
--build-from-source`, `brew test`, `brew audit --strict --formula`.

CI: [.github/workflows/conda.yml](../.github/workflows/conda.yml) (the
conda package on all four platforms, conda-forge NumPy on top on Linux
and macOS), [.github/workflows/windows-shared.yml](../.github/workflows/windows-shared.yml)
(the front and the backends on Windows, Netlib tests).

## Results (2026-11-10)

- osx-arm64 (M5): package builds and passes its test (backend `sme`);
  blas-feedstock with the patch: 107/107 tests (LAPACK test suite, Netlib
  BLAS, LAPACKE examples); `numpy scipy "libblas=*=*ulmblas"` from the
  local channels: NumPy 2.5.4 test suite passes (46739), `scipy.linalg`
  11143 passed, `a @ b` n = 1000: 370 GFLOP/s.
- Linux x86-64 (heim, Alder Lake, no AVX-512): front selects `avx2`;
  Netlib tests 25/25 per backend, 17/17 through the front.  Debian's
  NumPy 1.24 and SciPy 1.10 on ulmBLAS through `libblas.so.3`:
  `numpy.linalg`, `numpy.core.tests.test_multiarray`, `scipy.linalg`: 5352
  passed, 1 failed (below).
- Windows: cross-built with MinGW-w64 GCC 16 on macOS (exports, imports:
  only system DLLs); not yet run (CI).
- AVX-512 (bwUniCluster, AMD EPYC 9454, RHEL 9, GCC 11): front selects
  `avx512`; Netlib tests 25/25 per backend, 17/17 through the front.
  Without `libstdc++.a` (RHEL) the C++ runtime is linked dynamically.

## Known issue

`scipy.linalg.tests.test_decomp::test_orth_memory_efficiency` (slow test)
fails on x86-64 with the `sse3` and `avx2` backends: `orth` of a 10^7 x 2
matrix of ones, element 0 is off by 4.3e-11, allowed are 3.2e-11.  The
second Householder vector of the QR is made of rounding noise, so the
result depends on how the BLAS rounds; the reference BLAS is off by
2.8e-11, OpenBLAS by 4e-14.  It passes on Apple Silicon (`sme`, `neon`).
Not an error of ulmBLAS, but worth a look (rank-1 update, `dger`).

## Publishing (per release)

1. Tag `vX.Y.Z` on master, GitHub release.
2. sha256 of the release tarball:
   `curl -sL https://github.com/michael-lehn/ulmBLAS/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256`
   into `conda-forge/ulmblas/recipe.yaml` and `homebrew/ulmblas.rb`.
3. First release: PR to [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
   with `recipes/ulmblas/` = the files of `conda-forge/ulmblas`
   (text: [conda-forge/staged-recipes-PR.md](conda-forge/staged-recipes-PR.md)).
   Later releases: the bot of the feedstock opens the update PR.
4. When `libulmblas` is on conda-forge: PR to conda-forge/blas-feedstock
   with the two patches, from a fork (`git am`; if the feedstock has moved
   on, rebase and re-render again: `conda-smithy rerender`); text:
   [conda-forge/blas-feedstock-PR.md](conda-forge/blas-feedstock-PR.md).
5. Homebrew: repository `michael-lehn/homebrew-ulmblas` with
   `Formula/ulmblas.rb`.
