# Packaging

| Directory | For |
| --- | --- |
| [conda-forge/ulmblas](conda-forge/ulmblas) | conda-forge package `libulmblas` (recipe for staged-recipes, later the feedstock `ulmblas-feedstock`) |
| [conda-forge/blas-feedstock](https://github.com/michael-lehn/ulmBLAS/tree/master/packaging/conda-forge/blas-feedstock) | patches for [conda-forge/blas-feedstock](https://github.com/conda-forge/blas-feedstock) (on top of `BASE_COMMIT`): the variant `ulmblas` of `libblas`/`libcblas` (`conda install numpy "libblas=*=*ulmblas"`), build number 13, and the re-render (conda-smithy) that adds the CI jobs of the variant |
| [homebrew](homebrew) | formula for the tap `michael-lehn/homebrew-ulmblas` |
| [homebrew-core](homebrew-core) | formula for homebrew-core (`ENV.runtime_cpu_detection`, `ENV.O3`, BLAS only) |
| [debian](debian) | `debian/` directory (Debian, Ubuntu): `libulmblas0` (alternative `libblas.so.3` on amd64), `libulmblas-dev`; copy with `cp -rL` (`copyright` is a link to [common/copyright](common/copyright)) |
| [fedora](fedora) | RPM spec for Fedora with `flexiblas-ulmblas` (`/etc/flexiblasrc.d/ulmblas.conf`) |
| [opensuse](opensuse) | RPM spec for openSUSE (`libulmblas0`, alternatives `libblas.so.3`, `libcblas.so.3` on x86-64) |
| [arch](arch) | AUR `PKGBUILD`: `ulmblas` and `blas-ulmblas` (provides `blas`, `cblas`) |
| [alpine](alpine) | `APKBUILD` (aports) |
| [nix](nix) | nixpkgs `package.nix` (provider for the `blas` wrapper) |
| [spack](spack) | Spack `package.py` (`provides("blas")`) |
| [easybuild](easybuild) | EasyBuild easyconfig (GCCcore toolchain) |
| [msys2](msys2) | MSYS2 `PKGBUILD` (`mingw-w64-ulmblas`, ucrt64 and mingw64) |

All recipes take the version from `set-version.sh VERSION TARBALL`, which
puts the version and the checksums of the release tarball into each of
them (per release), and they all build the same way: `ULMBLAS_STEP=build
interfaces/install-shared.sh PREFIX` in the build step,
`packaging/check.sh` (Netlib test programs against the built library) in
the check step and `DESTDIR=... interfaces/install-shared.sh PREFIX` in the
install step.  None of them builds `libulmlapack` (`FC=none`): the
distributions put their reference LAPACK on top of `libblas.so.3` or
FlexiBLAS.

CI dry run: [.github/workflows/packaging-dryrun.yml](https://github.com/michael-lehn/ulmBLAS/blob/master/.github/workflows/packaging-dryrun.yml)
builds, installs and tests every recipe in the container or runner of its
distribution (`gh workflow run packaging-dryrun.yml --ref BRANCH -f
job=NAME`), without publishing anything.

All of them build with `interfaces/install-shared.sh` (`make
install-shared`), see [INSTALL.md](../INSTALL.md#shared-libraries-for-numpy-and-other-programs).

## For distribution packages

`interfaces/install-shared.sh PREFIX` takes these variables from the
environment (see the script):

| Variable | Meaning |
| --- | --- |
| `DESTDIR` | staging directory: files go to `DESTDIR/PREFIX/...`, the paths in the pkg-config files are `PREFIX/...` |
| `ULMBLAS_LIBDIR` | libraries (default `PREFIX/lib`, e.g. `PREFIX/lib/x86_64-linux-gnu`) |
| `ULMBLAS_BINDIR` | DLLs on Windows (default `PREFIX/bin`) |
| `ULMBLAS_INCLUDEDIR` | `cblas.h` (default `PREFIX/include`) |
| `ULMBLAS_RPATH=no` | no `-Wl,-rpath` in the pkg-config files (never on Windows) |
| `ULMBLAS_VERSION` | version in the pkg-config files (default: last release tag, the file `VERSION.txt` of a release tarball, or the directory name `ulmBLAS-X.Y.Z`) |
| `FC=none` | no `libulmlapack` (it downloads Reference-LAPACK) |
| `ULMBLAS_STATIC_LIBSTDCXX=no` | Linux: `libstdc++` linked dynamically (default: statically if `libstdc++.a` exists) |
| `ULMBLAS_BACKENDDIR` | the backends of the dispatching library (default: `LIBDIR`; packages: `LIBDIR/ulmblas`, compiled into the dispatching library) |
| `ULMBLAS_STEP=build` | only build (the build step of a package); the install step runs the script again with the same environment |
| `LAPACK_TARBALL` | Reference-LAPACK tarball for `libulmlapack` (no download) |
| `ULMBLAS_SONAMES=yes` | Linux x86-64: also `libblas.so.3` and `libcblas.so.3` (the dispatching library with these SONAMEs, for the alternatives of Debian and openSUSE and Gentoo's providers) in `ULMBLAS_SONAMEDIR` (default `LIBDIR/ulmblas`); they load the backends from `ULMBLAS_BACKENDDIR` |

The dispatching library (`libulmblas.so.0`, `libulmblas.0.dylib`) loads
its backend `libulmblas_NAME` from the first of these directories where it
can be loaded:

1. the environment variable `ULMBLAS_BACKEND_DIR`;
2. `ULMBLAS_BACKEND_DIR` of the build (`-DULMBLAS_BACKEND_DIR='"..."'`),
   for packages that copy the library away from its backends;
3. the directory of the library after resolving links, e.g. with a link
   `libblas.so.3` of the Debian or openSUSE alternatives in another
   directory;
4. the directory of the file that was loaded.

`ULMBLAS_BACKEND=NAME` selects a backend (e.g. for tests).

## Platforms

| conda-forge | Library | Backends | Built with |
| --- | --- | --- | --- |
| linux-64 | `libulmblas.so.0` | `avx512`, `avx2`, `sse3`, `generic` | GCC |
| linux-aarch64 | `libulmblas.so.0` | (one library, NEON) | GCC |
| osx-arm64 | `libulmblas.0.dylib` | `sme`, `neon` | clang |
| win-64 | `libulmblas.dll`, `ulmblas.lib` | `avx512`, `avx2`, `sse3`, `generic` | MinGW-w64 GCC (UCRT), `compiler('m2w64_c')` |

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

CI: [.github/workflows/conda.yml](https://github.com/michael-lehn/ulmBLAS/blob/master/.github/workflows/conda.yml) (the
conda package on all four platforms, conda-forge NumPy on top on Linux
and macOS), [.github/workflows/windows-shared.yml](https://github.com/michael-lehn/ulmBLAS/blob/master/.github/workflows/windows-shared.yml)
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
2. Version and checksums into all recipes:
   `curl -sLo vX.Y.Z.tar.gz https://github.com/michael-lehn/ulmBLAS/archive/refs/tags/vX.Y.Z.tar.gz`,
   `sh packaging/set-version.sh X.Y.Z vX.Y.Z.tar.gz`; the changelogs of
   `debian/changelog` and the RPM specs by hand.  Before that: the
   packaging dry run on the release commit.
3. First release: PR to [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
   with `recipes/ulmblas/` = the files of `conda-forge/ulmblas`
   (text: [conda-forge/staged-recipes-PR.md](https://github.com/michael-lehn/ulmBLAS/blob/master/packaging/conda-forge/staged-recipes-PR.md)).
   Later releases: the bot of the feedstock opens the update PR.
4. When `libulmblas` is on conda-forge: PR to conda-forge/blas-feedstock
   with the two patches, from a fork (`git am`; if the feedstock has moved
   on, rebase and re-render again: `conda-smithy rerender`); text:
   [conda-forge/blas-feedstock-PR.md](https://github.com/michael-lehn/ulmBLAS/blob/master/packaging/conda-forge/blas-feedstock-PR.md).
5. Homebrew: repository `michael-lehn/homebrew-ulmblas` with
   `Formula/ulmblas.rb`.
6. The other channels (all submissions by the maintainer, from his
   accounts):
   - Debian: ITP (Intent To Package) bug, Debian Science team on Salsa,
     upload to mentors.debian.net, RFS (Request For Sponsorship); Ubuntu
     follows from Debian unstable (or a PPA (Personal Package Archive) on
     Launchpad in the meantime).
   - Fedora: review request in Red Hat Bugzilla, sponsor for the packager
     group; EPEL (Extra Packages for Enterprise Linux) after Fedora.
   - openSUSE: package in an OBS (Open Build Service) home project, then
     a submit request to `science`.
   - Arch: AUR (Arch User Repository) account and SSH key, push of
     `PKGBUILD` and `.SRCINFO` (`makepkg --printsrcinfo`).
   - Alpine: merge request to aports (`testing`).
   - nixpkgs, Spack, EasyBuild, MSYS2: pull requests to
     NixOS/nixpkgs (`pkgs/by-name/ul/ulmblas/package.nix`),
     spack/spack-packages, easybuilders/easybuild-easyconfigs (with a
     test report), msys2/MINGW-packages.
   - homebrew-core: pull request by someone other than the owner of the
     repository (notability rule for self-submissions); the formula needs
     an entry in homebrew-core's allowlist for `ENV.runtime_cpu_detection`.
