# Releasing

Notes for the maintainer.  Versions `X.Y.Z`, tags `vX.Y.Z` on master.
Policy: 0.4.1, 0.4.2, ... until on par with OpenBLAS (BLAS (Basic Linear
Algebra Subprograms) and the LAPACK (Linear Algebra PACKage) subset).

## Version source

- The tag is the only place the version is written by hand.
- `VERSION.txt` holds `$Format:%(describe:tags=true,abbrev=0)$` and has
  `export-subst` in `.gitattributes`: `git archive` (also GitHub's archive
  of a tag) writes the tag into it, e.g. `v0.4.1`.  It is never edited.
- `interfaces/install-shared.sh` takes the version (pkg-config files) from
  `ULMBLAS_VERSION`, else `git describe --tags --abbrev=0`, else
  `VERSION.txt`, else the directory name `ulmBLAS-X.Y.Z`.
- Per release, before the tag: `CITATION.cff` (`version`,
  `date-released`: `packaging/set-version.sh --citation X.Y.Z
  [YYYY-MM-DD]`) and `CHANGELOG.md` (by hand); after the tag the recipes
  get the version and checksums from `packaging/set-version.sh X.Y.Z
  TARBALL`.

## Procedure

1. On master, clean tree, CI (continuous integration) green, `make check`
   on Apple Silicon and on x86-64 Linux.
2. `CHANGELOG.md`: rename "X.Y.Z (unreleased)" to "X.Y.Z (YYYY-MM-DD)";
   `CITATION.cff`: `sh packaging/set-version.sh --citation X.Y.Z` (date:
   today, or as a second argument).  Commit
   ("Release X.Y.Z").
3. License check: `git archive HEAD | tar -t` against
   `packaging/common/copyright` and `NOTICE` (new vendored files? new
   export-ignore rules?).
4. Tag (signed): `git tag -s vX.Y.Z -m "ulmBLAS X.Y.Z"`.
5. Tarball, checksum, signature, with GNU (GNU's Not Unix) gzip (Linux;
   macOS: `brew install gzip`, Apple's gzip gives a different file), on
   the machine with the signing key:

       tools/release/make-tarball.sh -o dist vX.Y.Z

   (Key elsewhere: `-k none`, copy `dist/` to the machine with the key,
   `sha256sum -c SHA256SUMS`, then `gpg --armor --detach-sign
   --local-user 5BD0789A912F5874A1366392D8B6E130B9F67007
   ulmBLAS-X.Y.Z.tar.gz`.)

   gives `dist/ulmBLAS-X.Y.Z.tar.gz`, `dist/SHA256SUMS`,
   `dist/ulmBLAS-X.Y.Z.tar.gz.asc`.  Nothing is tagged, pushed or
   uploaded by the script.  Run it twice (or in a second clone): the
   SHA256 (Secure Hash Algorithm, 256 bits) must be the same.
6. Push master and the tag: `git push origin master vX.Y.Z`.
7. Compare with GitHub's archive (the recipes download it):

       curl -sL https://github.com/michael-lehn/ulmBLAS/archive/refs/tags/vX.Y.Z.tar.gz | sha256sum

   must equal `dist/SHA256SUMS` (it did for v0.4.0: `git archive | gzip
   -n` with GNU gzip 1.12 is byte-identical).  If not, the tarball of step
   5 is the reference: upload it and point the recipes at the release
   asset instead of `archive/refs/tags/`.
8. GitHub release `vX.Y.Z`: notes from `CHANGELOG.md`; assets
   `ulmBLAS-X.Y.Z.tar.gz`, `SHA256SUMS`, `ulmBLAS-X.Y.Z.tar.gz.asc`
   (`gh release create vX.Y.Z dist/* --notes-file ...`).
9. Recipes: `packaging/set-version.sh X.Y.Z dist/ulmBLAS-X.Y.Z.tar.gz`
   (version and checksums into every recipe), then the packaging dry run
   (`.github/workflows/packaging-dryrun.yml`) and `packaging/check.sh`.
10. Per channel (see `packaging/README.md`, "Publishing"):
    - conda-forge: first release via staged-recipes, later the feedstock
      bot; blas-feedstock patches.
    - Homebrew: tap `michael-lehn/homebrew-ulmblas`; homebrew-core
      formula.
    - Debian/Ubuntu: `packaging/debian` as `debian/`, `debian/copyright` =
      `packaging/common/copyright`, `debian/changelog` entry
      (`X.Y.Z-1`); `uscan` (`debian/watch`) checks the signature once
      `debian/upstream/signing-key.asc` exists (below).
    - Fedora, openSUSE: spec files (`Source0` = GitHub archive).
    - Arch (AUR: Arch User Repository), Alpine (aports), MSYS2 (Minimal
      SYStem 2), nixpkgs,
      Spack, EasyBuild: the recipe files with the new checksums.

## ABI (application binary interface) and soname

Libraries and their names:

| File | Name the loader looks for | Contents |
|---|---|---|
| `libulmblas.so.0` (Linux) | soname `libulmblas.so.0` | front: Fortran 77 interface (LP64 (long and pointer 64-bit): 32-bit integers) and CBLAS (C interface to the BLAS), forwards to a backend |
| `libulmblas.0.dylib` (macOS) | install name `@rpath/libulmblas.0.dylib` | same |
| `libulmblas.dll` (Windows) | `libulmblas.dll` | same |
| `libulmblas_{sse3,avx2,avx512,generic}.so`, `libulmblas_{sme,neon}.dylib` | loaded with `dlopen` by the front | backends, no version |
| `libblas.so.3`, `libcblas.so.3` (Linux x86-64, `ULMBLAS_SONAMES=yes`) | sonames of the distributions' BLAS | the front once more, for `update-alternatives` |
| `libulmlapack.so` | no version | Reference-LAPACK on top of `libulmblas` (not in the distribution packages) |

Rules:

- The public ABI is the list of exported functions of the front (version
  script `interfaces/blas/dispatch/build.dispatch/*/exports.txt`, from the
  backends' common exports) with the signatures of `cblas.h` and the
  Fortran 77 conventions (gfortran return values for complex functions,
  hidden string lengths).
- The major number `N` of `libulmblas.so.N` changes only when a program
  linked against the old library could break: an exported function is
  removed or renamed, a signature or the meaning of an argument changes,
  the values of the CBLAS enums change, the default integer type
  (`ULMBLAS_INDEX_TYPE=int`, LP64) changes.  New functions do not change
  it (check: `abidiff` or the sorted `nm -D --defined-only` of the old and
  new library; nothing may disappear).
- If `N` changes: new package names (Debian `libulmblasN`, openSUSE
  `libulmblasN`), `SHARED_VERSION` and the soname in
  `interfaces/blas/dispatch/Makefile` and `interfaces/install-shared.sh`.
  A 64-bit integer (ILP64: integer, long and pointer 64-bit) build would get its own name
  (`libulmblas64`), not a new `N`.
- `libblas.so.3`/`libcblas.so.3` keep the soname of the reference BLAS;
  they implement its interface, which does not change.
- The backends are private to the front of the same build: a package
  installs front and backends together; the interface between them is not
  stable (the front checks that all backends export the same functions).
- The C++ template interface (headers in `ulmblas/`) has no ABI; it is
  compiled into the program.
- State 2026-11-17 (Linux x86-64, `nm -D --defined-only`): the front
  `libulmblas.so.0` exports 505 functions, all `cblas_*`, `ulmblas_*` or
  Fortran names with `_`; no C++ symbols (version script, `local: *`),
  no symbol versions; `xerbla_` and `cblas_xerbla` are weak undefined
  references (a program may define them).  The backends export the same
  functions plus `xerbla_`, `cblas_xerbla` (weak) and the two variables
  of the CBLAS tests (`RowMajorStrg`, `CBLAS_CallFromC`).  macOS: 505
  (front) and 511 (backends), no C++ symbols.
- Before a release: compare the sorted exports with those of the last
  release (`nm -D --defined-only libulmblas.so.0 | awk '{print $3}' |
  sort`); removals mean a new `N`.

## GPG (GNU Privacy Guard) signing key

Key: ed25519, signing only, "Michael Lehn <michael.lehn@uni-ulm.de>",
fingerprint `5BD0789A912F5874A1366392D8B6E130B9F67007`, expires
2028-11-16; on keys.openpgp.org (findable by e-mail).  Renew before
2028-11-16 with
`gpg --quick-set-expire 5BD0789A912F5874A1366392D8B6E130B9F67007 2y`
and upload the key again.

- Tags: `git config user.signingkey
  5BD0789A912F5874A1366392D8B6E130B9F67007`, then `git tag -s`.
- Tarball: `make-tarball.sh` signs with this key by default (if its
  secret key is in the keyring; `-k KEY` or `ULMBLAS_GPG_KEY` for another
  key, `-k none` for no signature).
- Open (maintainer):
  - Add the public key to the GitHub account (Settings, SSH (Secure
    Shell) and GPG keys), so signed tags show as verified.
  - Attach `gpg --armor --export 5BD0789A912F5874A1366392D8B6E130B9F67007`
    as `signing-key.asc` to the GitHub releases; fingerprint in the
    release notes and in `README.md` (verification of downloads).
  - Debian: the armored public key as
    `packaging/debian/upstream/signing-key.asc` (for `uscan`).  GitHub's
    tag archives have no `.asc`, so `debian/watch` has to point at the
    release assets (`https://github.com/michael-lehn/ulmBLAS/releases`,
    pattern `.*/ulmBLAS-(\d[\d.]*)\.tar\.gz`, option
    `pgpsigurlmangle=s%$%.asc%`); the asset is the same file as the tag
    archive (step 7).
  - Optional: Fedora `%gpgverify` with `Source2:
    gpgkey-5BD0789A912F5874A1366392D8B6E130B9F67007.gpg`; Arch
    `validpgpkeys=('5BD0789A912F5874A1366392D8B6E130B9F67007')`.

## Reproducible builds

- Tarball: `make-tarball.sh` (see the script).
- Libraries: two builds of the same tarball in different directories
  (`interfaces/install-shared.sh`, default flags, no `-g`) give
  byte-identical libraries with the same compiler: checked 2026-11-17 on
  Linux x86-64 (GCC (GNU Compiler Collection) 12, Debian 12) and on macOS
  (Apple clang, Apple Silicon).  No build path, time stamp or host name is in the
  libraries, so `SOURCE_DATE_EPOCH` is not needed.  With `-g` (debug
  information, e.g. the default flags of Debian and Fedora) the build
  directory is recorded; the distributions add `-ffile-prefix-map`
  themselves.
