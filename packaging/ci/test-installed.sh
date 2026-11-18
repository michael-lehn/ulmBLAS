#! /bin/sh
#
#  test-installed.sh [libblas]: tests an installed ulmBLAS package (the
#  packaging dry run, .github/workflows/packaging-dryrun.yml), run in the
#  unpacked source tree:
#
#  1. a CBLAS program through pkg-config, with every backend the CPU
#     supports (x86-64);
#  2. the Netlib test programs of the Fortran 77 interface against the
#     installed library;
#  3. with "libblas": libblas.so.3 of the system (alternatives, provider
#     packages) must be ulmBLAS; LAPACK through liblapack.so.3 if there
#     is one (libblas-test.c).
#
#  Needs cc, gfortran and pkg-config.
#
set -eu
PC=${PKG_CONFIG:-pkg-config}
TMP=$(mktemp -d)
LIBDIR=$($PC --variable=libdir ulmblas)
INCDIR=$($PC --variable=includedir ulmblas)
echo "== pkg-config: ulmblas $($PC --modversion ulmblas): $($PC --cflags --libs ulmblas)"
cc packaging/conda-forge/ulmblas/test_ulmblas.c $($PC --cflags --libs ulmblas) \
    -lm -o "$TMP/test_ulmblas"
"$TMP/test_ulmblas"
BACKENDS=
if [ "$(uname -m)" = x86_64 ]; then
    flags=$(grep -m1 '^flags' /proc/cpuinfo)
    BACKENDS="generic sse3"
    echo "$flags" | grep -qw avx2 && BACKENDS="$BACKENDS avx2"
    echo "$flags" | grep -qw avx512f && BACKENDS="$BACKENDS avx512"
fi
for b in $BACKENDS; do
    ULMBLAS_BACKEND=$b "$TMP/test_ulmblas"
done

echo "== Netlib test programs against $LIBDIR"
sh test/check-shared.sh "$LIBDIR" "$INCDIR" "$TMP/netlib" ulmblas f77
if [ -n "$BACKENDS" ]; then
    ULMBLAS_BACKEND=generic sh test/check-shared.sh "$LIBDIR" "$INCDIR" \
        "$TMP/netlib-generic" ulmblas f77
fi

if [ "${1:-}" = libblas ]; then
    echo "== libblas.so.3 (and liblapack.so.3) of the system"
    cc packaging/ci/libblas-test.c -o "$TMP/libblas-test" -ldl
    "$TMP/libblas-test"
fi
echo "== done"
