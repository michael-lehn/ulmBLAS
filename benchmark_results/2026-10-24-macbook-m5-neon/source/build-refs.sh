#!/bin/sh
#  build-refs.sh DIR : build BLIS 2.0 (configuration firestorm) and OpenBLAS
#  v0.3.34 (static, single threaded, CBLAS) below DIR/refs with Apple clang.
#  Nothing is installed outside DIR.
set -u
D=${1:?usage: build-refs.sh DIR}; R=$D/refs; J=${J:-8}
mkdir -p $R/src || exit 1
cd $R/src || exit 1
curl -fsSL -o blis-2.0.tar.gz https://codeload.github.com/flame/blis/tar.gz/refs/tags/2.0 || exit 1
curl -fsSL -o OpenBLAS-0.3.34.tar.gz https://codeload.github.com/OpenMathLib/OpenBLAS/tar.gz/refs/tags/v0.3.34 || exit 1
shasum -a 256 *.tar.gz > $R/sources.sha256; cat $R/sources.sha256
rm -rf blis-2.0 OpenBLAS-0.3.34
tar xzf blis-2.0.tar.gz && tar xzf OpenBLAS-0.3.34.tar.gz || exit 1

echo "== BLIS 2.0 (configuration firestorm: dgemm micro kernel armv8a 8x6r, no threading)"
cd $R/src/blis-2.0 || exit 1
( CC=/usr/bin/clang ./configure --prefix=$R/blis --disable-shared --enable-cblas -t no firestorm && make -j$J && make install ) > $R/blis-build.log 2>&1
echo "BLIS build rc=$?: $(ls $R/blis/lib/libblis.a 2>/dev/null)"

echo "== OpenBLAS v0.3.34 (target detected, no threading, no Fortran/LAPACK)"
cd $R/src/OpenBLAS-0.3.34 || exit 1
( make -j$J CC=/usr/bin/clang USE_THREAD=0 USE_OPENMP=0 NO_SHARED=1 NOFORTRAN=1 NO_LAPACK=1 && make PREFIX=$R/openblas NO_SHARED=1 install ) > $R/openblas-build.log 2>&1
echo "OpenBLAS build rc=$?: $(ls $R/openblas/lib/libopenblas.a 2>/dev/null)"
grep -E '^(CORE|LIBCORE|USE_THREAD|NUM_THREADS|DYNAMIC_ARCH|ARCH)=' Makefile.conf* 2>/dev/null | sort -u | head
