#!/bin/sh
#  build-refs.sh : build BLIS 2.0 and OpenBLAS v0.3.34 (static, single
#  threaded, CBLAS) below /tmp/ulmblas-avx.tGdSUq/refs.  No installation outside /tmp/ulmblas-avx.tGdSUq, no root.
set -u
D=/tmp/ulmblas-avx.tGdSUq; R=$D/refs
mkdir -p $R/src $D/tmp; TMPDIR=$D/tmp; export TMPDIR
cd $R/src || exit 1
curl -fsSL -o blis-2.0.tar.gz https://codeload.github.com/flame/blis/tar.gz/refs/tags/2.0 || exit 1
curl -fsSL -o OpenBLAS-0.3.34.tar.gz https://codeload.github.com/OpenMathLib/OpenBLAS/tar.gz/refs/tags/v0.3.34 || exit 1
sha256sum *.tar.gz > $R/sources.sha256; cat $R/sources.sha256
tar xzf blis-2.0.tar.gz && tar xzf OpenBLAS-0.3.34.tar.gz || exit 1

echo "== BLIS 2.0 (configuration haswell: dgemm micro kernel 6x8, no threading)"
cd $R/src/blis-2.0 || exit 1
( ./configure --prefix=$R/blis --disable-shared --enable-cblas -t no haswell   && make -j12 && make install ) > $R/blis-build.log 2>&1
echo "BLIS build rc=$?: $(ls $R/blis/lib/libblis.a 2>/dev/null)"
grep -m3 -iE 'configuration|threading|cblas' $R/blis-build.log | head -3

echo "== OpenBLAS v0.3.34 (target detected, no threading, no Fortran/LAPACK)"
cd $R/src/OpenBLAS-0.3.34 || exit 1
( make -j12 USE_THREAD=0 USE_OPENMP=0 NO_SHARED=1 NOFORTRAN=1 NO_LAPACK=1   && make PREFIX=$R/openblas NO_SHARED=1 install ) > $R/openblas-build.log 2>&1
echo "OpenBLAS build rc=$?: $(ls $R/openblas/lib/libopenblas.a 2>/dev/null)"
grep -E '^(CORE|LIBCORE|USE_THREAD|NUM_THREADS|DYNAMIC_ARCH|ARCH)=' Makefile.conf* 2>/dev/null | sort -u | head
grep -m1 -E 'OpenBLAS build complete|has been built' $R/openblas-build.log; tail -3 $R/openblas-build.log | head -2
