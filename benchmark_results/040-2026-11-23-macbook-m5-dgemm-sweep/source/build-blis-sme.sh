#!/bin/sh
#  build-blis-sme.sh DIR : build BLIS with pull request #921 (SME2 kernels
#  for Apple M4/M5, fork Luislo1/blis), configuration m4sme_p, static,
#  single threaded, CBLAS, with Apple clang, below DIR (installed to
#  DIR/refs/blis-sme).  The campaign used the head of the pull request
#  235ef8ec097c43ab8299c89071af1b5b5e180843 ("Fix missing parameter",
#  2026-10-23).  BLIS 2.0 and OpenBLAS 0.3.34: build-refs.sh of the campaign
#  2026-10-24-macbook-m5-neon.  Nothing is installed outside DIR.
set -u
D=${1:?usage: build-blis-sme.sh DIR}; J=${J:-8}; C=235ef8ec097c43ab8299c89071af1b5b5e180843
mkdir -p $D/refs || exit 1
cd $D || exit 1
git clone -q https://github.com/flame/blis.git blis-pr921 || exit 1
cd blis-pr921 && git fetch -q origin pull/921/head:pr921 && git checkout -q $C || exit 1
( CC=/usr/bin/clang ./configure --prefix=$D/refs/blis-sme --disable-shared --enable-cblas -t no m4sme_p && make -j$J && make install ) > $D/blis-sme-build.log 2>&1
echo "BLIS PR #921 build rc=$?: $(ls $D/refs/blis-sme/lib/libblis.a 2>/dev/null)"
