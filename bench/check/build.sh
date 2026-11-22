#!/bin/sh
#
#  build.sh ULM OUT: the ATLAS testers (ULM/bench/l[123]blastst.c) for the
#  types s, d, c, z in two modes, against the static libraries of ULM
#  (libcblas.a, libf77blas.a, librefblas.a: make cblas f77blas refblas):
#
#    OUT/cb-x?l?blastst  ATL_ -> cblas_?xxx of ulmBLAS (shim, genshim.py cblas);
#                        trusted: reference BLAS (ULM/librefblas.a)
#    OUT/f7-x?l?blastst  ATL_ -> ?xxx_ of ulmBLAS (shim, genshim.py f77);
#                        trusted: reference BLAS without trailing underscores
#
#  The trusted wrappers, norms and generators: ATLAS 3.10.3, src/testing
#  (and ATL_set.c of src/auxil), unmodified, in atlas/; atlas/?.files lists
#  the files each type needs.  Headers: ULM/bench (those of ATLAS) and inc/.
#  CC, FC from the environment (default cc, gfortran); OUT/build receives
#  the objects.
#
set -eu
ULM=$(cd "$1" && pwd)
mkdir -p "$2"
OUT=$(cd "$2" && pwd)
D=$(cd "$(dirname "$0")" && pwd)
A=$D/atlas
CC=${CC:-cc}
FC=${FC:-gfortran}
case $(uname -s) in
Darwin) CXXRT="-lc++"; GS=""; GE="" ;;
*)      CXXRT="-lstdc++ -pthread -lm"; GS="-Wl,--start-group"; GE="-Wl,--end-group" ;;
esac
for l in libcblas.a libf77blas.a librefblas.a; do
    [ -f "$ULM/$l" ] || { echo "error: $ULM/$l not found (make cblas f77blas refblas)" >&2; exit 2; }
done
B=$OUT/build
/bin/rm -rf "$B"
mkdir -p "$B"

#  reference BLAS without trailing underscores (F77 mode)
mkdir -p "$B/refnu"
for f in "$ULM"/refblas/*.f; do
    $FC -O2 -fno-underscoring -c -o "$B/refnu/$(basename "$f" .f).o" "$f"
done
ar cr "$B/librefblas_nu.a" "$B"/refnu/*.o
ranlib "$B/librefblas_nu.a"

#  shims (generated C, formatted if clang-format is found)
for m in cblas f77; do
    python3 "$D/genshim.py" $m > "$B/shim-$m.c"
    if command -v clang-format > /dev/null 2>&1; then
        clang-format -i "$B/shim-$m.c"
    fi
    $CC -O2 -I"$ULM" -c -o "$B/shim-$m.o" "$B/shim-$m.c"
done
for s in stub-ger2 stub-sger2 stub-dger2; do
    $CC -O2 -c -o "$B/$s.o" "$D/$s.c"
done

for M in Add_ NoChange; do
    case $M in Add_) FU="" ;; NoChange) FU="-fno-underscoring" ;; esac
    BASE="-O2 -DL2SIZE=4194304 -D$M -DF77_INTEGER=int -DStringSunStyle -DATL_SSE2 -I$D/inc -I$ULM/bench"
    for P in s d c z; do
        case $P in
        s) T=SREAL; STUB=stub-sger2 ;;
        d) T=DREAL; STUB=stub-dger2 ;;
        c) T=SCPLX; R=SREAL; STUB=stub-ger2 ;;
        z) T=DCPLX; R=DREAL; STUB=stub-ger2 ;;
        esac
        O=$B/obj-$M-$P
        mkdir -p "$O"
        for f in $(cat "$A/$P.files"); do
            $CC $BASE -D$T -w -c -o "$O/$f.o" "$A/$f.c"
        done
        #  ATL_set of the real type (complex types: of their real type,
        #  ATL_set.c does not compile for complex types)
        case $P in
        s|d) $CC $BASE -D$T -w -c -o "$O/ATL_set.o" "$A/ATL_set.c" ;;
        c|z) $CC $BASE -D$R -w -c -o "$O/ATL_set_real.o" "$A/ATL_set.c" ;;
        esac
        for h in ATL_cputime ATL_flushcache; do
            $CC $BASE -DDREAL -w -c -o "$O/$h.o" "$ULM/bench/$h.c"
        done
        $CC $BASE -DDREAL -w -c -o "$O/ATL_dzero.o" "$ULM/bench/ATL_zero.c"
        $FC -O2 $FU -c -o "$O/wrap.o" "$A/ATL_${P}f77wrap.f"
        /bin/cp "$B/$STUB.o" "$O/"
        /bin/rm -f "$B/libtst-$M-$P.a"
        ar cr "$B/libtst-$M-$P.a" "$O"/*.o
        ranlib "$B/libtst-$M-$P.a"
        for t in l1blastst l2blastst l3blastst; do
            $CC $BASE -D$T -w -c -o "$B/$t-$M-$P.o" "$ULM/bench/$t.c"
            case $M in
            Add_)     X=cb; LIBS="$ULM/libcblas.a $ULM/librefblas.a"; SHIM=$B/shim-cblas.o ;;
            NoChange) X=f7; LIBS="$ULM/libf77blas.a $B/librefblas_nu.a"; SHIM=$B/shim-f77.o ;;
            esac
            $FC -o "$OUT/$X-x$P$t" "$B/$t-$M-$P.o" "$SHIM" $GS "$B/libtst-$M-$P.a" $LIBS $GE $CXXRT
        done
    done
done
echo "built: $(ls "$OUT" | grep -c 'blastst$') testers"
