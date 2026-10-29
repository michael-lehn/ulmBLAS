#!/bin/sh
#
#  check_arch.sh: the ARCHFLAGS configurations of ulmBLAS that this machine
#  (CPU and operating system) and the C++ compiler support, the micro kernel
#  each one selects, and a suggested one.  Started by 'make check_arch'
#  with CXX and ARCHFLAGS of the build; changes nothing in the tree.
#
#  Detection: CPU features are read without executing any of the
#  instructions in question: on x86 with __builtin_cpu_supports (CPUID and
#  the operating system's support, mk/probe/cpu.cc built without extra
#  flags), on macOS arm64 with sysctl.  A configuration is listed if its
#  features are supported, the compiler accepts its flags and the ulmBLAS
#  headers compile with them; the selected kernels are reported by
#  bench/gemm/ulmblas-kernels.cc, compiled with the flags (it is only run
#  after the features have been verified).
#
set -u
TOP=$(cd "$(dirname "$0")/.." && pwd)
CXX=${CXX:-c++}
TMP=$(mktemp -d 2>/dev/null || mktemp -d -t ulmblas-check-arch) || exit 1
trap 'rm -rf "$TMP"' EXIT INT TERM
OS=$(uname -s); ARCH=$(uname -m)

#  ---- system, compiler, CPU features -------------------------------------
case $OS in
    Darwin) MODEL=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
            VENDOR=$(sysctl -n machdep.cpu.vendor 2>/dev/null);;
    *)      MODEL=$(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)
            VENDOR=$(awk -F': ' '/^vendor_id/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)
            [ -n "$MODEL" ] || MODEL=$(awk -F': ' '/^(CPU part|Model)/ {print $2; exit}' /proc/cpuinfo 2>/dev/null);;
esac
echo "System:    $OS $ARCH, ${MODEL:-unknown CPU}${VENDOR:+ ($VENDOR)}"
CXXV=$($CXX --version 2>/dev/null | head -1)
if [ -z "$CXXV" ]; then
    echo "error: the C++ compiler CXX=$CXX does not run" >&2; exit 1
fi
echo "Compiler:  CXX=$CXX ($CXXV)"

FEATURES=; UNKNOWN=
case $ARCH in
x86_64|amd64|i386|i686)
    if $CXX -o "$TMP/cpu" "$TOP/mk/probe/cpu.cc" > "$TMP/err" 2>&1; then
        for f in sse3 avx fma avx2 avx512f; do
            "$TMP/cpu" $f >/dev/null 2>&1 && FEATURES="$FEATURES $f"
        done
    else
        UNKNOWN="the feature probe mk/probe/cpu.cc does not build with CXX: $(grep -m1 -i error "$TMP/err" | cut -c1-100)"
        echo "note: $UNKNOWN"
    fi;;
arm64|aarch64)
    FEATURES=" neon"
    if [ "$OS" = Darwin ] \
       && [ "$(sysctl -n hw.optional.arm.FEAT_SME2 2>/dev/null)" = 1 ] \
       && [ "$(sysctl -n hw.optional.arm.FEAT_SME_F64F64 2>/dev/null)" = 1 ] \
       && [ "$(sysctl -n hw.optional.arm.sme_max_svl_b 2>/dev/null)" = 64 ]; then
        FEATURES="$FEATURES sme2-f64-svl512"
    fi;;
esac
if [ -n "$UNKNOWN" ]; then
    echo "Features:  unknown"
else
    echo "Features:  ${FEATURES# }${FEATURES:+  (CPU and operating system)}"
fi
has() { case " $FEATURES " in *" $1 "*) return 0;; esac; return 1; }

#  ---- one configuration --------------------------------------------------
#  try LABEL FLAGS FEATURES EXPECT: OK=1 and KERNEL (what ulmBLAS selects)
#  if the configuration is usable here and selects the kernel EXPECT (a
#  kernel set such as avx, or the name of a vendored kernel; empty: any),
#  else OK= and WHY
try() {
    OK=; KERNEL=; WHY=
    for f in $3; do
        [ -n "$UNKNOWN" ] && { WHY="CPU features unknown"; return; }
        has $f || { WHY="needs $f"; return; }
    done
    if ! $CXX -std=c++11 $2 -I"$TOP" -o "$TMP/k" \
            "$TOP/bench/gemm/ulmblas-kernels.cc" > "$TMP/err" 2>&1; then
        WHY="the compiler rejects the flags or the kernel: $(grep -m1 -i error "$TMP/err" | cut -c1-90)"
        return
    fi
    out=$("$TMP/k") || { WHY="the probe does not run"; return; }
    set_=$(echo "$out" | awk '/kernel set:/ {print $NF}')
    vend=$(echo "$out" | sed -n 's/.*vendored kernel \([^ ]*\).*/\1/p')
    #  per type: "ulmBLAS double: kernel neon, MR x NR = 6 x 8, MC, ..."
    k() { echo "$out" | sed -n "s/^ulmBLAS $1: kernel \([^,]*\), MR x NR = \([0-9]*\) x \([0-9]*\),.*/\1 \2x\3/p"; }
    KERNEL="double $(k double)"
    others=
    for t in float "complex float" "complex double"; do
        kt=$(k "$t")
        case $kt in
            ref*) others="$others${others:+, }$t";;
            *)    KERNEL="$KERNEL; $t $kt";;
        esac
    done
    [ -n "$others" ] && KERNEL="$KERNEL; reference kernel for $others"
    if [ -n "$4" ] && [ "$4" != "$set_" ] && [ "$4" != "$vend" ]; then
        WHY="with this compiler these flags select ${vend:-$set_}, not $4"
        return
    fi
    OK=1
}

ROWS=; NOT=
add() {     # add GROUP LABEL FLAGS FEATURES EXPECT
    try "$2" "$3" "$4" "${5:-}"
    if [ -n "$OK" ]; then
        ROWS="$ROWS$1|$2|$3|$KERNEL
"
    else
        NOT="$NOT  $2: $WHY
"
    fi
}

DEFAULT_FLAGS=${ARCHFLAGS_DEFAULT:-}
case $ARCH in
x86_64|amd64)
    add own "ulmBLAS SSE3 (default)" "$DEFAULT_FLAGS" "sse3" sse
    add own "ulmBLAS AVX/FMA"        "-mavx -mfma" "avx fma" avx
    add own "-march=native"          "-march=native" ""
    add ven "BLIS penryn 4x4"        "-msse3 -DULMBLAS_DGEMM_UKERNEL_blis_penryn_4x4" "sse3" penryn_4x4
    add ven "BLIS sandybridge 8x4"   "-mavx -DULMBLAS_DGEMM_UKERNEL_blis_sandybridge_8x4" "avx" sandybridge_8x4
    add ven "BLIS haswell 6x8"       "-mavx -mfma -DULMBLAS_DGEMM_UKERNEL_blis_haswell_6x8" "avx fma" haswell_6x8
    add ven "BLIS haswell 8x6"       "-mavx -mfma -DULMBLAS_DGEMM_UKERNEL_blis_haswell_8x6" "avx fma" haswell_8x6
    add ven "OpenBLAS HASWELL 4x8"   "-mavx -mavx2 -mfma -DULMBLAS_DGEMM_UKERNEL_openblas_haswell_4x8" "avx avx2 fma" haswell_4x8
    add ven "BLIS skx 16x14"         "-mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_blis_skx_16x14" "avx fma avx512f" skx_16x14
    add ven "AOCL-BLAS zen4 32x6"    "-mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_blis_zen4_32x6" "avx fma avx512f" zen4_32x6
    add ven "AOCL-BLAS zen5 8x24"    "-mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_blis_zen5_8x24" "avx fma avx512f" zen5_8x24
    add ven "OpenBLAS SKYLAKEX 16x12" "-mavx -mfma -mavx512f -DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12" "avx fma avx512f" skylakex_16x12;;
arm64|aarch64)
    add own "ulmBLAS NEON (default)" "$DEFAULT_FLAGS" "neon" neon
    add own "ulmBLAS SME2 (experimental)" "-mcpu=apple-m4" "sme2-f64-svl512" sme;;
*)
    add own "reference kernels (default)" "$DEFAULT_FLAGS" "";;
esac

#  ---- suggestion ---------------------------------------------------------
#  From the benchmark campaigns in benchmark_results/ where a measurement
#  exists, otherwise a stated heuristic.
SUG=; REASON=; EVIDENCE=
#  evidence NNN ...: the campaign directories with these numbers
evidence() {
    for n in "$@"; do
        for d in "$TOP"/benchmark_results/$n-*; do
            [ -d "$d" ] && EVIDENCE="$EVIDENCE    benchmark_results/$(basename "$d")
"
        done
    done
}
listed() { echo "$ROWS" | grep -q "^[a-z]*|$1|"; }
intel=; amd=
case $VENDOR in GenuineIntel) intel=1;; AuthenticAMD) amd=1;; esac
if listed "ulmBLAS SME2 (experimental)"; then
    SUG="ulmBLAS SME2 (experimental)"
    REASON="measured on an Apple M5: DGEMM about 7 times as fast as with the NEON kernel for large matrices"
    evidence 004 006 007
elif has avx512f && [ -n "$intel" ] && listed "OpenBLAS SKYLAKEX 16x12"; then
    SUG="OpenBLAS SKYLAKEX 16x12"
    REASON="measured on an Intel Xeon Platinum 8358 (Ice Lake): the fastest ulmBLAS build, DGEMM at 0.97 of native OpenBLAS; other AVX-512 CPUs not measured"
    evidence 023
elif has avx512f && [ -n "$amd" ] && listed "AOCL-BLAS zen4 32x6"; then
    SUG="AOCL-BLAS zen4 32x6"
    REASON="preliminary measurement on an AMD EPYC 9454 (Zen 4, one socket of a shared node): the fastest kernel; AVX-512 kernels only 0-4% faster than AVX2 ones there; Zen 5 not measured"
    evidence 015
elif has avx2 && has fma && [ -n "$intel" ] && listed "OpenBLAS HASWELL 4x8"; then
    SUG="OpenBLAS HASWELL 4x8"
    REASON="measured on an Intel Core i5-12500 (Alder Lake): DGEMM at 0.98-1.00 of native OpenBLAS, 1-2% faster than ulmBLAS's own AVX/FMA kernel from n = 64 on"
    evidence 024
elif has avx && has fma && listed "ulmBLAS AVX/FMA"; then
    SUG="ulmBLAS AVX/FMA"
    REASON="heuristic, not measured on this CPU type: ulmBLAS's port of BLIS's haswell 6x8 kernel, which BLIS and AOCL-BLAS use for AVX2/FMA CPUs without AVX-512"
elif has avx && listed "BLIS sandybridge 8x4"; then
    SUG="BLIS sandybridge 8x4"
    REASON="heuristic: the only kernel for AVX without FMA; on an Alder Lake CPU it was 1.8 times the SSE3 kernel, not measured on a Sandy Bridge CPU"
    evidence 011
elif listed "ulmBLAS SSE3 (default)"; then
    SUG="ulmBLAS SSE3 (default)"
    REASON="the SSE3 kernel and BLIS penryn 4x4 were equally fast: no reason to prefer one, the default needs no selection macro"
    evidence 011
elif listed "ulmBLAS NEON (default)"; then
    SUG="ulmBLAS NEON (default)"
    REASON="the only kernel usable here with this compiler; measured on an Apple M5 only"
    evidence 002
elif listed "reference kernels (default)"; then
    SUG="reference kernels (default)"
    REASON="no optimized kernel for this architecture"
fi

#  ---- report -------------------------------------------------------------
cmd() { [ -n "$1" ] && echo "make ARCHFLAGS='$1'" || echo "make"; }
print_group() {
    echo "$ROWS" | while IFS='|' read g label flags kernel; do
        [ "$g" = "$1" ] || continue
        if [ "$label" = "$SUG" ]; then
            printf '=> %-28s %s  (suggested)\n' "$label" "$(cmd "$flags")"
        else
            printf '   %-28s %s\n' "$label" "$(cmd "$flags")"
        fi
        printf '   %-28s -> %s\n' "" "$kernel"
    done
}
row() { echo "$ROWS" | awk -F'|' -v l="$1" '$2==l {print $'"$2"'}'; }
echo
case ${ARCHFLAGS_ORIGIN:-} in
    file)            origin="the default of make.inc";;
    "command line")  origin="set on the command line";;
    environment*)    origin="set in the environment";;
    *)               origin=${ARCHFLAGS_ORIGIN:-unknown};;
esac
echo "Current ARCHFLAGS: '${ARCHFLAGS:-}' ($origin)"
echo
echo "Supported configurations (CPU, operating system and compiler checked):"
echo
echo " ulmBLAS's own micro kernels, selected by the instruction set:"
print_group own
if echo "$ROWS" | grep -q '^ven|'; then
    echo
    echo " Vendored DGEMM micro kernels (BLIS, AOCL-BLAS, OpenBLAS), selected by a"
    echo " macro in ARCHFLAGS; they replace the kernel for double only:"
    print_group ven
fi
if [ -n "$NOT" ]; then
    echo
    echo "Not supported here:"
    printf '%s' "$NOT"
fi
echo
DEF=$(echo "$ROWS" | awk -F'|' '$2 ~ /\(default\)/ {print $2; exit}')
if [ -n "$DEF" ] && [ "$DEF" != "$SUG" ]; then
    echo "Default:   $(cmd "$(row "$DEF" 3)")"
    echo "           -> $(row "$DEF" 4)"
fi
if [ -n "$SUG" ]; then
    echo "Suggested: $(cmd "$(row "$SUG" 3)")"
    echo "           -> $(row "$SUG" 4)"
    echo "  Why: $REASON."
    if [ -n "$EVIDENCE" ]; then
        echo "  Benchmark campaigns:"
        printf '%s' "$EVIDENCE"
    fi
else
    echo "Suggested: make   (no configuration could be verified; the generic default)"
fi
echo
echo "Wider instruction sets alone (e.g. -march=native) do not select a faster"
echo "kernel; vendored kernels need their selection macro.  The kernels:"
echo "ulmblas/level3/ukernel/README.md."
