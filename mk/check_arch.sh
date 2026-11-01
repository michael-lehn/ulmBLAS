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
#  flags).  A configuration is listed if its
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
#  kernel set such as sse; empty: any),
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
    #  per type: "ulmBLAS double: kernel sse, MR x NR = 4 x 4, MC, ..."
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
    if [ -n "$4" ] && [ "$4" != "$set_" ]; then
        WHY="with this compiler these flags select $set_, not $4"
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
    add own "ulmBLAS SSE3 (default)" "$DEFAULT_FLAGS" "sse3" sse;;
*)
    add own "reference kernels (default)" "$DEFAULT_FLAGS" "";;
esac

#  ---- suggestion ---------------------------------------------------------
SUG=; REASON=; EVIDENCE=
listed() { echo "$ROWS" | grep -q "^[a-z]*|$1|"; }
if listed "ulmBLAS SSE3 (default)"; then
    SUG="ulmBLAS SSE3 (default)"
    REASON="the only optimized micro kernel; no selection needed"
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
echo "kernel.  The kernels: ulmblas/level3/ukernel/README.md."
