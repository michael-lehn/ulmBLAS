#!/bin/sh
# Builds all examples with GCC and clang (C++11, as ulmBLAS) and stores the
# outputs shown in the guide in out/.  Some examples fail to compile or to
# link on purpose; then the first lines of the messages are kept.
# macOS with Homebrew: g++-16 and the clang of Homebrew's llvm.
cd "$(dirname "$0")"
GXX=${GXX:-g++-16}
CLANGXX=${CLANGXX:-clang++}
STD=-std=c++11
FLAGS="$STD -O2 -Wall -Wextra"
mkdir -p out bin

# run <file> <name> [extra flags]: build with both compilers and run
run() {
    f=$1; n=$2; shift 2
    $GXX $FLAGS "$@" $f -o bin/$n.gcc && bin/$n.gcc > out/$n-gcc.txt 2>&1
    $CLANGXX $FLAGS "$@" $f -o bin/$n.clang && bin/$n.clang > out/$n-clang.txt 2>&1
}

# fails <file> <name> <lines> <pattern> [extra flags]: compile, keep the
# first <lines> lines of the messages that match <pattern>
fails() {
    f=$1; n=$2; k=$3; p=$4; shift 4
    $GXX $FLAGS "$@" $f -o bin/$n.gcc 2>&1 | grep -E "$p" | head -$k > out/$n-gcc.txt
    $CLANGXX $FLAGS "$@" $f -o bin/$n.clang 2>&1 | grep -E "$p" | head -$k > out/$n-clang.txt
}

# Chapter 2: function templates
run templates/scal.cc scal
fails templates/deduce.cc deduce 8 "error|note|candidate|conflicting|^ +[0-9]+ \|" -c
$GXX $FLAGS -DFIXED -c templates/deduce.cc -o /dev/null

# the symbols and the assembly code of scal.cc (clang, -O1 without
# inlining, so that every instantiation stays a function of its own)
# asm <file.s> <symbol>: the instructions of one function
asm() {
    awk -v f="$2:" '$1 == f {p = 1} p {print} p && $1 == "ret" {exit}' "$1" |
	grep -vE '^[[:space:]]*\.|^Lfunc|^[[:space:]]*;' | sed 's/[[:space:]]*;.*$//'
}
$CLANGXX $STD -O1 -fno-inline -c templates/scal.cc -o bin/scal.o
nm bin/scal.o | awk '$2 ~ /^[TtWw]$/ && $3 ~ /scal/ {print $3}' |
    while read s; do
	printf '%-36s %s\n' "$s" "$(echo "$s" | c++filt)"
    done > out/scal-nm.txt
$CLANGXX $STD -O1 -fno-inline -S templates/scal.cc -o bin/scal.s
asm bin/scal.s __Z5sscallfPfl > out/scal-asm-sscal.txt
asm bin/scal.s __Z4scalIlffEvT_RKT0_PT1_S0_ > out/scal-asm-template.txt

# Chapter 3: names
$CLANGXX $STD -O0 -c linkage/interface.cc -o bin/interface.o
nm bin/interface.o | awk '$2 ~ /^[TtWw]$/ && $3 !~ /^ltmp/ {print $3}' |
    while read s; do
	printf '%-42s %s\n' "$s" "$(echo "$s" | c++filt)"
    done > out/interface-nm.txt
$CLANGXX $STD -O1 -S linkage/interface.cc -o bin/interface.s
asm bin/interface.s _cblas_dscal > out/interface-asm.txt

# Chapter 4: values and references
run values/categories.cc categories
for c in "$GXX gcc" "$CLANGXX clang"; do
    set -- $c
    $1 $STD -O0 -g -fsanitize=address -Wall -Wextra values/lifetime.cc \
	-o bin/lifetime.$2 > out/lifetime-warn-$2.txt 2>&1
    bin/lifetime.$2 2>&1 | grep -E "^[a-z0-9]+ =|ERROR|READ|#0" |
	awk '!(/#0/ && seen++)' |
	sed -e 's/0x[0-9a-f]*/0x.../g' -e 's/==[0-9]*==/==...==/' > out/lifetime-$2.txt
    $1 $STD -O2 values/lifetime.cc -o bin/lifetime-O2.$2 2>/dev/null &&
	bin/lifetime-O2.$2 > out/lifetime-O2-$2.txt
    $1 $STD -O0 values/odruse.cc -o bin/odruse.$2 2>&1 |
	grep -vE "^collect2|use -v" > out/odruse-$2.txt
    $1 $STD -O0 -DFIXED values/odruse.cc -o bin/odruse.$2 && bin/odruse.$2 > /dev/null
    $1 -std=c++17 -O0 values/odruse.cc -o bin/odruse.$2 && bin/odruse.$2 > /dev/null
done

# Chapter 5: complex numbers
run complex/layout.cc layout
run complex/multiply.cc multiply -mcpu=native

# Chapter 6: types
run types/promote.cc promote
fails types/promote.cc promote-mixed 1 "error" -c -DMIXED
run types/traits.cc traits
fails types/traits.cc traits-wrong 1 "error" -c -DWRONG

# Chapter 7: descriptors
run descriptors/kernels.cc kernels

# Chapter 8: casts
run casts/casts.cc casts
fails casts/casts.cc casts-wrong 1 "error" -c -DWRONG

# Chapter 9: RAII
run raii/buffer.cc buffer

# Chapter 10: lambdas
run lambdas/unitstride.cc unitstride

# Chapter 11: small things
run misc/conveniences.cc conveniences
fails misc/conveniences.cc conveniences-wrong 1 "error" -c -DWRONG
run misc/nanmax.cc nanmax

# Chapter 12: move semantics
run notused/move.cc move
$GXX $FLAGS -fno-elide-constructors notused/move.cc -o bin/move-noelide &&
    bin/move-noelide | sed -n '/Vector w/,$p' > out/move-noelide-11.txt
$GXX -std=c++17 -O2 -fno-elide-constructors notused/move.cc -o bin/move-noelide &&
    bin/move-noelide | sed -n '/Vector w/,$p' > out/move-noelide-17.txt
