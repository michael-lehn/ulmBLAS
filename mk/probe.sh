#!/bin/sh
#
# Usability checks of the tools needed by the build (see mk/tools.mk).
#
#     probe.sh KIND
#
# KIND is one of
#
#     cxx       C++ compiler          (compile mk/probe/cxx.cc)
#     cxx-link  C++ compiler, linking (compile and link mk/probe/cxx.cc)
#     ar        archiver and ranlib   (create a library, list its members)
#     cc        C compiler            (compile and link mk/probe/cc.c)
#     fc        Fortran compiler      (compile mk/probe/fc.f)
#     fc-link   Fortran objects linked by the C++ compiler using FLIBS
#               (link and run a program)
#     cpu       CPU features PROBE_FEATURES (e.g. avx fma) supported by the
#               compiler (with the flags of PROBE_CXX), the CPU and the
#               operating system (compile, link and run mk/probe/cpu.cc).
#               PROBE_CXX_BASE: the compiler without the flags required for
#               the features; if the probe cannot be built with it, or for
#               an unknown feature, the exit status is 2 (an error, not an
#               unsupported feature)
#
# Building the libraries does not require executing programs for the target
# (cross compilation): only fc-link and cpu run a program.  They are used by
# targets that run programs anyway (tests, benchmarks); fc-link detects a
# Fortran runtime library that is not found at run time.
#
# The tools and flags are passed through the environment as command lines
# (they are split by the shell):
#
#     PROBE_CXX      $(CXX) and C++ compiler flags
#     PROBE_CC       $(CC) and C compiler flags
#     PROBE_FC       $(FC) and Fortran compiler flags
#     PROBE_LD       linker flags and libraries ($(LDFLAGS), $(LDLIBS))
#     PROBE_FLIBS    $(FLIBS)
#     PROBE_AR       $(AR) $(ARFLAGS)
#     PROBE_RANLIB   $(RANLIB)
#
# Exit status 0 if the tool works.  Otherwise an explanation is printed to
# standard error and the exit status is 1 (cpu: 1 if a feature is not
# supported, 2 for an error of the probe itself).
#

src=`dirname "$0"`/probe
tmp=${TMPDIR:-/tmp}/ulmblas-probe.$$
log=$tmp/log

trap 'rm -rf "$tmp"' 0
trap 'exit 1' 1 2 15
mkdir -p "$tmp" || exit 1

kind=$1

# what_for, variable name and command of the tool checked
case $kind in
    cxx)
        tool="C++ compiler"
        var=CXX
        cmd=$PROBE_CXX
        what="It is required for building the ulmBLAS libraries (C++11 or
  newer, selected by CXXSTD) and the C++ tests."
        ;;
    cxx-link)
        tool="C++ compiler (linking)"
        var=CXX
        cmd="$PROBE_CXX $PROBE_LD"
        what="Linking C++ programs is required for the tests and benchmarks
  (the C++ runtime library must be linked, e.g. g++ instead of gcc)."
        ;;
    ar)
        tool="archiver"
        var=AR
        cmd="$PROBE_AR ; $PROBE_RANLIB"
        what="It is required for creating the static libraries."
        ;;
    cc)
        tool="C compiler"
        var=CC
        cmd=$PROBE_CC
        what="It is required for the Netlib CBLAS test drivers (make check)
  and for the benchmarks (make bench)."
        ;;
    fc)
        tool="Fortran compiler"
        var=FC
        cmd=$PROBE_FC
        what="It is required for the Fortran 77 interface (make f77blas), the
  reference BLAS (make refblas), the Netlib test drivers (make check-netlib)
  and the benchmarks."
        ;;
    fc-link)
        tool="Fortran runtime library (linked by the C++ compiler)"
        var=FLIBS
        cmd="$PROBE_CXX ... $PROBE_FLIBS"
        what="Fortran objects are linked with the C++ compiler as linker front
  end.  FLIBS must name the libraries of the Fortran runtime (for gfortran
  determined automatically: -L<dir> -lgfortran)."
        ;;
    cpu)
        tool="CPU features ($PROBE_FEATURES)"
        var=CXX
        cmd=$PROBE_CXX
        what="They are required by a test configuration (CONFIG_REQUIRES)."
        ;;
    *)
        echo "$0: unknown kind '$kind'" >&2
        exit 2
        ;;
esac

# fail MESSAGE: print diagnostic and exit
fail()
{
    {
        echo
        echo "error: $tool not usable: $1"
        echo "  $what"
        echo "  Command: $cmd"
        if [ -s "$log" ]; then
            echo "  Output:"
            sed -n '1,25s/^/    /p' "$log"
        fi
        echo "  Select another one with '$var=...' on the make command line"
        echo "  or in the environment, e.g. 'make $hint'."
        echo
    } >&2
    exit 1
}

# exists COMMAND-LINE: check that the program of a command line exists
exists()
{
    set -- $1
    if [ $# -eq 0 ]; then
        fail "no program specified"
    fi
    if ! command -v "$1" > /dev/null 2>&1; then
        fail "program '$1' not found"
    fi
}

# run COMMAND...: run a command, output goes to the log
run()
{
    "$@" > "$log" 2>&1
}

case $kind in
    cxx)
        hint="CXX=clang++ or CXX=g++-14"
        exists "$PROBE_CXX"
        run $PROBE_CXX -c -o "$tmp/cxx.o" "$src/cxx.cc" \
            || fail "compiling a C++11 program failed"
        ;;
    cxx-link)
        hint="CXX=clang++ or CXX=g++-14"
        exists "$PROBE_CXX"
        run $PROBE_CXX -o "$tmp/cxx" "$src/cxx.cc" $PROBE_LD \
            || fail "compiling and linking a C++11 program failed"
        ;;
    ar)
        hint="AR=gcc-ar RANLIB=gcc-ranlib"
        exists "$PROBE_AR"
        exists "$PROBE_RANLIB"
        exists "$PROBE_CXX"
        run $PROBE_CXX -c -o "$tmp/lib.o" "$src/lib.cc" \
            || fail "compiling the probe failed (check the C++ compiler)"
        run $PROBE_AR "$tmp/libprobe.a" "$tmp/lib.o" \
            || fail "creating a library failed"
        run $PROBE_RANLIB "$tmp/libprobe.a" || fail "ranlib failed"
        set -- $PROBE_AR
        run "$1" t "$tmp/libprobe.a" \
            && grep -q 'lib\.o' "$log" \
            || fail "the created library does not contain the object file"
        ;;
    cc)
        hint="CC=clang or CC=gcc-14"
        exists "$PROBE_CC"
        run $PROBE_CC -o "$tmp/cc" "$src/cc.c" $PROBE_LD \
            || fail "compiling and linking a C program failed"
        ;;
    fc)
        hint="FC=gfortran-14"
        exists "$PROBE_FC"
        run $PROBE_FC -c -o "$tmp/fc.o" "$src/fc.f" \
            || fail "compiling a Fortran 77 program failed"
        ;;
    fc-link)
        hint="FLIBS='-L/path/to/fortran/lib -lgfortran'"
        exists "$PROBE_FC"
        exists "$PROBE_CXX"
        run $PROBE_FC -c -o "$tmp/fc_link_f.o" "$src/fc_link.f" \
            || fail "compiling the Fortran probe failed (check FC)"
        run $PROBE_CXX -c -o "$tmp/fc_link_cc.o" "$src/fc_link.cc" \
            || fail "compiling the C++ probe failed (check CXX)"
        run $PROBE_CXX -o "$tmp/fc_link" "$tmp/fc_link_f.o" \
                "$tmp/fc_link_cc.o" $PROBE_LD $PROBE_FLIBS \
            || fail "linking a Fortran program with the C++ compiler failed"
        run "$tmp/fc_link" \
            || fail "the linked program does not run (runtime library not found?)"
        ;;
    cpu)
        hint="CXX=g++"
        exists "$PROBE_CXX_BASE"
        if ! run $PROBE_CXX_BASE -o "$tmp/cpu" "$src/cpu.cc" $PROBE_LD; then
            ( fail "the probe cannot be built (without the flags of the features)" )
            exit 2
        fi
        run "$tmp/cpu" $PROBE_FEATURES
        if [ $? -eq 2 ]; then
            ( fail "unknown feature" )
            exit 2
        fi
        run $PROBE_CXX -o "$tmp/cpu" "$src/cpu.cc" $PROBE_LD \
            || fail "the compiler does not accept the flags for the features"
        run "$tmp/cpu" $PROBE_FEATURES
        status=$?
        case $status in
            0)  ;;
            1)  fail "not supported by the CPU or the operating system" ;;
            2)  ( fail "unknown feature" ); exit 2 ;;
            132)
                fail "not supported by the CPU or the operating system (illegal instruction)" ;;
            *)  ( fail "the probe terminated abnormally (status $status)" ); exit 2 ;;
        esac
        ;;
esac

exit 0
