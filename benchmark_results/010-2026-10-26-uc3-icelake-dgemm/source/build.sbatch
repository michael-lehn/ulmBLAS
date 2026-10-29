#!/bin/bash
#  build.sbatch ARCH: build all DGEMM benchmark programs for one CPU type of
#  bwUniCluster 3.0 on a compute node of that type (BLIS "auto" and
#  OpenBLAS select their kernels from the build host).  Submit from the
#  workspace, with the dev partition of the CPU type:
#
#      sbatch -p dev_cpu_il -J build-icelake build.sbatch icelake
#      sbatch -p dev_cpu    -J build-zen4    build.sbatch zen4
#
#  Expects in $W/src (W = workspace directory): blis-2.0.tar.gz,
#  OpenBLAS-0.3.34.tar.gz, eigen-5.0.1.tar.gz, a clone ulmBLAS of
#  https://github.com/michael-lehn/ulmBLAS, and the benchmark sources in
#  $W/src/bench (bench2.c, verify-large.c, bench-eigen.cpp, verify-eigen.cpp,
#  eigen-info.cpp, peak.c, blis-info.c, openblas-info.c, mkl-info.c,
#  include-mkl/cblas.h).  Writes everything below $W/ARCH; installs nothing
#  elsewhere.
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -u
ARCH=$1
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
ULMBLAS_REV=${ULMBLAS_REV:-21197f2}
S=$W/src; A=$W/$ARCH; R=$A/refs; L=$A/logs; BIN=$A/bin
mkdir -p $R $L $BIN $A/tmp; export TMPDIR=$A/tmp
cd $A || exit 1
: > $L/failures.txt
fail() { echo "FAILED (exit $1): $2" | tee -a $L/failures.txt; }
J=$SLURM_CPUS_PER_TASK

module purge
module load compiler/gnu/14.2
{
    echo "build host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"
    grep -m1 'model name' /proc/cpuinfo
    gcc --version | head -1
    echo "gcc -march=native: $(gcc -march=native -Q --help=target | grep -E '^ +-march=' | tr -s ' \t' ' ')"
    sha256sum $S/*.tar.gz
} > $L/build-env.txt 2>&1

#  ulmBLAS: as on heim (make cblas ARCHFLAGS="-mavx -mfma", -O3)
rm -rf ulmblas && mkdir ulmblas
git -C $S/ulmBLAS archive $ULMBLAS_REV | tar xf - -C ulmblas || fail $? "ulmBLAS archive"
echo "ulmBLAS $(git -C $S/ulmBLAS rev-parse $ULMBLAS_REV)" >> $L/build-env.txt
(cd ulmblas && make -j$J cblas CXX=g++ ARCHFLAGS="-mavx -mfma") > $L/build-ulmblas.log 2>&1 || fail $? "build ulmblas"
cat ulmblas/interfaces/blas/C/build.cblas/compile.cmd > $L/ulmblas-compile.txt 2>&1

#  BLIS 2.0: configuration auto (skx on Ice Lake, zen3 on Zen 4) and
#  haswell (the kernel of the ulmBLAS AVX/FMA backend); static, no threads
rm -rf $R/src && mkdir -p $R/src && cd $R/src || exit 1
tar xzf $S/blis-2.0.tar.gz && tar xzf $S/OpenBLAS-0.3.34.tar.gz || fail $? "unpack"
for c in auto haswell; do
    rm -rf blis-$c && cp -R blis-2.0 blis-$c
    (cd blis-$c && ./configure --prefix=$R/blis-$c --disable-shared --enable-cblas -t no $c \
        && make -j$J && make install) > $L/build-blis-$c.log 2>&1 || fail $? "build blis-$c"
    grep -m1 -iE 'configuration.*(chosen|selected|auto)|automatically detected|config_name' $L/build-blis-$c.log >> $L/build-env.txt
    grep -E '^(CONFIG_NAME|CONFIG_LIST)' blis-$c/config.mk >> $L/build-env.txt 2>&1
done

#  OpenBLAS 0.3.34: target detected on this node, no threads
cd $R/src/OpenBLAS-0.3.34 || exit 1
(make -j$J USE_THREAD=0 USE_OPENMP=0 NO_SHARED=1 NOFORTRAN=1 NO_LAPACK=1 \
    && make PREFIX=$R/openblas NO_SHARED=1 install) > $L/build-openblas.log 2>&1 || fail $? "build openblas"
grep -hE '^(CORE|LIBCORE|ARCH|BINARY|DYNAMIC_ARCH|USE_THREAD|HAVE_AVX512VL|HAVE_AVX2)=' Makefile.conf >> $L/build-env.txt 2>&1

#  Eigen 5.0.1 (header only)
cd $A && rm -rf eigen-5.0.1 && tar xzf $S/eigen-5.0.1.tar.gz || fail $? "unpack eigen"

#  Programs: bench-V, verify-V, info programs.  C programs with gcc -O2 as
#  on heim; static libraries, -no-pie so that nm resolves kernel addresses.
B=$S/bench
MKL=/software/bwhpc/common/toolkit/Intel_OneAPI/mkl/2025.0/lib
AOCL=/software/bwhpc/common/lib/aocl/5.0.0/aocc
MKLLIBS="-Wl,--start-group $MKL/libmkl_intel_lp64.a $MKL/libmkl_sequential.a $MKL/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl"
inc() { case $1 in
    ulmblas)      echo "-I$A/ulmblas";;
    blis-*)       echo "-I$R/$1/include/blis";;
    openblas)     echo "-I$R/openblas/include";;
    mkl)          echo "-I$B/include-mkl -I$MKL/../include";;
    aocl)         echo "-I$AOCL/include";;
esac; }
lib() { case $1 in
    ulmblas)      echo "$A/ulmblas/libcblas.a -lstdc++ -lm";;
    blis-*)       echo "$R/$1/lib/libblis.a -lm -lpthread";;
    openblas)     echo "$R/openblas/lib/libopenblas.a -lm -lpthread";;
    mkl)          echo "$MKLLIBS";;
    aocl)         echo "$AOCL/lib/libblis.a -lm -lpthread";;
esac; }
: > $L/compile-commands.txt
cc_() { echo "$*" >> $L/compile-commands.txt; "$@" >> $L/build-programs.log 2>&1; }
for v in ulmblas blis-auto blis-haswell openblas mkl aocl; do
    for p in bench2:bench verify-large:verify; do
        cc_ gcc -O2 -no-pie $(inc $v) $B/${p%%:*}.c $(lib $v) -o $BIN/${p##*:}-$v || fail $? "${p##*:}-$v"
    done
done
for v in blis-auto blis-haswell aocl; do
    cc_ gcc -O2 -no-pie $(inc $v) $B/blis-info.c $(lib $v) -o $BIN/info-$v ||
    cc_ gcc -O2 -no-pie -DBLIS_OLD_CNTX $(inc $v) $B/blis-info.c $(lib $v) -o $BIN/info-$v || fail $? "info-$v"
done
cc_ gcc -O2 -no-pie $B/openblas-info.c $(lib openblas) -o $BIN/info-openblas || fail $? "info-openblas"
cc_ gcc -O2 -no-pie -I$MKL/../include $B/mkl-info.c $MKLLIBS -o $BIN/info-mkl || fail $? "info-mkl"
EF="-std=c++17 -O3 -DNDEBUG -march=native -I$A/eigen-5.0.1"
for p in bench-eigen:bench-eigen verify-eigen:verify-eigen eigen-info:info-eigen; do
    cc_ g++ $EF $B/${p%%:*}.cpp -o $BIN/${p##*:} || fail $? "${p##*:}"
done
cc_ gcc -O2 $B/peak.c -o $BIN/peak || fail $? "peak"

#  Which kernels are selected (run here, on a node of the target type)
cd $BIN
{
    for v in blis-auto blis-haswell aocl; do
        echo "== info-$v (BLIS_ARCH_DEBUG=1)"
        BLIS_ARCH_DEBUG=1 ./info-$v 2>&1
        a=$(./info-$v | awk '$1=="dgemm_ukr" {print $2}' | sed 's/^0x0*//')
        echo "dgemm_ukr symbol: $(nm info-$v | awk -v a=$a '{x=$1; sub(/^0+/, "", x)} x==a {print $3}' | head -3 | tr '\n' ' ')"
    done
    echo "== info-openblas"; ./info-openblas
    echo "== info-mkl (MKL_VERBOSE=1)"; MKL_VERBOSE=1 ./info-mkl 2>&1 | grep -vE '^$'
    echo "== info-eigen"; ./info-eigen
    echo "== ldd (all programs must be free of shared BLAS libraries)"
    for f in bench-* verify-*; do echo "$f: $(ldd $f | awk '{print $1}' | tr '\n' ' ')"; done
} > $L/kernels.txt 2>&1
sha256sum bench-* verify-* info-* peak > $L/programs.sha256
echo "build done $(date -Iseconds), failures: $(wc -l < $L/failures.txt)" >> $L/build-env.txt
