#!/bin/bash
#  kernels-build.sbatch ARCH REV [TAG]: build ulmBLAS at git revision REV
#  once for every DGEMM micro kernel that the CPU type ARCH can execute (the
#  BLIS-derived kernels of ulmblas/level3/ukernel/blis/, the OpenBLAS-derived
#  kernel of ulmblas/level3/ukernel/openblas/ and ulmBLAS's own AVX/FMA
#  kernel), and link bench2.c and verify-large.c against each
#  library.  Programs: $W/ARCH/bin/bench-ulmblas-V, verify-ulmblas-V; logs:
#  $W/ARCH/kernels-TAG/ (TAG defaults to REV).  Run build.sbatch for ARCH
#  first (reference libraries, Eigen).  Optional: KERNELS (environment)
#  restricts the build to some kernels (e.g. KERNELS=openblas_skylakex_16x12);
#  MC, KC, NC (environment)
#  override the block sizes of the BLIS kernels, e.g. for a tuning
#  iteration: MC=336 sbatch ... kernels-build.sbatch icelake REV mc336
#
#      sbatch -p dev_cpu_il -J kbuild-icelake kernels-build.sbatch icelake REV
#      sbatch -p dev_cpu    -J kbuild-zen4    kernels-build.sbatch zen4 REV
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -u
ARCH=$1; REV=$2; TAG=${3:-$2}
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
A=$W/$ARCH; K=$A/kernels-$TAG; BIN=$A/bin; B=$W/src/bench
mkdir -p $K/logs $BIN && cd $K || exit 1
: > logs/failures.txt
fail() { echo "FAILED (exit $1): $2" | tee -a logs/failures.txt; }
J=$SLURM_CPUS_PER_TASK

module purge
module load compiler/gnu/14.2

git -C $W/src/ulmBLAS fetch -q origin || fail $? "git fetch"
FULL=$(git -C $W/src/ulmBLAS rev-parse $REV) || exit 1
rm -rf src && mkdir src && git -C $W/src/ulmBLAS archive $FULL | tar xf - -C src || exit 1

AVX512="-mavx -mfma -mavx512f"
case $ARCH in
    icelake|zen4) KERNELS=${KERNELS:-"avx blis_haswell_6x8 blis_haswell_8x6 blis_haswell_12x4 blis_haswell_4x12 blis_sandybridge_8x4 blis_penryn_4x4 blis_skx_16x14 blis_skx_16x12_l2 blis_zen4_32x6 blis_zen4_8x24 blis_zen5_8x24 openblas_skylakex_16x12"};;
    *) echo "unknown ARCH $ARCH"; exit 1;;
esac
BS=""
[ -n "${MC:-}" ] && BS="$BS -DULMBLAS_DGEMM_MC=$MC"
[ -n "${KC:-}" ] && BS="$BS -DULMBLAS_DGEMM_KC=$KC"
[ -n "${NC:-}" ] && BS="$BS -DULMBLAS_DGEMM_NC=$NC"
flags() { case $1 in
    avx)                  echo "-mavx -mfma";;
    blis_haswell_*)       echo "-mavx -mfma -DULMBLAS_DGEMM_UKERNEL_$1$BS";;
    blis_sandybridge_8x4) echo "-mavx -DULMBLAS_DGEMM_UKERNEL_$1$BS";;
    blis_penryn_4x4)      echo "-msse3 -mfpmath=sse -DULMBLAS_DGEMM_UKERNEL_$1$BS";;
    *)                    echo "$AVX512 -DULMBLAS_DGEMM_UKERNEL_$1$BS";;
esac; }
{
    echo "host $(hostname) job $SLURM_JOB_ID date $(date -Iseconds)"
    echo "ulmBLAS $FULL tag $TAG; MC=${MC:-} KC=${KC:-} NC=${NC:-}"
    grep -m1 'model name' /proc/cpuinfo; gcc --version | head -1
} > logs/build-env.txt
: > logs/compile-commands.txt
for k in $KERNELS; do
    v=${k#blis_}; [ $k = avx ] && v=avx; [ $k = openblas_skylakex_16x12 ] && v=openblas-skylakex_16x12
    rm -rf lib-$v && cp -R src lib-$v
    (cd lib-$v && make -j$J cblas CXX=g++ ARCHFLAGS="`flags $k`") > logs/build-$v.log 2>&1 || { fail $? "build $v"; continue; }
    echo "$v: $(cat lib-$v/interfaces/blas/C/build.cblas/compile.cmd)" >> logs/compile-commands.txt
    gcc -O2 -no-pie -Ilib-$v $B/bench2.c lib-$v/libcblas.a -lstdc++ -lm -o $BIN/bench-ulmblas-$v-$TAG || fail $? "link bench $v"
    gcc -O2 -no-pie -Ilib-$v $B/verify-large.c lib-$v/libcblas.a -lstdc++ -lm -o $BIN/verify-ulmblas-$v-$TAG || fail $? "link verify $v"
    rm -rf lib-$v/interfaces/blas/C/build.cblas lib-$v/interfaces/blas/F77/build.f77
done
(cd $BIN && sha256sum bench-ulmblas-*-$TAG verify-ulmblas-*-$TAG) > logs/programs.sha256
echo "build done $(date -Iseconds), failures: $(wc -l < logs/failures.txt)" >> logs/build-env.txt
