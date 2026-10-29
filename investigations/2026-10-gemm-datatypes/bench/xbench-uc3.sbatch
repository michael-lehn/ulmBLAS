#!/bin/bash
#  xbench-uc3.sbatch PREC ARCH TARBALL NAME=ARCHFLAGS ...: single-threaded
#  ?GEMM (PREC = s, d, c, z) on one exclusive compute node of bwUniCluster
#  3.0, m = n = k = 50, 100, ..., 4000: ulmBLAS (source tree packed in
#  TARBALL, sha256 recorded) built once per NAME with ARCHFLAGS, against the
#  native libraries built by build.sbatch for ARCH (OpenBLAS 0.3.34, BLIS
#  2.0 auto and haswell, oneMKL 2025.0 sequential, AOCL-BLAS 5.0).  Method of
#  full-bench.sbatch (campaign 023) with benchx.c (in BENCHDIR, default
#  $W/up/xbench): ROUNDS rounds (default 3) with a rotated order of the variants,
#  variants in ONCE in the first round only, SAMPLES samples (default 3) of
#  at least 0.2 s after a warm-up of at least 0.3 s, median; every result
#  checked (400 elements of C against sums in long double); one process at
#  a time, bound to core CPU and its NUMA node; one thread in every library.
#  Records which kernels the native libraries use for PREC (kernel-info.c,
#  MKL_VERBOSE).  Results: $W/ARCH/xrun-JOBID/{res,logs}.
#
#      ONCE=ulmblas-ref sbatch -p cpu_il -J xbench-z-icelake xbench-uc3.sbatch \
#          z icelake $W/up/zt.tgz ulmblas-ref="-mavx -mfma" ...
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=32000mb
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -u
module purge
module load compiler/gnu/14.2
PREC=$1; ARCH=$2; TARBALL=$3; shift 3
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
#  benchx.c and kernel-info.c (Slurm runs a copy of this script elsewhere)
D=${BENCHDIR:-$W/up/xbench}
CPU=${CPU:-3}; ROUNDS=${ROUNDS:-3}; SAMPLES=${SAMPLES:-3}; ONCE=${ONCE:-}
SIZES=${SIZES:-$(seq 50 50 4000)}
R=$W/$ARCH/refs
MKL=/software/bwhpc/common/toolkit/Intel_OneAPI/mkl/2025.0/lib
AOCL=/software/bwhpc/common/lib/aocl/5.0.0/aocc
MKLLIBS="-Wl,--start-group $MKL/libmkl_intel_lp64.a $MKL/libmkl_sequential.a $MKL/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl"
O=$W/$ARCH/xrun-$SLURM_JOB_ID; L=$O/logs
mkdir -p $O/res $L && cd $O || exit 1
: > $L/failures.txt
fail() { echo "FAILED (exit $1): $2" | tee -a $L/failures.txt; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       MKL_DYNAMIC=FALSE BLIS_NUM_THREADS=1
J=$(nproc)

#  ulmBLAS variants
mkdir src && tar xzf $TARBALL -C src || fail $? "unpack"
VARIANTS=""
: > $L/compile-commands.txt
while [ $# -gt 0 ]; do
    v=${1%%=*}; f=${1#*=}; shift
    rm -rf v-$v && cp -r src v-$v
    (cd v-$v && make -j$J cblas CXX=g++ ARCHFLAGS="$f") > $L/build-$v.log 2>&1 || fail $? "build $v"
    echo "$v: $(cat v-$v/interfaces/blas/C/build.cblas/compile.cmd)" >> $L/compile-commands.txt
    nm -C v-$v/libcblas.a 2>/dev/null | grep -E ' [TtWw] .*(bli_[sdcz]gemm|[sdcz]gemm_kernel)' | sort -u | sed "s/^/$v: symbol /" >> $L/compile-commands.txt
    gcc -O2 -DPREC=$PREC -Iv-$v $D/benchx.c v-$v/libcblas.a -lstdc++ -lm -o bench-$v || fail $? "link $v"
    rm -rf v-$v
    VARIANTS="$VARIANTS $v"
done
rm -rf src
#  native libraries
gcc -O2 -DPREC=$PREC -I$R/openblas/include $D/benchx.c $R/openblas/lib/libopenblas.a -lm -o bench-openblas || fail $? "link openblas"
gcc -O2 -DPREC=$PREC -I$R/blis-auto/include/blis $D/benchx.c $R/blis-auto/lib/libblis.a -lpthread -lm -o bench-blis-auto || fail $? "link blis-auto"
gcc -O2 -DPREC=$PREC -I$R/blis-haswell/include/blis $D/benchx.c $R/blis-haswell/lib/libblis.a -lpthread -lm -o bench-blis-haswell || fail $? "link blis-haswell"
gcc -O2 -no-pie -DPREC=$PREC -I$W/src/bench/include-mkl -I$MKL/../include $D/benchx.c $MKLLIBS -o bench-mkl || fail $? "link mkl"
gcc -O2 -DPREC=$PREC -I$AOCL/include $D/benchx.c $AOCL/lib/libblis.a -lm -lpthread -o bench-aocl || fail $? "link aocl"
VARIANTS="$VARIANTS openblas blis-auto blis-haswell mkl aocl"
sha256sum $TARBALL bench-* > $L/programs.sha256

ALLOWED=$(numactl --show | awk '$1=="physcpubind:" {$1=""; print}')
NODE=$(lscpu -p=CPU,NODE | awk -F, -v c=$CPU '$1==c {print $2}')
PIN="numactl --physcpubind=$CPU --membind=$NODE"

#  kernels of the native libraries for PREC
{
    OBP=; nm $R/openblas/lib/libopenblas.a 2>/dev/null | grep -qE " [DBC] ${PREC}gemm_r$" && OBP=-DOB_PARAMS; gcc -O2 -DPREC=$PREC -DOPENBLAS $OBP $D/kernel-info.c $R/openblas/lib/libopenblas.a -lm -o info-openblas && $PIN ./info-openblas
    for b in blis-auto blis-haswell; do
        echo "== $b"
        gcc -O2 -DPREC=$PREC -DBLIS -I$R/$b/include/blis $D/kernel-info.c $R/$b/lib/libblis.a -lpthread -lm -o info-$b && $PIN ./info-$b
    done
    echo "== aocl"
    gcc -O2 -DPREC=$PREC -DBLIS -DBLIS_0X -I$AOCL/include $D/kernel-info.c $AOCL/lib/libblis.a -lpthread -lm -o info-aocl && $PIN ./info-aocl
    echo "== mkl (MKL_VERBOSE=1, n = 1000)"
    MKL_VERBOSE=1 $PIN ./bench-mkl 1 1000 2>&1 | grep -E 'MKL_VERBOSE' | head -3
} > $L/kernel-identity.txt 2>&1

{
    echo "host $(hostname) job $SLURM_JOB_ID partition $SLURM_JOB_PARTITION date $(date -Iseconds)"
    echo "cpu $CPU numa node $NODE: $PIN"
    grep -m1 'model name' /proc/cpuinfo; grep -m1 microcode /proc/cpuinfo; uname -r
    echo "governor $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor) driver $(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_driver)"
    echo "no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null) boost $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null) smt $(cat /sys/devices/system/cpu/smt/control)"
    echo "allocation: $SLURM_JOB_CPUS_PER_NODE CPUs; exclusive: $(scontrol show job $SLURM_JOB_ID | grep -o 'OverSubscribe=[A-Z]*')"
    echo "prec $PREC; variants:$VARIANTS; once: $ONCE; rounds $ROUNDS; samples $SAMPLES"
    echo "sizes $(echo $SIZES | tr '\n' ' ')"
    echo "load $(cat /proc/loadavg); processes of other users: $(ps -eo user= | grep -vcE "^(root|$USER)$")"
    gcc --version | head -1
    echo "== environment"; env | grep -E '^(OMP|OPENBLAS|MKL|BLIS|GOMP|KMP)_' | sort
} > $L/system.txt 2>&1

rot() { r=$1; shift; i=1; while [ $i -lt $r ]; do x=$1; shift; set -- "$@" $x; i=$((i+1)); done; echo "$@"; }
once() { case " $ONCE " in *" $1 "*) return 0;; esac; return 1; }
echo "time,variant,round,load1,cpu_MHz" > $L/conditions.csv
echo "start $(date -Iseconds)" >> $L/run.log
for r in $(seq 1 $ROUNDS); do
    for v in $(rot $r $VARIANTS); do
        if [ $r -gt 1 ] && once $v; then continue; fi
        echo "$(date +%s),$v,$r,$(cut -d' ' -f1 /proc/loadavg),$(awk -v c=$CPU '$1=="processor"{p=$3} $1=="cpu" && $2=="MHz" && p==c {print $4}' /proc/cpuinfo)" >> $L/conditions.csv
        $PIN ./bench-$v $SAMPLES $SIZES > res/$PREC-$v-$r.txt || fail $? "$v $r"
        echo "$(date -Iseconds) round $r $v done" >> $L/run.log
    done
done
echo "wrong results: $(cat res/$PREC-* | grep -c WRONG)" >> $L/run.log
echo "end $(date -Iseconds)" >> $L/run.log
if [ -s $L/failures.txt ]; then cat $L/failures.txt >> $L/run.log; else echo "overall: all commands succeeded" >> $L/run.log; fi
