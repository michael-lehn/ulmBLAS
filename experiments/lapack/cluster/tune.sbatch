#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=tune-%j.out
#
#  tune.sbatch SRC STAGE OUT: tuning experiments on an exclusive Ice Lake
#  node, sizes off the final grid (130, 270, 520, 1030, 2020, 3530).
#  Every line of OUT is a line of the benchmark program (+ the stage tag).
#
set -euo pipefail
SRC=$1; STAGE=$2; OUT=$3
source $SRC/experiments/lapack/cluster/env.sh
cd $SRC/experiments/lapack
[ -x build/bench-ulmblas ] || bash cluster/build.sh $SRC
B="${NUMA[*]}"
SIZES=${SIZES:-"130 270 520 1030 2020 3530"}
OPT="--min-time 0.2 --min-samples 5 --max-samples 100"
{
echo "# stage $STAGE  host $(hostname)  $(date -Is)  threads $OMP_NUM_THREADS"
run() {   # tag lib routine n args...
    local tag=$1 lib=$2 r=$3 n=$4; shift 4
    $B ./build/bench-$lib $r $n $n 1 $OPT "$@" | grep -v '^#' | sed "s/^/$tag /"
}
case $STAGE in
  lu)
    for n in $SIZES; do
        run ref openblas getrf $n
        run serial ulmblas getrf $n --threads 1
        for nb in 64 96 128 192 256; do
            for pt in 1 8 32; do
                run nb$nb-pt$pt ulmblas getrf $n --serial-n 0 --nb $nb --pt $pt --ib 32
            done
        done
    done ;;
  luib)
    for n in $SIZES; do
        for nb in ${NBS:-128 192}; do
            for ib in 8 16 32 64; do
                for pt in ${PTS:-8 16 32}; do
                    run nb$nb-ib$ib-pt$pt ulmblas getrf $n --serial-n 0 --nb $nb --pt $pt --ib $ib
                done
            done
        done
    done ;;
  chol)
    for n in $SIZES; do
        run ref openblas potrf $n
        run serial ulmblas potrf $n --threads 1
        for nb in 48 64 96 128 160 192 256 320; do
            run nb$nb ulmblas potrf $n --serial-n 0 --nb $nb
        done
    done ;;
  cholfine)
    for n in $SIZES; do
        for nb in 32 40 48 56 64 80 96 112 128 144 160 192; do
            run nb$nb ulmblas potrf $n --serial-n 0 --nb $nb
        done
    done ;;
  base)
    # base case sizes of the serial recursive kernels (one thread)
    for b in ${BASES:-8 16 32 64}; do
        make -s -j 16 "${MAKEVARS[@]}" BUILD=build-base$b \
             CPPFLAGS="-DULMLAPACK_POTRF_BASE=$b -DULMLAPACK_GETRF_BASE=$b" \
             build-base$b/bench-ulmblas > /dev/null
    done
    for r in getrf potrf; do
        for n in 16 24 32 48 64 96 128 192 256; do
            OMP_NUM_THREADS=1 run ref openblas $r $n
            OMP_NUM_THREADS=1 $B ./build/bench-ulmblas $r $n $n 1 $OPT --threads 1 \
                | grep -v '^#' | sed "s/^/default /"
            for b in ${BASES:-8 16 32 64}; do
                OMP_NUM_THREADS=1 $B ./build-base$b/bench-ulmblas $r $n $n 1 $OPT --threads 1 \
                    | grep -v '^#' | sed "s/^/base$b /"
            done
        done
    done ;;
  mid)
    # threads and tile size for medium sizes (LU: ib 16, panel team <= 8)
    for r in getrf potrf; do
        for n in 330 470 730 1030 1530; do
            run ref openblas $r $n
            for nb in 48 64 96 128; do
                for th in 8 16 24 32; do
                    run nb$nb-th$th ulmblas $r $n --serial-n 0 --nb $nb --threads $th --pt 8 --ib 16
                done
            done
        done
    done ;;
  defaults)
    # the tuned defaults against OpenBLAS, sizes off the benchmark grid
    for r in getrf potrf; do
        for n in 30 70 110 130 170 190 230 260 310 370 440 520 610 730 870 1030 1270 1530 2030 2530 3030 3530 3930; do
            run ref openblas $r $n
            run default ulmblas $r $n
        done
    done ;;
  overhead)
    # cost of the task graph: default, tasks forced, the same graph with
    # empty tasks, the serial algorithm; OpenBLAS as reference
    for r in getrf potrf; do
        for n in 50 100 150 200 300 400 600 800 1000 1500 2000 3000 4000; do
            run ref openblas $r $n
            run default ulmblas $r $n
            run tasks ulmblas $r $n --serial-n 0
            run empty ulmblas $r $n --serial-n 0 --empty 1
            run serial ulmblas $r $n --threads 1
        done
    done ;;
  profile)
    # profile of the task graph with the default parameters (diagnostic
    # build with -DULMLAPACK_CHECK_DEPS: per-task timestamps)
    make -s -j 16 "${MAKEVARS[@]}" BUILD=build-prof CPPFLAGS=-DULMLAPACK_CHECK_DEPS \
         build-prof/bench-ulmblas > /dev/null
    for r in getrf potrf; do
        for n in 200 300 400 600 800 1000 1500 2000 3000 4000; do
            $B ./build-prof/bench-ulmblas $r $n $n 1 $OPT --serial-n 0 \
                | grep -v '^# routine\|^# ulmlapack' | sed "s/^/profile /"
        done
    done ;;
  gendeps)
    # LU: GEMM tasks with 3 (default) or 2 dependencies (-DULMLAPACK_LU_GEMM_2DEPS);
    # profile builds report the task creation time
    for v in "" 2DEPS; do
        make -s -j 16 "${MAKEVARS[@]}" BUILD=build-g$v CPPFLAGS="${v:+-DULMLAPACK_LU_GEMM_2DEPS}" \
             build-g$v/bench-ulmblas > /dev/null
        make -s -j 16 "${MAKEVARS[@]}" BUILD=build-gp$v \
             CPPFLAGS="-DULMLAPACK_CHECK_DEPS ${v:+-DULMLAPACK_LU_GEMM_2DEPS}" \
             build-gp$v/bench-ulmblas > /dev/null
    done
    for rnd in 1 2 3; do
        for n in 1030 1530 2030 2530 3030 3530 3930; do
            for v in "" 2DEPS; do
                $B ./build-g$v/bench-ulmblas getrf $n $n 1 $OPT | grep -v '^#' | sed "s/^/deps${v:-3} /"
            done
        done
    done
    for n in 1030 2030 3030 3930; do
        for v in "" 2DEPS; do
            $B ./build-gp$v/bench-ulmblas getrf $n $n 1 $OPT | grep '^# profile' | sed "s/^/deps${v:-3} /"
        done
    done ;;
  small)
    # small matrices: serial algorithm vs. tasks vs. OpenBLAS, n = 10..400
    for r in getrf potrf; do
        for n in 10 20 30 40 60 80 110 130 170 220 270 330 400; do
            run ref openblas $r $n
            run serial ulmblas $r $n --threads 1
            for nb in 32 48 64 96; do
                for th in 4 8 16 32; do
                    run nb$nb-th$th ulmblas $r $n --serial-n 0 --nb $nb --threads $th --pt 8
                done
            done
        done
    done ;;
  scale)
    # 32 threads (one socket) vs. 64 threads (both sockets)
    for th in 32 64; do
        if [ $th = 64 ]; then B="numactl --interleave=all"; fi
        export OMP_NUM_THREADS=$th
        for r in getrf potrf; do
            for n in 1030 2020 3530; do
                run t$th openblas $r $n
                run t$th ulmblas $r $n
            done
        done
    done ;;
esac
} > $OUT 2>&1
echo done $(date -Is) >> $OUT
