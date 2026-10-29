#!/bin/bash
#  build-blis-extra.sbatch ARCH: native BLIS 2.0 with the configurations
#  sandybridge and penryn (the native references for the vendored kernels
#  sandybridge_8x4 and penryn_4x4), static, no threads, CBLAS, built as in
#  build.sbatch; programs $W/ARCH/bin/bench-blis-CONFIG, verify-blis-CONFIG,
#  info-blis-CONFIG.  Run build.sbatch for ARCH first.
#
#      sbatch -p cpu_il -J blisx-icelake build-blis-extra.sbatch icelake
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -u
ARCH=$1
W=${W:-/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench}
S=$W/src; A=$W/$ARCH; R=$A/refs; L=$A/logs; BIN=$A/bin; B=$S/bench
export TMPDIR=$A/tmp
module purge
module load compiler/gnu/14.2
cd $R/src || exit 1
for c in sandybridge penryn; do
    rm -rf blis-$c && cp -R blis-2.0 blis-$c
    (cd blis-$c && ./configure --prefix=$R/blis-$c --disable-shared --enable-cblas -t no $c \
        && make -j$SLURM_CPUS_PER_TASK && make install) > $L/build-blis-$c.log 2>&1 \
        || echo "FAILED build blis-$c" >> $L/failures.txt
    gcc -O2 -no-pie -I$R/blis-$c/include/blis $B/bench2.c $R/blis-$c/lib/libblis.a -lm -lpthread -o $BIN/bench-blis-$c
    gcc -O2 -no-pie -I$R/blis-$c/include/blis $B/verify-large.c $R/blis-$c/lib/libblis.a -lm -lpthread -o $BIN/verify-blis-$c
    gcc -O2 -no-pie -I$R/blis-$c/include/blis $B/blis-info.c $R/blis-$c/lib/libblis.a -lm -lpthread -o $BIN/info-blis-$c
    (cd $BIN && echo "== info-blis-$c (BLIS_ARCH_DEBUG=1)" && BLIS_ARCH_DEBUG=1 ./info-blis-$c 2>&1 \
        && a=$(./info-blis-$c | awk '$1=="dgemm_ukr" {print $2}' | sed 's/^0x0*//') \
        && echo "dgemm_ukr symbol: $(nm info-blis-$c | awk -v a=$a '{x=$1; sub(/^0+/, "", x)} x==a {print $3}' | head -1)") >> $L/kernels-extra.txt 2>&1
done
(cd $BIN && sha256sum bench-blis-sandybridge bench-blis-penryn verify-blis-sandybridge verify-blis-penryn) >> $L/programs-extra.sha256
