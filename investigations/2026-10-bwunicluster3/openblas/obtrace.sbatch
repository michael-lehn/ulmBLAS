#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000mb
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#  Which OpenBLAS DGEMM kernel is executed (functional check, not a measurement)
W=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench; A=$W/icelake
module purge; module load compiler/gnu/14.2
cd $A/refs/openblas/lib || exit 1
echo "== objects of libopenblas.a with dgemm_kernel"; ar t libopenblas.a | grep -E '^dgemm_(kernel|beta|incopy|itcopy|oncopy|otcopy|small)' 
mkdir -p /tmp/ob.$$ && cd /tmp/ob.$$ && ar x $A/refs/openblas/lib/libopenblas.a dgemm_kernel.o 2>/dev/null; ls
echo "== instructions in dgemm_kernel.o"; objdump -d --no-show-raw-insn dgemm_kernel.o | awk '{print $2}' | grep -E '^v(fmadd|movddup|broadcast|unpck|mulpd|movupd)' | sort | uniq -c
echo "== symbols"; nm dgemm_kernel.o | head
echo "== gdb: break in the kernel during a 2048^3 DGEMM"
which gdb && gdb -q -batch -ex 'set breakpoint pending on' -ex 'break dgemm_kernel' -ex run -ex bt -ex 'info args' -ex 'p/x $pc' --args $A/bin/bench-openblas 1 1 2048 2048 2048 2>&1 | grep -vE '^\[|warning' | head -30
rm -rf /tmp/ob.$$
