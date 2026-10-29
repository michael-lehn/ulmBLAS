#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000mb
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
#  Arguments (m, n, k) of the OpenBLAS DGEMM kernel calls for 2048^3 and
#  4000^3 (functional check under gdb, not a measurement)
A=/pfs/work9/workspace/scratch/ul_mlehn-ulmblas-bench/icelake
cat > args.gdb <<'G'
set pagination off
break dgemm_kernel
commands 1
silent
printf "m %ld n %ld k %ld\n", $rdi, $rsi, $rdx
continue
end
run
G
for n in 2048 4000; do
  echo "== n = $n: kernel calls of the first dgemm (count, m, n, k)"
  gdb -q -batch -x args.gdb --args $A/bin/bench-openblas 1 1 $n $n $n 2>&1 | grep '^m ' > calls-$n.txt
  echo "total calls (3 dgemm calls: checked, warm-up, sample): $(wc -l < calls-$n.txt)"
  sort calls-$n.txt | uniq -c | sort -rn | head -12
done
