#
#  Step 10: GFLOP/s of steps 01 to 10, cuBLAS and MAGMA for square DGEMM,
#  from the saved output of bench_gemm (make plot, or gnuplot plot.gp in
#  this directory).  Columns of the log: impl m n k median_ms min_ms max_ms
#  GFLOP/s /cuBLAS; only rows with m = n = k are plotted.
#
data = "results/2026-10-29-a100/bench-square.log"

set terminal svg size 900,680 dynamic enhanced font "Helvetica,14" \
    background rgb "white"
set output "bench.svg"

set title "DGEMM C = AB + C, m = n = k, one NVIDIA A100 80GB PCIe\n" \
          . "median of 10 calls, GPU time between CUDA events"
set xlabel "matrix size n (m = n = k)"
set ylabel "GFLOP/s"
set xrange [0:4100]
set xtics 500
set yrange [0:*]
set pointsize 0.6
set grid xtics ytics lc rgb "#dcdcdc"
set key below maxcols 2 box

#  the rows of one implementation with m = n = k (the rows of the
#  implementations alternate in the log, so they are selected with awk
#  to get connected lines)
rows(name) = sprintf("< awk '$1 == \"%s\" && $2 == $3 && $3 == $4' %s", \
                     name, data)

plot rows("cuBLAS") using 2:8 with linespoints lw 2 pt 7 \
          lc rgb "#2a78d6" title "cuBLAS (cublasDgemm)", \
     rows("magma") using 2:8 with linespoints lw 2 dt 2 pt 6 ps 0.9 \
          lc rgb "#8a8984" title "MAGMA magma\\_dgemm (calls cublasDgemm)", \
     rows("magmablas") using 2:8 with linespoints lw 2 pt 5 \
          lc rgb "#008300" title "MAGMA magmablas\\_dgemm (own kernel)", \
     rows("step01") using 2:8 with linespoints lw 2 pt 9 \
          lc rgb "#e34948" title "cULMblas step 01 (naive)", \
     rows("step02") using 2:8 with linespoints lw 2 pt 11 \
          lc rgb "#eda100" title "cULMblas step 02 (4 elements per thread)", \
     rows("step03") using 2:8 with linespoints lw 2 pt 13 \
          lc rgb "#6250d6" title "cULMblas step 03 (8 x 4 block per thread)", \
     rows("step04") using 2:8 with linespoints lw 2 pt 15 \
          lc rgb "#d6409f" title "cULMblas step 04 (4 x 4, passes of 4 values of l)", \
     rows("step05") using 2:8 with linespoints lw 2 pt 4 \
          lc rgb "#1a1a1a" title "cULMblas step 05 (panels in shared memory)", \
     rows("step06") using 2:8 with linespoints lw 2 pt 8 \
          lc rgb "#00a3a3" title "cULMblas step 06 (Tensor Cores, a warp per 8 x 8 block)", \
     rows("step07") using 2:8 with linespoints lw 2 pt 12 \
          lc rgb "#c55a00" title "cULMblas step 07 (Tensor Cores, a warp per 32 x 32 block)", \
     rows("step08") using 2:8 with linespoints lw 2 pt 10 \
          lc rgb "#7a1fa2" title "cULMblas step 08 (Tensor Cores, panels in shared memory)", \
     rows("step09") using 2:8 with linespoints lw 2 pt 14 \
          lc rgb "#b8860b" title "cULMblas step 09 (cp.async, two panel buffers)", \
     rows("step10") using 2:8 with linespoints lw 2 pt 6 \
          lc rgb "#d62728" title "cULMblas step 10 (mma.sync, 16-byte operand loads)"
