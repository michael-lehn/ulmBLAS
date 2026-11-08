# ?getrf / ?potrf of ulmBLAS (templates) vs. OpenBLAS and oneMKL (gnuplot >= 5.4).
#
#   gnuplot -e "R='getrf'; NAME='?GETRF'; D='results/<campaign>'; O='plots'" scripts/plot-core.gp
#
# D/summary-R.dat (scripts/summarize-core.py): column 1 n, then per
# precision s, d, c, z five columns: gflops ulmBLAS, OpenBLAS, oneMKL,
# speedup over OpenBLAS, over oneMKL (s: 2-6, d: 7-11, c: 12-16, z: 17-21).
# Writes O/R-core-vs-openblas.svg, O/R-core-vs-mkl.svg and
# O/R-core-gflops-{s,d,c,z}.svg.

C1 = "#2a78d6"; C2 = "#eb6834"; C3 = "#1baf7a"; C4 = "#eda100"
INK = "#52514e"
if (!exists("SUB")) SUB = "Xeon Platinum 8358 Ice Lake, one socket, 32 threads"
F = D."/summary-".R.".dat"

set terminal svg size 900,520 dynamic enhanced font "Helvetica,13" background rgb "white"
set border 3 lw 1.0 lc rgb INK
set tics nomirror tc rgb INK
set grid xtics ytics lc rgb "#e2e2e2" lw 1 dt 1
set key outside right top Left reverse box lc rgb "#dcdcdc" spacing 1.25 width 1
set xlabel "m = n"
set xrange [0:4050]
set xtics 0,500,4000
set yrange [0:*]

set output O."/".R."-core-vs-openblas.svg"
set title NAME.", ".SUB.": ulmBLAS / OpenBLAS"
set ylabel "time OpenBLAS / time ulmBLAS"
plot 1 w l lc rgb INK dt 2 lw 1.2 notitle, \
     F u 1:5  w lp lc rgb C1 lw 2 pt 7 ps 0.4 t "float", \
     F u 1:10 w lp lc rgb C2 lw 2 pt 7 ps 0.4 t "double", \
     F u 1:15 w lp lc rgb C3 lw 2 pt 7 ps 0.4 t "complex float", \
     F u 1:20 w lp lc rgb C4 lw 2 pt 7 ps 0.4 t "complex double"

set output O."/".R."-core-vs-mkl.svg"
set title NAME.", ".SUB.": ulmBLAS / oneMKL"
set ylabel "time oneMKL / time ulmBLAS"
plot 1 w l lc rgb INK dt 2 lw 1.2 notitle, \
     F u 1:6  w lp lc rgb C1 lw 2 pt 7 ps 0.4 t "float", \
     F u 1:11 w lp lc rgb C2 lw 2 pt 7 ps 0.4 t "double", \
     F u 1:16 w lp lc rgb C3 lw 2 pt 7 ps 0.4 t "complex float", \
     F u 1:21 w lp lc rgb C4 lw 2 pt 7 ps 0.4 t "complex double"

set ylabel "GFLOP/s"
array PN[4] = ["s", "d", "c", "z"]
array PT[4] = ["float", "double", "complex float", "complex double"]
do for [i=1:4] {
    b = 2 + 5*(i-1)
    set output O."/".R."-core-gflops-".PN[i].".svg"
    set title NAME." (".PT[i]."), ".SUB
    plot F u 1:b     w lp lc rgb C1 lw 2 pt 7 ps 0.4 t "ulmBLAS", \
         F u 1:(column(b+1)) w lp lc rgb C2 lw 2 pt 7 ps 0.4 t "OpenBLAS", \
         F u 1:(column(b+2)) w lp lc rgb C3 lw 2 pt 7 ps 0.4 t "oneMKL"
}
