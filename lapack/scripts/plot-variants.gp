# Copying variants vs. in place (gnuplot >= 5.4).
#
#   gnuplot -e "R='potrf'; NAME='DPOTRF'; D='results/<campaign>'; O='plots'" scripts/plot-variants.gp
#
# D/summary-R-variants.dat (scripts/summarize-strides.py): groups g = 0..8
# (inplace, colmajor, tiled) x (incRow 1, 2, 42); group g has
# t in column 4+5g, gflops in 5+5g; t_inplace/t_variant in column 49+g.
# Writes O/R-variants-gain.svg and O/R-variants-gflops.svg.
#
# Color = row stride (slots 1, 3, 4 as in plot-strides.gp), line and point
# type = variant.

U1  = "#2a78d6"
OB  = "#eb6834"
U2  = "#1baf7a"
U42 = "#eda100"
INK = "#52514e"

if (!exists("SUB")) SUB = "Xeon Platinum 8358 Ice Lake, one socket, 32 threads"

set terminal svg size 900,520 dynamic enhanced font "Helvetica,13" background rgb "white"
set border 3 lw 1.0 lc rgb INK
set tics nomirror tc rgb INK
set grid xtics ytics lc rgb "#e2e2e2" lw 1 dt 1
set key top left Left reverse box lc rgb "#dcdcdc" opaque spacing 1.25 width 1
set xlabel "m = n"
set xrange [0:4050]
set xtics 0,500,4000
F = D."/summary-".R."-variants.dat"

# gain of copying: time in place / time of the variant, same row stride
set output O."/".R."-variants-gain.svg"
set title NAME.", ".SUB.": copy first, or work in place?"
set ylabel "time in place / time with copy (same incRow)"
# headroom above the curves for the legend (top left)
set yrange [0:(R eq "getrf" ? 6 : 5)]
if (R eq "potrf") {
plot 1 w l lc rgb INK dt 2 lw 1.2 t "equal time", \
     F u 1:52 w lp lc rgb U1  lw 2 pt 7  ps 0.5 t "colmajor copy, incRow = 1", \
     F u 1:53 w lp lc rgb U2  lw 2 pt 7  ps 0.5 t "colmajor copy, incRow = 2", \
     F u 1:54 w lp lc rgb U42 lw 2 pt 7  ps 0.5 t "colmajor copy, incRow = 42", \
     F u 1:55 w lp lc rgb U1  lw 2 dt 3 pt 4 ps 0.6 t "tiled copy, incRow = 1", \
     F u 1:56 w lp lc rgb U2  lw 2 dt 3 pt 4 ps 0.6 t "tiled copy, incRow = 2", \
     F u 1:57 w lp lc rgb U42 lw 2 dt 3 pt 4 ps 0.6 t "tiled copy, incRow = 42"
} else {
plot 1 w l lc rgb INK dt 2 lw 1.2 t "equal time", \
     F u 1:52 w lp lc rgb U1  lw 2 pt 7  ps 0.5 t "colmajor copy, incRow = 1", \
     F u 1:53 w lp lc rgb U2  lw 2 pt 7  ps 0.5 t "colmajor copy, incRow = 2", \
     F u 1:54 w lp lc rgb U42 lw 2 pt 7  ps 0.5 t "colmajor copy, incRow = 42"
}

# GFLOP/s of the variants at incRow = 1 and 42, and OpenBLAS
set output O."/".R."-variants-gflops.svg"
set title NAME.", ".SUB.": GFLOP/s, in place vs. copy"
set ylabel "GFLOP/s  (".(R eq "getrf" ? "2/3 n^3" : "1/3 n^3")." / time, copies included)"
set yrange [0:*]
set key top left
if (R eq "potrf") {
plot F u 1:5  w lp lc rgb U1  lw 2 pt 7  ps 0.5 t "in place, incRow = 1", \
     F u 1:20 w lp lc rgb U1  lw 2 dt 2 pt 6 ps 0.5 t "colmajor copy, incRow = 1", \
     F u 1:35 w lp lc rgb U1  lw 2 dt 3 pt 4 ps 0.6 t "tiled copy, incRow = 1", \
     F u 1:15 w lp lc rgb U42 lw 2 pt 7  ps 0.5 t "in place, incRow = 42", \
     F u 1:30 w lp lc rgb U42 lw 2 dt 2 pt 6 ps 0.5 t "colmajor copy, incRow = 42", \
     F u 1:45 w lp lc rgb U42 lw 2 dt 3 pt 4 ps 0.6 t "tiled copy, incRow = 42", \
     F u 1:3  w lp lc rgb OB  lw 2 pt 5  ps 0.5 t "OpenBLAS 0.3.34 (column major)"
} else {
plot F u 1:5  w lp lc rgb U1  lw 2 pt 7  ps 0.5 t "in place, incRow = 1", \
     F u 1:20 w lp lc rgb U1  lw 2 dt 2 pt 6 ps 0.5 t "colmajor copy, incRow = 1", \
     F u 1:15 w lp lc rgb U42 lw 2 pt 7  ps 0.5 t "in place, incRow = 42", \
     F u 1:30 w lp lc rgb U42 lw 2 dt 2 pt 6 ps 0.5 t "colmajor copy, incRow = 42", \
     F u 1:3  w lp lc rgb OB  lw 2 pt 5  ps 0.5 t "OpenBLAS 0.3.34 (column major)"
}
