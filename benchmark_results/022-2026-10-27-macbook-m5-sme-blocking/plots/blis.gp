load "_style.gp"
#  BLIS experiment: medians of rounds 1 to 3 (lines) and every program run
#  of these rounds (open points), so that the two states of the 2240
#  configurations are visible; rounds 4 to 7 are not used (README.md).
set terminal svg size 1150,900 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "blis.svg"
set multiplot layout 2,2 title "DGEMM: BLIS (PR #921, m4sme\\_p) and ulmBLAS, Apple M5, one core; lines: medians of 3 interleaved rounds, points: every run\ndotted: 1024, 2048 (ulmBLAS KC = NC), 2240 (BLIS KC), 3072 (BLIS NC)" font ",14"
set ylabel "performance [GFLOP/s]"
set yrange [280:500]
F = D."blis.csv"
RUNS = "../data/runs.csv"
s(v, c) = (strcol("config") eq v && c) ? column("gflops") : NaN
r(v, c) = (strcol("experiment") eq "blis" && strcol("used") eq "1" && strcol("config") eq v && c) ? column("gflops") : NaN
sq = 'column("m")==column("n") && column("n")==column("k") && column("n")>=N0 && column("n")<=N1'
array V[3] = ["b608_2240_3072", "blis-sme", "b384_1024_1024"]
array T[3] = ["ulmBLAS with MC, KC, NC = 608, 2240, 3072", "BLIS", "ulmBLAS production"]
array LS[3] = [2, 7, 1]
array PR[3] = [6, 4, 6]
do for [x in "1024 2048 2240 3072"] { set arrow from int(x), graph 0 to int(x), graph 1 nohead ls 9 }

set xlabel "matrix size n (square)"
set xrange [990:1110]
N0 = 990; N1 = 1110
set xtics 1000, 25, 1100
set key bottom right font ",11" spacing 1.0
plot for [i=1:3] F using "n":(s(V[i], @sq)) with linespoints ls LS[i] title T[i], \
     for [i=1:3] RUNS using "n":(r(V[i], @sq)) with points ls LS[i] pt PR[i] ps 0.8 notitle
unset key
set xrange [1990:2310]
N0 = 1990; N1 = 2310
set xtics 2000, 50, 2300
plot for [i=1:3] F using "n":(s(V[i], @sq)) with linespoints ls LS[i], \
     for [i=1:3] RUNS using "n":(r(V[i], @sq)) with points ls LS[i] pt PR[i] ps 0.8
set xrange [3060:3110]
N0 = 3060; N1 = 3110
set xtics 3060, 10, 3110
plot for [i=1:3] F using "n":(s(V[i], @sq)) with linespoints ls LS[i], \
     for [i=1:3] RUNS using "n":(r(V[i], @sq)) with points ls LS[i] pt PR[i] ps 0.8
unset arrow
set arrow from 2240, graph 0 to 2240, graph 1 nohead ls 9
set xrange [1980:2270]
set xtics 2000, 50, 2250
set xlabel "k  (m = n = 2000)"
c2 = 'column("m")==2000 && column("n")==2000'
plot for [i=1:3] F using "k":(s(V[i], @c2)) with linespoints ls LS[i], \
     for [i=1:3] RUNS using "k":(r(V[i], @c2)) with points ls LS[i] pt PR[i] ps 0.8
unset multiplot
