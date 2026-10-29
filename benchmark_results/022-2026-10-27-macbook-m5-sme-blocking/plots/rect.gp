load "_style.gp"
set terminal svg size 1500,1000 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "rect.svg"
set multiplot layout 2,3 title "DGEMM, one dimension varied at a time, Apple M5, one core; medians of 3 interleaved rounds (rect, disturbed shapes from rect2)\nproduction MC, KC, NC = 384, 1024, 1024 (thick); KC = 1088 and KC = 2048: diagnostics outside the fixed-KC experiment (dashed)" font ",14"
set ylabel "performance [GFLOP/s]"
set yrange [395:470]
F = D."rect-used.csv"
s(v, c) = (strcol("config") eq v && c) ? column("gflops") : NaN
array V[5] = ["b384_1024_4096", "b352_1024_1024", "b384_1088_1024", "b384_2048_1024", "b384_1024_1024"]
array T[5] = ["NC 4096", "MC 352", "KC 1088 (diagnostic)", "KC 2048 (diagnostic)", "production"]
array DT[5] = [1, 1, 2, 2, 1]
array LW[5] = [1.6, 1.6, 1.8, 1.8, 3.0]
array LC[5] = ["#e8a33d", "#c4553a", "#7d4fc4", "#1baf7a", "#2a78d6"]
array PT[5] = [5, 9, 6, 4, 7]
xm = "column(\"m\")"; xn = "column(\"n\")"; xk = "column(\"k\")"

# first row: around 1024
set xrange [990:1210]
set xtics 1000, 50, 1200
set arrow 1 from 1024, graph 0 to 1024, graph 1 nohead ls 9
set xlabel "k  (m = n = 1000)"
set key bottom right font ",11" spacing 1.0
plot for [i=1:5] F using "k":(s(V[i], column("m")==1000 && column("n")==1000)) with linespoints lw LW[i] lc rgb LC[i] dt DT[i] pt PT[i] ps 0.6 title T[i]
unset key
set xlabel "n  (m = k = 1000)"
plot for [i=1:5] F using "n":(s(V[i], column("m")==1000 && column("k")==1000)) with linespoints lw LW[i] lc rgb LC[i] dt DT[i] pt PT[i] ps 0.6 notitle
set xlabel "m  (n = k = 1000); MC = 384: 1152 = 3 MC"
set arrow 1 from 1152, graph 0 to 1152, graph 1 nohead ls 9
plot for [i=1:5] F using "m":(s(V[i], column("n")==1000 && column("k")==1000)) with linespoints lw LW[i] lc rgb LC[i] dt DT[i] pt PT[i] ps 0.6 notitle

# second row: around 2048
set xrange [1990:2110]
set xtics 2000, 20, 2100
set arrow 1 from 2048, graph 0 to 2048, graph 1 nohead ls 9
set xlabel "k  (m = n = 1000)"
plot for [i=1:5] F using "k":(s(V[i], column("m")==1000 && column("n")==1000)) with linespoints lw LW[i] lc rgb LC[i] dt DT[i] pt PT[i] ps 0.6 notitle
set xlabel "n  (m = k = 1000)"
plot for [i=1:5] F using "n":(s(V[i], column("m")==1000 && column("k")==1000)) with linespoints lw LW[i] lc rgb LC[i] dt DT[i] pt PT[i] ps 0.6 notitle
set xlabel "k  (m = n = 2000)"
set yrange [420:495]
plot for [i=1:5] F using "k":(s(V[i], column("m")==2000 && column("n")==2000)) with linespoints lw LW[i] lc rgb LC[i] dt DT[i] pt PT[i] ps 0.6 notitle
unset multiplot
