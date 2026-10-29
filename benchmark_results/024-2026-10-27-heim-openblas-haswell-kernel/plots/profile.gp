load "_style.gp"
set terminal svg size 900,520 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "profile.svg"
set title "Where the time goes (baseline build, prof-hsw): GFLOP/s of variants of the same computation\nIntel Core i5-12500, one core, MC, KC, NC = 512, 256, 13824, medians of 3 rounds" font ",13"
F = D."prof.csv"
p(mode, c) = (strcol("mode") eq mode) ? column(c) : NaN
set xlabel "matrix size n (m = n = k)"
set ylabel "performance [GFLOP/s]"
set xrange [0:4200]
set yrange [48:64]
set key bottom right font ",11"
plot F using "n":(p("lib","gflops")) with linespoints ls 2 title "ulmBLAS (per-panel calls, n = 8)", \
     F using "n":(p("nopack","gflops")) with linespoints lc rgb "#7d4fc4" lw 1.6 pt 9 title "... without packing", \
     F using "n":(p("macro","gflops")) with linespoints ls 1 title "one kernel call per packed block", \
     F using "n":(p("macronopack","gflops")) with linespoints lc rgb "#1baf7a" lw 1.6 pt 13 title "... without packing", \
     F using "n":(p("cblas","gflops")) with linespoints ls 3 title "native OpenBLAS"
