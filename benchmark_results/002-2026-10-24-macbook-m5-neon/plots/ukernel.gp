load "_style.gp"
set output "ukernel.svg"
set title "Micro kernel alone (packed panels in cache, beta = 1)\nApple M5 (MacBook Air), one performance core"
set xlabel "kc"
set ylabel "performance [GFLOP/s]"
set xrange [0:3200]
set xtics 0, 512, 3072
set mxtics 2
set yrange [0:80]
set key inside right bottom
stats S."peak.csv" using (strcol("run") eq "B" ? column("gflops") : NaN) name "P" nooutput
plot P_median with lines ls 9 title sprintf("NEON FMA peak (measured, %.1f)", P_median), \
     S."ukernel.csv" using (strcol("variant") eq "neon" ? column("kc") : NaN):"gflops_median" with linespoints ls 4 title "NEON 6x8", \
     ""              using (strcol("variant") eq "ref"  ? column("kc") : NaN):"gflops_median" with linespoints ls 1 title "reference 4x4"
