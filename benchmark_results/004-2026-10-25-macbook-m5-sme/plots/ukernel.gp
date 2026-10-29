load "_style.gp"
set output "ukernel.svg"
set title "Micro kernels alone (packed panels in L1/L2, C tile in cache, alpha = beta = 1)\nApple M5 (MacBook Air), one performance core"
set xlabel "kc"
set ylabel "performance [GFLOP/s]"
set xrange [0:4200]
set yrange [0:580]
set xtics 0, 512, 4096
set mxtics 2
set key inside right center vertical nobox
plot PEAK_max with lines ls 9 title sprintf("SME2 FMOPA peak (measured, %.0f)", PEAK_max), \
     S."ukernel_kc.csv" using "case":"sme"      with linespoints ls 1 title "ulmBLAS SME2 32 x 16", \
     ""                 using "case":"blis_sme" with linespoints ls 2 title "BLIS PR #921 SME2 32 x 16", \
     ""                 using "case":"neon"     with linespoints ls 4 title "ulmBLAS NEON 6 x 8"
