load "_style.gp"
set output "breakdown.svg"
set title "Where the time of a DGEMM call goes (ulmBLAS d25e3eb, instrumented driver, run P1)\nApple M5 (MacBook Air), one performance core"
set style histogram rowstacked
set key outside right center box opaque spacing 1.2 width 1
set xlabel "matrix shape m x n x k"
set ylabel "fraction of the time of a call"
set yrange [0.70:1.0]
set ytics 0.05
plot S."breakdown.csv" using "kernel":xtic(1) lc rgb "#08519c" title "micro kernel at its own speed", \
     "" using "macro" lc rgb "#6baed6" title "macro kernel (C, edges)", \
     "" using "packA" lc rgb "#cb181d" title "packing A", \
     "" using "packB" lc rgb "#fc9272" title "packing B"
