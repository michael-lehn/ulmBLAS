load "_style.gp"
set output "before_after.svg"
set title "CBLAS library before (d25e3eb) and after the packing change (run P4)\nApple M5 (MacBook Air), one performance core, median of 5 rounds"
set style histogram clustered gap 1
set key inside top left horizontal maxrows 1 samplen 1.5 font ",11"
set xlabel "routine, matrix shape m x n x k"
set ylabel "GFLOP/s relative to d25e3eb"
set yrange [0.96:1.14]
set ytics 0.02
set xtics rotate by -45 font ",10"
plot S."before_after.csv" using "p4_over_master":xtic(sprintf("%s %s", strcol(1), strcol(2))) lc rgb "#6baed6" title "4 panels packed together", \
     "" using "p8_over_master" lc rgb "#08519c" title "8 panels packed together", \
     1 with lines ls 9 notitle
