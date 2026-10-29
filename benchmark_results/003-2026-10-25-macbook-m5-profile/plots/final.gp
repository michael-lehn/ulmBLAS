load "_style.gp"
set output "final.svg"
set title "Packing change (panels of A packed together) relative to d25e3eb, CBLAS library\nApple M5 (one performance core, median of 5 rounds) and Intel Core i5-12500 (SSE3, AVX/FMA; median of 3 rounds)"
set style histogram clustered gap 1
set key inside top left horizontal maxrows 1 samplen 1.5 font ",11"
set xlabel "routine, matrix shape m x n x k"
set ylabel "GFLOP/s relative to d25e3eb"
set yrange [0.96:1.14]
set ytics 0.02
set xtics rotate by -45 font ",10"
plot S."final_all.csv" using "m5":xtic(sprintf("%s %s", strcol(1), strcol(2))) lc rgb "#08519c" title "Apple M5 (NEON)", \
     "" using "heim_avx" lc rgb "#cb181d" title "i5-12500 AVX/FMA", \
     "" using "heim_sse" lc rgb "#fc9272" title "i5-12500 SSE3", \
     1 with lines ls 9 notitle
