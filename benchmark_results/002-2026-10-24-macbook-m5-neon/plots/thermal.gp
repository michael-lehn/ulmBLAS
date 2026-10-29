load "_style.gp"
set output "thermal.svg"
set title "Thermal throttling: DGEMM n = 2048 (NEON, MC=240, KC=256) without pauses\nApple M5 (MacBook Air), one performance core, passive cooling"
set xlabel "time [s]"
set ylabel "performance [GFLOP/s]"
set yrange [45:70]
set xrange [0:*]
set key inside bottom left
set style fill transparent solid 0.15 noborder
plot S."thermal_sustained.csv" using "seconds":(column("thermal_fair") > 0 ? 70 : NaN) with impulses lc rgb "#fdae6b" lw 3 title "macOS thermal state: fair", \
     ""                        using "seconds":"gflops" with points ls 4 ps 0.5 title "GFLOP/s of one sample (one call)"
