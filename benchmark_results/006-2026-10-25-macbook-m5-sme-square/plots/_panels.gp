#  Two panels (loaded with VARS, TITLE, ERR set): the full range with the
#  peak lines, and a zoom on 380 to 510 GFLOP/s where the SME2 curves
#  cross; ERR = 1 draws the minimum and maximum over the rounds as error
#  bars in the zoom panel.
set multiplot title TITLE font ",15"
set lmargin at screen 0.07; set rmargin at screen 0.68
set key outside right center vertical maxrows 20 font ",11" spacing 1.0 nobox
set origin 0, 0.50; set size 1, 0.42
set yrange [0:580]
set xlabel ""
set title "full range; dotted: peak instruction throughput measured by microbenchmarks (not DGEMM)" font ",12"
plot SMEPEAK with lines ls 9 title sprintf("SME2 FMOPA peak, microbenchmark (%.0f)", SMEPEAK), \
     NEONPEAK with lines ls 9 title sprintf("NEON FMA peak, microbenchmark (%.1f)", NEONPEAK), \
     S."square.csv" using "n":"accelerate" with linespoints ls 1 title "Apple Accelerate", \
     ""             using "n":"blis_sme"   with linespoints ls 2 title "BLIS PR #921 SME2", \
     ""             using "n":"neon"       with linespoints ls 3 title "ulmBLAS NEON", \
     for [v in VARS] "" using "n":v with linespoints lc rgb col(v) dt dtk(v) pt ptn(v) ps 0.8 lw 1.8 title lab(v)
set key off
set origin 0, 0.0; set size 1, 0.50
set yrange [380:510]
set xlabel "matrix size m = n = k"
set title (ERR ? "zoom: 380 to 510 GFLOP/s; bars: minimum and maximum over the rounds" : "zoom: 380 to 510 GFLOP/s") font ",12"
if (ERR) {
plot S."square.csv" using "n":"accelerate":"accelerate_min":"accelerate_max" with yerrorlines ls 1 notitle, \
     ""             using "n":"blis_sme":"blis_sme_min":"blis_sme_max" with yerrorlines ls 2 notitle, \
     for [v in VARS] "" using "n":v:(column(v."_min")):(column(v."_max")) with yerrorlines lc rgb col(v) dt dtk(v) pt ptn(v) ps 0.8 lw 1.8 notitle
} else {
plot S."square.csv" using "n":"accelerate" with linespoints ls 1 notitle, \
     ""             using "n":"blis_sme"   with linespoints ls 2 notitle, \
     for [v in VARS] "" using "n":v with linespoints lc rgb col(v) dt dtk(v) pt ptn(v) ps 0.8 lw 1.8 notitle
}
unset multiplot
