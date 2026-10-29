load "_style.gp"
load "_selection.gp"
set terminal svg size 1250,900 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "square_selected.svg"
set multiplot title "DGEMM, square matrices (m = n = k): selected fixed SME2 configurations (best for at least 3 shapes, the default, BLIS)\nApple M5 (MacBook Air), one performance core, medians of 3 interleaved rounds" font ",15"
set xrange [0:4200]
set xtics 0, 512, 4096
set mxtics 2
set key outside right center vertical font ",11" spacing 1.05 nobox
FILE = "square.csv"; XCOL = "n"; VARS = SEL
set lmargin at screen 0.07; set rmargin at screen 0.70
set origin 0, 0.52; set size 1, 0.40
set yrange [0:540]
XLABEL = ""; TITLE = "full range"
load "_family.gp"
set key off
set origin 0, 0.0; set size 1, 0.52
set yrange [380:500]
XLABEL = "matrix size m = n = k"; TITLE = "zoom: 380 to 500 GFLOP/s (NEON below the range)"
load "_family.gp"
unset multiplot
