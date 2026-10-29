load "_style.gp"

set terminal svg size 1250,700 dynamic enhanced font "Helvetica,14" background rgb "white"
set output "small_m_all.svg"
set xrange [0:400]
set xtics 0, 64, 384
set mxtics 2
set yrange [0:540]
set key outside right center vertical font ",11" spacing 1.05 nobox
FILE = "small_m.csv"; XCOL = "m"; XLABEL = "m (n = k = 2048)"; VARS = ALL
TITLE = "DGEMM, n = k = 2048, m varies: all 12 fixed SME2 configurations\nApple M5 (MacBook Air), one performance core, medians of 3 interleaved rounds"
load "_family.gp"
