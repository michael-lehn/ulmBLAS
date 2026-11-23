#  Common style of the plots of this campaign (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
#  Colors as in the other M5 campaigns: ulmBLAS blue, Apple Accelerate
#  green, BLIS pull request #921 dark gray (dash-dot).
set datafile separator ","
set datafile columnheaders
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out
SW = "../data/summary/"

set style line 5 lc rgb "#2a78d6" lw 3.2 pt 7 ps 0.7                       # ulmBLAS
set style line 6 lc rgb "#1baf7a" lw 2.6 pt 13 ps 0.8                      # Accelerate
set style line 7 lc rgb "#3a3a38" lw 2.0 dt (12,4,3,4) pt 4 ps 0.6         # BLIS SME2
set style line 9 lc rgb "#6b6a64" lw 1.2 dt 3                              # peak, 1

L5 = "ulmBLAS 0.4.2, SME2"
L6 = "Apple Accelerate"
L7 = "BLIS PR #921 SME2"

#  Peak instruction throughput measured with a microbenchmark in the campaign
#  004-2026-10-25-macbook-m5-sme (FMOPA only; not DGEMM, not theoretical).
SMEPEAK = 542.0
