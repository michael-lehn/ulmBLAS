#  Common style of the plots of this campaign (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
#
#  The ulmBLAS steps: published (gray, dashed), the three intermediate
#  versions (thin), final (blue, thick); Apple Accelerate green, BLIS PR
#  #921 SME2 dark gray (dash-dot), as in the other M5 campaigns.
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out
S  = "../data/small/summary/"    # small-matrix campaign
SW = "../data/sweep/summary/"    # sweep n = 50, 100, ..., 4000

set style line 1 lc rgb "#8c8c8c" lw 2.2 dt (8,4) pt 6 ps 0.6             # published
set style line 2 lc rgb "#e8a33d" lw 1.4 pt 7 ps 0.45                      # + packing tail
set style line 3 lc rgb "#c4553a" lw 1.4 pt 7 ps 0.45                      # + buffers
set style line 4 lc rgb "#7d4fc4" lw 1.4 pt 7 ps 0.45                      # + direct path
set style line 5 lc rgb "#2a78d6" lw 3.2 pt 7 ps 0.7                       # final
set style line 6 lc rgb "#1baf7a" lw 2.6 pt 13 ps 0.8                      # Accelerate
set style line 7 lc rgb "#3a3a38" lw 2.0 dt (12,4,3,4) pt 4 ps 0.6         # BLIS SME2
set style line 9 lc rgb "#6b6a64" lw 1.2 dt 3                              # peak lines

L1 = "ulmBLAS SME2, published (881d321)"
L2 = "+ packing tail through ZA (d7ca66a)"
L3 = "+ per-thread buffers (2f18d32)"
L4 = "+ direct NEON path for m·n·k ≤ 8192 (d3566f2)"
L5 = "+ prefetch outside streaming mode: final (b38f3d5)"
L6 = "Apple Accelerate"
L7 = "BLIS PR #921 SME2"

#  Peak instruction throughput measured with microbenchmarks in the campaign
#  004-2026-10-25-macbook-m5-sme (not DGEMM, not theoretical values).
SMEPEAK = 542.0
NEONPEAK = 68.9
