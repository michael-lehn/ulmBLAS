#  Common style of the plots of this campaign (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
#
#  ulmBLAS AVX/FMA: published (gray, dashed), the two intermediate versions
#  (thin), current (blue, thick); SSE builds lighter; BLIS and OpenBLAS as
#  in the campaign 001-2026-10-24-heim-avx-fma.  Bands: minimum to maximum
#  over the rounds.
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out
D = "../data/summary/"

set style line 1 lc rgb "#8c8c8c" lw 2.2 dt (8,4) pt 6 ps 0.6             # AVX published
set style line 2 lc rgb "#e8a33d" lw 1.4 pt 7 ps 0.45                      # + panel packing
set style line 3 lc rgb "#c4553a" lw 1.4 pt 7 ps 0.45                      # + per-thread buffers
set style line 4 lc rgb "#2a78d6" lw 3.2 pt 7 ps 0.7                       # AVX current
set style line 5 lc rgb "#b8b8b8" lw 1.6 dt (8,4) pt 6 ps 0.5              # SSE published
set style line 6 lc rgb "#8fb8ea" lw 2.2 pt 7 ps 0.5                       # SSE current
set style line 7 lc rgb "#3a3a38" lw 2.0 dt (12,4,3,4) pt 4 ps 0.6         # BLIS
set style line 8 lc rgb "#1baf7a" lw 2.0 pt 13 ps 0.7                      # OpenBLAS
set style line 9 lc rgb "#6b6a64" lw 1.2 dt 3                              # reference lines

L1 = "ulmBLAS AVX/FMA, published (e195bea)"
L2 = "+ panel grouping, constexpr parameters (57e1c8a)"
L3 = "+ per-thread buffer cache (04cf17b)"
L4 = "current (febb852)"
L5 = "ulmBLAS SSE, published (e195bea)"
L6 = "ulmBLAS SSE, current (febb852)"
L7 = "BLIS 2.0 (haswell)"
L8 = "OpenBLAS 0.3.34 (HASWELL)"
