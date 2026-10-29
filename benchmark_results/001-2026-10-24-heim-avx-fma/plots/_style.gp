#  Common style of the plots of this campaign (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
set terminal svg size 960,600 dynamic enhanced font "Helvetica,15" \
    background rgb "white"
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set key outside right center box opaque spacing 1.25 width 1
set tics nomirror out
set mytics 2

#  one style per variant, the same in all plots
set style line 1 lc rgb "#7f7f7f" lw 2.2 pt 6  ps 0.9 dt 1   # ulmBLAS reference
set style line 2 lc rgb "#e08214" lw 2.2 pt 9  ps 0.9 dt 1   # ulmBLAS SSE
set style line 3 lc rgb "#6baed6" lw 2.2 pt 4  ps 0.9 dt 2   # ulmBLAS AVX MC=72
set style line 4 lc rgb "#08519c" lw 2.8 pt 5  ps 0.9 dt 1   # ulmBLAS AVX MC=288
set style line 5 lc rgb "#238b45" lw 2.2 pt 7  ps 0.9 dt 1   # BLIS
set style line 6 lc rgb "#cb181d" lw 2.2 pt 13 ps 1.0 dt 1   # OpenBLAS
set style line 9 lc rgb "#000000" lw 1.2 dt 3                # reference lines

S = "../data/summary/"
