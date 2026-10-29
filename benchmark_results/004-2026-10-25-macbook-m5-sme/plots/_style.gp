#  Common style of the plots of this campaign (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
set terminal svg size 960,600 dynamic enhanced font "Helvetica,15" \
    background rgb "white"
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out

#  one style per variant, the same in all plots (colors and point types
#  differ, so that the variants are not distinguished by color alone)
set style line 1 lc rgb "#2a78d6" lw 2.8 pt 7  ps 0.9 dt 1   # ulmBLAS SME2
set style line 2 lc rgb "#eb6834" lw 2.2 pt 5  ps 0.9 dt 1   # BLIS PR #921 SME2
set style line 3 lc rgb "#1baf7a" lw 2.2 pt 13 ps 1.0 dt 1   # Accelerate
set style line 4 lc rgb "#eda100" lw 2.2 pt 9  ps 1.0 dt 1   # ulmBLAS NEON
set style line 5 lc rgb "#e87ba4" lw 1.6 pt 11 ps 1.0 dt 2   # BLIS 2.0 NEON
set style line 6 lc rgb "#008300" lw 1.6 pt 3  ps 0.9 dt 2   # OpenBLAS NEON
set style line 9 lc rgb "#000000" lw 1.2 dt 3                # reference lines

S = "../data/summary/"
#  FMOPA peak of the core: smepeak, 8 tiles, median of the rounds
stats S."smepeak.csv" using (strcol("parameter") eq "tiles=8" ? column("gflops") : NaN) name "PEAK" nooutput
