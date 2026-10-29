#  Common style of the plots of this campaign (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
set terminal svg size 1100,560 dynamic enhanced font "Helvetica,14" \
    background rgb "white"
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out
set mytics 2
set style data histograms
set style fill solid 0.9 border rgb "#404040"
set boxwidth 0.9
set xtics rotate by -30 font ",12"
set style line 9 lc rgb "#000000" lw 1.2 dt 3                # reference lines

S = "../data/summary/"
