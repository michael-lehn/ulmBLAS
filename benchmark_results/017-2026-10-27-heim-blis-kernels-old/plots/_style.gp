#  Common style of the kernel campaign plots (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
#  _labels.gp defines CPU, the arrays V (column names of the variants in
#  data/summary/*.csv), T (titles), K (1: micro kernel in ulmBLAS, 0: native
#  library), the arrays R and RT (columns VARIANT_over_ref and titles) and
#  PYMM, PZMM (FMA peaks), YMAX.
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out
D = "../data/summary/"
C = "#2a78d6 #c8323c #1baf7a #e8871e #9b3fb5 #3a3a38 #0fa3b1 #b5862a #d14fa0 #6b8e23 #5d5fef #8c564b #7f7f7f #17becf #bcbd22"
color(i) = word(C, (i-1)%words(C)+1)
