#  Common style of the plots of this campaign (loaded by the other
#  scripts; files starting with '_' are not plotted by
#  'make add_benchmark_result').
set datafile separator ","
set datafile columnheaders
set grid xtics ytics lc rgb "#dcdcdc" lw 1 dt 1
set border lw 1.0 lc rgb "#52514e"
set tics nomirror out textcolor rgb "#0b0b0b"
D = "../data/summary/"
CPU = "Apple M5 (MacBook Air), one performance core, clang, -mcpu=apple-m4 (SME2)"
