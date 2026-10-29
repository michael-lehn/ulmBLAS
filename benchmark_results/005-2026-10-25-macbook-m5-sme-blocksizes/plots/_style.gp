#  Common style of the plots of the block size study (loaded by the other
#  scripts; files starting with '_' are not plotted by
#  'make add_benchmark_result').
#
#  Every fixed SME2 configuration bMC_KC_NC keeps its style in all plots:
#  color by MC (256 blue, 384 orange, 608 purple), line by KC (1024 solid,
#  2240 dashed), point by NC (1024 open circle, 3072 filled square).
#  Accelerate green, ulmBLAS NEON gold.
set datafile separator ","
set datafile columnheaders
set datafile missing "NaN"
set grid xtics ytics lc rgb "#d0d0d0" lw 1 dt 1
set border lw 1.2
set tics nomirror out

S = "../data/summary/"
ALL = "b256_1024_1024 b256_1024_3072 b256_2240_1024 b256_2240_3072 b384_1024_1024 b384_1024_3072 b384_2240_1024 b384_2240_3072 b608_1024_1024 b608_1024_3072 b608_2240_1024 b608_2240_3072"
mc(v) = int(v[2:4])
kc(v) = int(v[6:9])
nc(v) = int(v[11:14])
col(v) = mc(v) == 256 ? "#2a78d6" : mc(v) == 384 ? "#eb6834" : "#7d4fc4"
dtk(v) = kc(v) == 1024 ? 1 : 2
ptn(v) = nc(v) == 1024 ? 6 : 5
lab(v) = sprintf("SME2 MC=%d KC=%d NC=%d%s", mc(v), kc(v), nc(v), \
                 v eq "b384_1024_1024" ? " (default)" : v eq "b608_2240_3072" ? " (BLIS)" : "")
set style line 1 lc rgb "#1baf7a" lw 3 pt 13 ps 1.1 dt 1   # Accelerate
set style line 2 lc rgb "#eda100" lw 3 pt 9  ps 1.1 dt 1   # ulmBLAS NEON
