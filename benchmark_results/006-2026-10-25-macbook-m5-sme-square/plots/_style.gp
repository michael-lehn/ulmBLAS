#  Common style of the plots of this campaign (loaded by the other scripts;
#  files starting with '_' are not plotted by 'make add_benchmark_result').
#
#  Every fixed ulmBLAS SME2 configuration bMC_KC_NC keeps its style: color
#  by MC (256 blue, 384 orange, 608 purple), line by KC (1024 solid, 2240
#  dashed), point by NC (1024 open circle, 3072 filled square).  Apple
#  Accelerate green, BLIS PR #921 SME2 dark gray (dash-dot), ulmBLAS NEON
#  gold.
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
lab(v) = sprintf("ulmBLAS SME2 MC=%d KC=%d NC=%d%s", mc(v), kc(v), nc(v), \
                 v eq "b384_1024_1024" ? " (default)" : "")
set style line 1 lc rgb "#1baf7a" lw 3   pt 13 ps 1.1 dt 1           # Accelerate
set style line 2 lc rgb "#3a3a38" lw 2.5 pt 4  ps 1.0 dt (12,4,3,4)  # BLIS PR #921 SME2
set style line 3 lc rgb "#eda100" lw 3   pt 9  ps 1.1 dt 1           # ulmBLAS NEON
set style line 9 lc rgb "#6b6a64" lw 1.2 dt 3                        # peak lines

#  Peak values measured with microbenchmarks in the campaign
#  004-2026-10-25-macbook-m5-sme (not DGEMM results, not theoretical values):
#  FMOPA (8 tiles, data in registers) and NEON FMA instruction throughput.
SMEPEAK = 542.0
NEONPEAK = 68.9
set xlabel "matrix size m = n = k"
set ylabel "performance [GFLOP/s]"
#  Linear x axis (as in the BLIS performance plots); the sizes n <= 128 are
#  close together at the left.
set xrange [0:4200]
set xtics 0, 512, 4096
set mxtics 2
