#  One plot of a family (loaded with FILE, XCOL, XLABEL, VARS, TITLE set):
#  Accelerate, NEON and the SME2 configurations VARS.
set xlabel XLABEL
set ylabel "performance [GFLOP/s]"
set title TITLE
plot S.FILE using XCOL:"accelerate" with linespoints ls 1 title "Apple Accelerate, 1 thread (session 2)", \
     ""     using XCOL:"neon"       with linespoints ls 2 title "ulmBLAS NEON (session 2)", \
     for [v in VARS] "" using XCOL:v with linespoints lc rgb col(v) dt dtk(v) pt ptn(v) ps 0.8 lw 1.8 title lab(v)
