# prepack over the row stride, in place (gnuplot >= 5.4).
#
#   gnuplot -e "D='results/<campaign>'; O='plots'" scripts/plot-ppsweep.gp
#
# D/summary-sweep-potrf.dat (scripts/summarize-sweep.py, variants inplace
# and inplace+pp), one line per incRow R.  For n = 500, 1000, 2000, 4000:
#   inplace:    gflops in columns 2, 4, 6, 8, rel in 3, 5, 7, 9
#   inplace+pp: gflops in columns 26, 28, 30, 32, rel in 27, 29, 31, 33
# (rel = time in place with R = 1 / time).  Ratio = time without / time
# with prepack = rel_pp / rel.
#
# Writes
#   O/potrf-ppsweep.svg         ratio over R, one line per n
#   O/potrf-ppsweep-smooth.svg  the same, points and a smoothed curve
#   O/potrf-ppsweep-heat.svg    ratio as a heat map, R x n
#   O/potrf-ppsweep-gflops.svg  GFLOP/s with and without prepack

C1 = "#2a78d6"; C2 = "#eb6834"; C3 = "#1baf7a"; C4 = "#eda100"
INK = "#52514e"
if (!exists("SUB")) SUB = "Xeon Platinum 8358 Ice Lake, one socket, 32 threads"
F = D."/summary-sweep-potrf.dat"

set terminal svg size 900,520 dynamic enhanced font "Helvetica,13" background rgb "white"
set border 3 lw 1.0 lc rgb INK
set tics nomirror tc rgb INK
set grid xtics ytics lc rgb "#e2e2e2" lw 1 dt 1
set key outside right top Left reverse box lc rgb "#dcdcdc" spacing 1.25 width 1
set xrange [0:101]
set xtics 0,10,100
set mxtics 2
set xlabel "incRow"

# ratio, one line per n
set yrange [0.6:1.6]
set ylabel "time without / time with prepack"
set output O."/potrf-ppsweep.svg"
set title "DPOTRF in place, ".SUB.": when does prepack pay off?"
plot 1 w l lc rgb INK dt 2 lw 1.2 notitle, \
     F u 1:($27/$3) w lp lc rgb C1 lw 2 pt 7 ps 0.4 t "n = 500", \
     F u 1:($29/$5) w lp lc rgb C2 lw 2 pt 7 ps 0.4 t "n = 1000", \
     F u 1:($31/$7) w lp lc rgb C3 lw 2 pt 7 ps 0.4 t "n = 2000", \
     F u 1:($33/$9) w lp lc rgb C4 lw 2 pt 7 ps 0.4 t "n = 4000"

# the same, points only and a smoothed curve
set output O."/potrf-ppsweep-smooth.svg"
set title "DPOTRF in place, ".SUB.": prepack, smoothed"
plot 1 w l lc rgb INK dt 2 lw 1.2 notitle, \
     F u 1:($27/$3) w p lc rgb C1 pt 7 ps 0.35 notitle, \
     F u 1:($29/$5) w p lc rgb C2 pt 7 ps 0.35 notitle, \
     F u 1:($31/$7) w p lc rgb C3 pt 7 ps 0.35 notitle, \
     F u 1:($33/$9) w p lc rgb C4 pt 7 ps 0.35 notitle, \
     F u 1:($27/$3):(0.01) smooth acsplines lc rgb C1 lw 2 t "n = 500", \
     F u 1:($29/$5):(0.01) smooth acsplines lc rgb C2 lw 2 t "n = 1000", \
     F u 1:($31/$7):(0.01) smooth acsplines lc rgb C3 lw 2 t "n = 2000", \
     F u 1:($33/$9):(0.01) smooth acsplines lc rgb C4 lw 2 t "n = 4000"

# heat map: one row per n, one cell per R; gray = 1, blue = prepack
# faster, orange = slower
set output O."/potrf-ppsweep-heat.svg"
set title "DPOTRF in place, ".SUB.": time without / time with prepack"
unset grid
set yrange [0.4:4.6]
set ytics ("n = 500" 1, "n = 1000" 2, "n = 2000" 3, "n = 4000" 4)
set ylabel ""
set cbrange [0.6:1.4]
set cbtics 0.6,0.1,1.4
set cblabel "time without / time with prepack"
set palette defined (0.6 "#c4540f", 1.0 "#ececec", 1.4 "#1f5fae")
set rmargin at screen 0.84
set colorbox vertical user origin 0.88,0.15 size 0.025,0.7
unset key
hw(r) = r>50 ? 5 : 0.5
plot F u 1:(1):(hw($1)):(0.45):($27/$3) w boxxy fc palette fs solid noborder, \
     F u 1:(2):(hw($1)):(0.45):($29/$5) w boxxy fc palette fs solid noborder, \
     F u 1:(3):(hw($1)):(0.45):($31/$7) w boxxy fc palette fs solid noborder, \
     F u 1:(4):(hw($1)):(0.45):($33/$9) w boxxy fc palette fs solid noborder

# GFLOP/s, solid without, dashed with prepack
set rmargin
set output O."/potrf-ppsweep-gflops.svg"
set title "DPOTRF in place, ".SUB.": GFLOP/s"
set grid xtics ytics lc rgb "#e2e2e2" lw 1 dt 1
set key outside right top Left reverse box lc rgb "#dcdcdc" spacing 1.25 width 1
set autoscale y
set yrange [0:*]
set ytics auto
set ylabel "GFLOP/s"
plot F u 1:2  w lp lc rgb C1 lw 2 pt 7 ps 0.3 t "n = 500", \
     F u 1:4  w lp lc rgb C2 lw 2 pt 7 ps 0.3 t "n = 1000", \
     F u 1:6  w lp lc rgb C3 lw 2 pt 7 ps 0.3 t "n = 2000", \
     F u 1:8  w lp lc rgb C4 lw 2 pt 7 ps 0.3 t "n = 4000", \
     F u 1:26 w lp lc rgb C1 lw 2 dt 2 pt 6 ps 0.3 t "n = 500, prepack", \
     F u 1:28 w lp lc rgb C2 lw 2 dt 2 pt 6 ps 0.3 t "n = 1000, prepack", \
     F u 1:30 w lp lc rgb C3 lw 2 dt 2 pt 6 ps 0.3 t "n = 2000, prepack", \
     F u 1:32 w lp lc rgb C4 lw 2 dt 2 pt 6 ps 0.3 t "n = 4000, prepack"
