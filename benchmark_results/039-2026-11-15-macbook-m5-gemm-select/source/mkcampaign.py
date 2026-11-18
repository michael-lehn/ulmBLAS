"""Benchmark campaign of the GEMM selection benchmark (selbench.cc): data
(raw, summary), Gnuplot scripts, sources.

    python3 mkcampaign.py DATA OUT CPU

DATA: directory of VARIANT-TYPE-ROUND.tsv (type m n k gflops per line);
OUT: the campaign directory (data/, plots/, source/ are written);
CPU: description of the machine for the plot titles.
"""
import collections, glob, os, shutil, statistics, sys

D = os.path.dirname(os.path.abspath(__file__))
DATA, OUT, CPU = sys.argv[1], sys.argv[2], sys.argv[3]
VARIANTS = ["master", "generic", "tuned"]
TITLES = {"master": "master (before the redesign)",
          "generic": "redesign, hand-written table",
          "tuned": "redesign, table of make tune"}
COLORS = {"master": "#52514e", "generic": "#2a78d6", "tuned": "#e34948"}
TYPENAMES = {"d": "double", "s": "float", "c": "complex float",
             "z": "complex double"}

for sub in ("data/raw", "data/summary", "plots", "source"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

#  samples[(type, m, n, k)][variant] = [gflops per round]
samples = collections.defaultdict(lambda: collections.defaultdict(list))
order = []
raw = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(DATA, "*.tsv"))):
    v, t, r = os.path.basename(f)[:-4].split("-")
    for line in open(f):
        x = line.split()
        key = (x[0], int(x[1]), int(x[2]), int(x[3]))
        if key not in order:
            order.append(key)
        samples[key][v].append(float(x[4]))
        raw[t].append((v, r, x[1], x[2], x[3], x[4]))

for t, rows in raw.items():
    with open(os.path.join(OUT, "data/raw", "%s.csv" % t), "w") as f:
        f.write("variant,round,m,n,k,gflops\n")
        for row in rows:
            f.write(",".join(row) + "\n")


def med(key, v):
    return statistics.median(samples[key][v])


def square(key):
    return key[1] == key[2] == key[3]


#  summary per type: squares (x = n) and all shapes (index, label)
for t in "dszc":
    keys = [k for k in order if k[0] == t]
    with open(os.path.join(OUT, "data/summary", "%s_square.csv" % t), "w") as f:
        f.write("n," + ",".join(VARIANTS) + ",generic_ratio,tuned_ratio\n")
        for k in keys:
            if square(k):
                m = med(k, "master")
                f.write("%d,%s,%.3f,%.3f\n" % (
                    k[1], ",".join("%.1f" % med(k, v) for v in VARIANTS),
                    med(k, "generic") / m, med(k, "tuned") / m))
    with open(os.path.join(OUT, "data/summary", "%s_shapes.csv" % t), "w") as f:
        f.write("index,shape,m,n,k," + ",".join(VARIANTS) +
                ",generic_ratio,tuned_ratio\n")
        for i, k in enumerate(keys):
            m = med(k, "master")
            f.write("%d,%dx%dx%d,%d,%d,%d,%s,%.3f,%.3f\n" % (
                i, k[1], k[2], k[3], k[1], k[2], k[3],
                ",".join("%.1f" % med(k, v) for v in VARIANTS),
                med(k, "generic") / m, med(k, "tuned") / m))

STYLE = """#  Common style of the plots of this campaign (loaded by the other
#  scripts; files starting with '_' are not plotted by
#  'make add_benchmark_result').
set datafile separator ","
set datafile columnheaders
set grid xtics ytics lc rgb "#dcdcdc" lw 1 dt 1
set border lw 1.0 lc rgb "#52514e"
set tics nomirror out textcolor rgb "#0b0b0b"
D = "../data/summary/"
CPU = "%s"
"""
open(os.path.join(OUT, "plots/_style.gp"), "w").write(STYLE % CPU)


def gp(name, lines):
    open(os.path.join(OUT, "plots", name + ".gp"), "w").write(
        "\n".join(lines) + "\n")


HEAD = ['load "_style.gp"',
        'set terminal svg size 1250,680 dynamic enhanced font "Helvetica,14" '
        'background rgb "white"']
KEY = ('set key outside right center vertical font ",11.5" spacing 1.2 '
       'width 1 box lc rgb "#dcdcdc"')

for t in "dszc":
    gp("%s_square" % t, HEAD + [
        'set output "%s_square.svg"' % t,
        'set title "%sgemm, square matrices, column major, %s: ulmBLAS::gemm\\n"'
        '.CPU."\\nmedian of 5 rounds (each the median of 5 samples of >= 20 ms)" '
        'font ",14"' % (t, TYPENAMES[t]),
        'set xlabel "matrix size n (m = n = k)"',
        'set ylabel "performance [GFLOP/s]"',
        'set logscale x 2',
        'set yrange [0:*]',
        KEY,
        'plot ' + ", \\\n     ".join(
            'D."%s_square.csv" using "n":"%s" with linespoints pt 7 ps 0.6 '
            'lc rgb "%s" lw 2 title "%s"' % (t, v, COLORS[v], TITLES[v])
            for v in VARIANTS)])

gp("d_square_ratio", HEAD + [
    'set output "d_square_ratio.svg"',
    'set title "dgemm, square matrices: relative to master\\n".CPU font ",14"',
    'set xlabel "matrix size n (m = n = k)"',
    'set ylabel "performance relative to master"',
    'set logscale x 2',
    'set yrange [0.8:1.3]',
    KEY,
    'plot 1 with lines lc rgb "#52514e" dt 2 title "master", \\\n'
    '     D."d_square.csv" using "n":"generic_ratio" with linespoints pt 7 '
    'ps 0.6 lc rgb "%s" lw 2 title "%s", \\\n'
    '     D."d_square.csv" using "n":"tuned_ratio" with linespoints pt 7 '
    'ps 0.6 lc rgb "%s" lw 2 title "%s"' % (COLORS["generic"],
                                           TITLES["generic"],
                                           COLORS["tuned"], TITLES["tuned"])])

gp("d_shapes_ratio", HEAD[:1] + [
    'set terminal svg size 1250,760 dynamic enhanced font "Helvetica,14" '
    'background rgb "white"',
    'set output "d_shapes_ratio.svg"',
    'set title "dgemm, all shapes (m x n x k): relative to master\\n".CPU '
    'font ",14"',
    'set ylabel "performance relative to master"',
    'set yrange [0.8:1.3]',
    'set xtics rotate by -60 font ",9"',
    'set style data histogram',
    'set style histogram clustered gap 1',
    'set style fill solid 0.85 border -1',
    'set boxwidth 0.9',
    KEY,
    'plot D."d_shapes.csv" using "generic_ratio":xticlabels(2) '
    'lc rgb "%s" title "%s", \\\n'
    '     D."d_shapes.csv" using "tuned_ratio" lc rgb "%s" title "%s", \\\n'
    '     1 with lines lc rgb "#52514e" dt 2 notitle' % (
        COLORS["generic"], TITLES["generic"], COLORS["tuned"],
        TITLES["tuned"])])

for f in ("selbench.cc", "run-mac.sh", "mkcampaign.py"):
    shutil.copy(os.path.join(D, f), os.path.join(OUT, "source", f))
print("ok", OUT)
