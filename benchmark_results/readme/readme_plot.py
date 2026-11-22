#  README plot: Cholesky (dpotrf) n = 4000 on all cores, ulmBLAS against
#  OpenBLAS, light and dark (data: UlmBLAS.jl README, bench/results
#  2026-11-21-*-potrf.tsv).  python3 readme_plot.py OUTDIR (macOS: Helvetica Neue)
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

font_manager.fontManager.addfont("/System/Library/Fonts/HelveticaNeue.ttc")
plt.rcParams["font.family"] = "Helvetica Neue"
plt.rcParams["svg.fonttype"] = "path"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8984",
                  grid="#e6e5e1", ulm="#2a78d6", ob="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#8f8e88",
                 grid="#33332f", ulm="#3987e5", ob="#d95926"),
}
machines = ["Apple M5\n10 cores", "Xeon 8358\nIce Lake, 64 cores",
            "EPYC 9454\nZen 4, 96 cores", "Core i5-12500\n6 cores"]
openblas = [223.4, 458.8, 388.4, 239.0]
ulmblas = [376.3, 812.5, 697.2, 265.5]



def bar(ax, x0, x1, v, rx, ry, color):
    """bar from 0 to v, top corners rounded (rx, ry in data units)"""
    P = [(x0, 0), (x0, v - ry), (x0, v), (x0 + rx, v), (x1 - rx, v), (x1, v),
         (x1, v - ry), (x1, 0), (x0, 0)]
    C = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
         Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(P, C), facecolor=color, edgecolor="none"))


W, H = 1000, 480
for name, t in THEMES.items():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=t["surface"])
    axw, axh = 0.88, 0.58
    ax = fig.add_axes([0.08, 0.16, axw, axh], facecolor=t["surface"])
    fig.text(0.08, 0.92, "Cholesky factorization (dpotrf), n = 4000, all cores",
             fontsize=17, fontweight="bold", color=t["ink"])
    fig.text(0.08, 0.865, "billion floating-point operations per second (GFLOP/s); "
             "above each machine: time of OpenBLAS / time of ulmBLAS",
             fontsize=12, color=t["ink2"])
    for k, (lab, c) in enumerate((("ulmBLAS 0.4.1", t["ulm"]), ("OpenBLAS 0.3.30", t["ob"]))):
        x0 = 0.08 + k * 0.18
        fig.patches.append(Rectangle((x0, 0.788), 0.016, 0.032, transform=fig.transFigure,
                                     facecolor=c, edgecolor="none"))
        fig.text(x0 + 0.024, 0.79, lab, fontsize=13, color=t["ink"])
    ymax, xmin, xmax = 1000, -0.6, 3.6
    ypx, xpx = ymax / (axh * H), (xmax - xmin) / (axw * W)
    w, gap = 0.34, 0.02
    for i, (o, u) in enumerate(zip(openblas, ulmblas)):
        for x, v, c in ((i - (w + gap) / 2, o, t["ob"]), (i + (w + gap) / 2, u, t["ulm"])):
            bar(ax, x - w / 2, x + w / 2, v, 4 * xpx, 4 * ypx, c)
            ax.text(x, v + 12, f"{v:.0f}", ha="center", va="bottom", fontsize=12, color=t["ink2"])
        ax.text(i, max(o, u) + 75, f"{u / o:.2f}×", ha="center", va="bottom",
                fontsize=15, fontweight="bold", color=t["ink"])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(4), machines, fontsize=12, color=t["ink2"])
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks([0, 200, 400, 600, 800])
    ax.tick_params(axis="y", length=0, labelsize=11, labelcolor=t["muted"])
    ax.grid(axis="y", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color=t["muted"], linewidth=1)
    fig.savefig(f"{sys.argv[1]}/cholesky-4000-{name}.svg", facecolor=t["surface"])
    fig.savefig(f"{sys.argv[1]}/cholesky-4000-{name}.png", facecolor=t["surface"])
