#!/bin/sh
#
# Adds a completed benchmark campaign to the benchmark archive (make
# add_benchmark_result, see benchmark_results/README.md).
#
#     benchmark_result.sh ARCHIVE CAMPAIGN_DIR [NAME]
#
# ARCHIVE is the archive directory (benchmark_results), CAMPAIGN_DIR the
# campaign to add (anywhere, or a campaign of the archive itself to
# regenerate its plots and documentation).  NAME, if given, must equal NAME
# in CAMPAIGN_DIR/campaign.conf.
#
# A campaign directory contains
#
#     campaign.conf   KEY = value lines: NAME, DATE, MACHINE, TITLE, SUMMARY
#     README.md       description of the experiment
#     data/           measurements: *.csv (also in subdirectories)
#     plots/          Gnuplot scripts *.gp (scripts starting with '_' are
#                     only loaded by the others); each writes an SVG file
#                     (set output "NAME.svg")
#     source/         optional: benchmark programs and scripts
#
# Steps (nothing in ARCHIVE is changed before all of them succeeded):
#
#   1. validate the campaign (metadata, files, no binaries or build
#      artifacts);
#   2. copy it to a temporary directory in ARCHIVE, run Gnuplot for all
#      plot scripts, check the SVG files, update the list of figures in
#      README.md (between <!-- BEGIN FIGURES --> and <!-- END FIGURES -->);
#   3. regenerate the index (the table between <!-- BEGIN CAMPAIGNS --> and
#      <!-- END CAMPAIGNS --> in ARCHIVE/README.md) from the campaign.conf
#      files of all campaigns;
#   4. move the campaign and the index into place.
#
# An archived campaign is never overwritten by different data: if
# ARCHIVE/NAME exists, the new campaign must have the same contents (apart
# from the generated SVG files and the generated list of figures), so that
# adding a campaign twice changes nothing.  Benchmarks are not run, nothing
# is committed.
#
# GNUPLOT selects the Gnuplot program (default: gnuplot).
#

archive=$1
src=$2
name=${3:-}
gnuplot=${GNUPLOT:-gnuplot}
usage="make add_benchmark_result BENCHMARK_DIR=<campaign directory> [BENCHMARK_NAME=<name>]"

# temporary files: $tmp (the new campaign) and $tmp.* in ARCHIVE
tmp=
cleanup()
{
    [ -n "$tmp" ] && rm -rf "$tmp" "$tmp".*
    rm -f "$archive/.import-check.$$"
}
die()
{
    echo "error: $*" >&2
    cleanup
    exit 1
}
trap cleanup 0
trap 'exit 1' 1 2 15

[ -n "$src" ] || die "BENCHMARK_DIR is not set.  Usage: $usage"
[ -d "$src" ] || die "BENCHMARK_DIR '$src' is not a directory"
[ -f "$archive/README.md" ] || die "$archive/README.md (the index) not found"

#
# 1. Validation
#
conf=$src/campaign.conf
[ -f "$conf" ] || die "$conf not found"

# get FILE KEY: value of 'KEY = value' (first occurrence)
get()
{
    sed -n "s/^$2[ 	]*=[ 	]*//p" "$1" | head -1 | sed 's/[ 	]*$//'
}

# check_conf FILE: required keys, no characters that break the index table
check_conf()
{
    for key in NAME DATE MACHINE TITLE SUMMARY; do
        value=`get "$1" $key`
        [ -n "$value" ] || die "$1: $key is missing"
        case $value in
            *'|'*) die "$1: $key must not contain '|' (index table)" ;;
        esac
    done
    c_name=`get "$1" NAME`
    c_date=`get "$1" DATE`
    echo "$c_date" | grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' \
        || die "$1: DATE '$c_date' is not YYYY-MM-DD"
    echo "$c_name" | grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}-[a-z0-9][a-z0-9._-]*$' \
        || die "$1: NAME '$c_name' is not DATE-machine-experiment (lower case letters, digits, '.', '_', '-')"
    case $c_name in
        "$c_date"-*) ;;
        *) die "$1: NAME '$c_name' does not start with DATE '$c_date'" ;;
    esac
}

check_conf "$conf"
cname=`get "$conf" NAME`
if [ -n "$name" ] && [ "$name" != "$cname" ]; then
    die "BENCHMARK_NAME '$name' differs from NAME '$cname' in $conf"
fi
name=$cname

[ -f "$src/README.md" ] || die "$src/README.md not found"
csv=`find "$src/data" -type f -name '*.csv' 2>/dev/null`
[ -n "$csv" ] || die "no measurements: $src/data/*.csv not found"
for f in $csv; do
    [ `wc -l < "$f"` -ge 2 ] || die "$f: no measurements (header line only or empty)"
    head -1 "$f" | grep -q , || die "$f: the first line is not a CSV header"
done
scripts=
for f in "$src"/plots/*.gp; do
    [ -f "$f" ] || continue
    case `basename "$f"` in _*) continue ;; esac
    grep -q '^[ 	]*set output "[^"/]*\.svg"' "$f" \
        || die "$f: no 'set output \"NAME.svg\"'"
    scripts="$scripts `basename "$f"`"
done
[ -n "$scripts" ] || die "no plot scripts: $src/plots/*.gp not found"

# no executables, object files, libraries or other binaries; no large files
find "$src" -type f | while read f; do
    case $f in
        *.o|*.obj|*.a|*.so|*.so.*|*.dylib|*.dll|*.exe|*.out|*.gch)
            echo "$f" ;;
        *)
            magic=`head -c 4 "$f" | od -An -tx1 | tr -d ' \n'`
            case $magic in
                7f454c46|cffaedfe|cefaedfe|feedfacf|feedface|cafebabe|4d5a*)
                    echo "$f" ;;
            esac
            [ `wc -c < "$f"` -le 4000000 ] || echo "$f"
            ;;
    esac
done > "$archive/.import-check.$$"
if [ -s "$archive/.import-check.$$" ]; then
    die "build artifacts, binaries or files larger than 4 MB:
`sed 's/^/  /' "$archive/.import-check.$$"`"
fi
rm -f "$archive/.import-check.$$"

command -v "$gnuplot" > /dev/null 2>&1 \
    || die "Gnuplot ('$gnuplot') not found.  The plots are generated with
  Gnuplot 5 (SVG terminal); install it (e.g. 'apt install gnuplot',
  'brew install gnuplot') or select it with GNUPLOT=/path/to/gnuplot."
"$gnuplot" -e 'set terminal svg' > /dev/null 2>&1 \
    || die "'$gnuplot' does not support the SVG terminal"

#
# 2. Copy and generate the plots in a temporary directory
#
dest=$archive/$name
tmp=$archive/.import-$name.$$
mkdir "$tmp" || die "cannot create $tmp"
cp -R "$src"/. "$tmp"/ || die "cannot copy $src"
rm -f "$tmp"/plots/*.svg

for s in $scripts; do
    out=`sed -n 's/^[ 	]*set output "\([^"]*\)".*/\1/p' "$tmp/plots/$s"`
    ( cd "$tmp/plots" && "$gnuplot" "$s" ) > "$tmp/gnuplot.log" 2>&1 \
        || { cat "$tmp/gnuplot.log" >&2; die "gnuplot failed for plots/$s"; }
    grep -i -E 'warning|error' "$tmp/gnuplot.log" | sed "s|^|plots/$s: |" >&2
    for svg in $out; do
        f=$tmp/plots/$svg
        [ -s "$f" ] || die "plots/$s did not write plots/$svg"
        head -c 200 "$f" | grep -q '<svg\|<?xml' && tail -c 100 "$f" | grep -q '</svg>' \
            || die "plots/$svg is not a complete SVG file"
    done
done
rm -f "$tmp/gnuplot.log"

# list of figures in README.md
figures=$tmp/.figures
{
    echo "<!-- BEGIN FIGURES (generated by make add_benchmark_result) -->"
    for svg in `cd "$tmp/plots" && ls *.svg`; do
        echo
        echo "![plots/$svg](plots/$svg)"
    done
    echo
    echo "<!-- END FIGURES -->"
} > "$figures"
readme=$tmp/README.md
if grep -q '^<!-- BEGIN FIGURES' "$readme"; then
    grep -q '^<!-- END FIGURES -->' "$readme" \
        || die "$src/README.md: <!-- BEGIN FIGURES --> without <!-- END FIGURES -->"
    awk -v f="$figures" '
        /^<!-- BEGIN FIGURES/ { while ((getline l < f) > 0) print l; skip = 1; next }
        /^<!-- END FIGURES -->/ { skip = 0; next }
        !skip' "$readme" > "$readme.new"
else
    { cat "$readme"; echo; echo "## Figures"; echo; cat "$figures"; } > "$readme.new"
fi
mv "$readme.new" "$readme"
rm -f "$figures"

# an archived campaign is only regenerated, never replaced by other data
inplace=no
if [ -d "$dest" ]; then
    if [ "`cd "$src" && pwd -P`" = "`cd "$dest" && pwd -P`" ]; then
        inplace=yes
    fi
    if ! diff -r -x '*.svg' "$dest" "$tmp" > /dev/null 2>&1; then
        if [ $inplace = yes ]; then
            : # regenerating an archived campaign: only generated parts change
        else
            diff -r -q -x '*.svg' "$dest" "$tmp" | sed 's/^/  /' >&2
            die "campaign '$name' is already archived with different contents
  ($dest).  Archived campaigns are not overwritten: use a new NAME, or
  remove or update $dest deliberately (e.g. with git)."
        fi
    fi
fi

#
# 3. Index of all campaigns
#
index=$archive/README.md
grep -q '^<!-- BEGIN CAMPAIGNS' "$index" && grep -q '^<!-- END CAMPAIGNS -->' "$index" \
    || die "$index: <!-- BEGIN CAMPAIGNS --> and <!-- END CAMPAIGNS --> not found"
rows=$tmp.rows
: > "$rows"
for c in "$archive"/*/campaign.conf "$tmp/campaign.conf"; do
    [ -f "$c" ] || continue
    d=`dirname "$c"`
    [ "$d" = "$dest" ] && continue        # replaced by the new version
    check_conf "$c"
    n=`get "$c" NAME`
    if [ "$d" != "$tmp" ] && [ "`basename "$d"`" != "$n" ]; then
        die "$c: NAME '$n' differs from the directory name"
    fi
    printf '| [%s](%s/README.md) | %s | %s | %s | %s |\n' "$n" "$n" \
        "`get "$c" DATE`" "`get "$c" MACHINE`" "`get "$c" TITLE`" \
        "`get "$c" SUMMARY`" >> "$rows"
done
{
    echo "<!-- BEGIN CAMPAIGNS (generated by make add_benchmark_result) -->"
    echo
    echo "| Campaign | Date | Machine | Experiment | Summary |"
    echo "|----------|------|---------|------------|---------|"
    LC_ALL=C sort "$rows"
    echo
    echo "<!-- END CAMPAIGNS -->"
} > "$tmp.table"
rm -f "$rows"
awk -v f="$tmp.table" '
    /^<!-- BEGIN CAMPAIGNS/ { while ((getline l < f) > 0) print l; skip = 1; next }
    /^<!-- END CAMPAIGNS -->/ { skip = 0; next }
    !skip' "$index" > "$tmp.index" || die "cannot write the index"
rm -f "$tmp.table"

#
# 4. Move into place
#
if [ -d "$dest" ]; then
    rm -rf "$dest.old"
    mv "$dest" "$dest.old" && mv "$tmp" "$dest" && rm -rf "$dest.old" \
        || die "cannot replace $dest (previous version: $dest.old)"
else
    mv "$tmp" "$dest" || die "cannot move the campaign to $dest"
fi
mv "$tmp.index" "$index" || die "cannot update $index"
tmp=
echo "benchmark campaign '$name': $dest ($(ls "$dest"/plots/*.svg | wc -l | tr -d ' ') plots), index $index"
