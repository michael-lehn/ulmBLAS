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
# Archived campaigns are named NNN-DATE-machine-experiment: NNN is the
# permanent sequence number of the campaign (three digits, 001, 002, ...),
# its primary identity; DATE is the day on which the measurements were
# completed (the start and end of the measurements are recorded in the
# campaign's README.md).  A new campaign gives NAME without the number
# (DATE-machine-experiment); it then receives the next number (the highest
# archived number plus one), or the number of the archived campaign with
# the same name if it is already archived.  Numbers follow the order in
# which campaigns are archived, not their dates: a campaign completed
# earlier but archived later receives the next number, and several
# campaigns may have the same DATE.  Numbers are never reassigned, so
# existing campaigns are never renumbered.  A lock directory
# (ARCHIVE/.add-lock) keeps two runs in the same checkout from assigning
# the same number.
#
# A campaign directory contains
#
#     campaign.conf   KEY = value lines: NAME, DATE, MACHINE, TITLE, SUMMARY,
#                     optional TYPES (the data types measured, comma
#                     separated: double, complex double, float, complex
#                     float; default double)
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
#   3. regenerate the index (between <!-- BEGIN CAMPAIGNS --> and
#      <!-- END CAMPAIGNS --> in ARCHIVE/README.md: one table per data type,
#      a campaign in the table of each of its TYPES) from the campaign.conf
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
locked=no
cleanup()
{
    [ -n "$tmp" ] && rm -rf "$tmp" "$tmp".*
    rm -f "$archive/.import-check.$$"
    [ $locked = yes ] && rmdir "$archive/.add-lock"
    locked=no
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

# check_conf FILE [new]: required keys, no characters that break the index
# table; NAME is NNN-DATE-machine-experiment (with 'new' also
# DATE-machine-experiment, without the number)
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
    c_base=`echo "$c_name" | sed -n 's/^[0-9]\{3\}-//p'`
    if [ -z "$c_base" ]; then
        [ "${2:-}" = new ] || die "$1: NAME '$c_name' has no sequence number (NNN-DATE-machine-experiment)"
        c_base=$c_name
    fi
    echo "$c_base" | grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}-[a-z0-9][a-z0-9._-]*$' \
        || die "$1: NAME '$c_name' is not NNN-DATE-machine-experiment (lower case letters, digits, '.', '_', '-')"
    case $c_base in
        "$c_date"-*) ;;
        *) die "$1: NAME '$c_name' does not contain DATE '$c_date'" ;;
    esac
    types "$1" > /dev/null
}

# the data types, in the order of the index
TYPE_ORDER='double|complex double|float|complex float'

# types FILE: the TYPES of a campaign, one per line (default double)
types()
{
    t_list=`get "$1" TYPES`
    [ -n "$t_list" ] || t_list=double
    t_ifs=$IFS; IFS=,
    for t in $t_list; do
        IFS=$t_ifs
        t=`echo "$t" | sed 's/^[ 	]*//; s/[ 	]*$//'`
        case "|$TYPE_ORDER|" in
            *"|$t|"*) echo "$t" ;;
            *) die "$1: TYPES: unknown data type '$t' ($TYPE_ORDER)" ;;
        esac
    done
    IFS=$t_ifs
}

check_conf "$conf" new
cname=`get "$conf" NAME`
if [ -n "$name" ] && [ "$name" != "$cname" ]; then
    die "BENCHMARK_NAME '$name' differs from NAME '$cname' in $conf"
fi

# one run at a time: the number is assigned here and taken when the
# campaign is moved into place (step 4)
mkdir "$archive/.add-lock" 2>/dev/null \
    || die "another 'make add_benchmark_result' is running ($archive/.add-lock
  exists; remove it if no other run is active)"
locked=yes

# sequence number: kept if archived, otherwise the next one
last=`ls "$archive" | grep -E '^[0-9]{3}-' | sort | tail -1`
lastnum=`echo "$last" | cut -c1-3`
base=`echo "$cname" | sed 's/^[0-9]\{3\}-//'`
existing=`ls "$archive" | grep -E "^[0-9]{3}-$base\$" | head -1`
if [ -n "$existing" ]; then
    name=$existing
    [ "$cname" = "$base" ] || [ "$cname" = "$existing" ] \
        || die "NAME '$cname': this campaign is archived as '$existing'"
elif [ "$cname" != "$base" ]; then
    die "NAME '$cname': new campaigns are given without the sequence number
  (NAME = $base); the next number is assigned automatically"
else
    next=`expr "${lastnum:-0}" + 1`
    [ $next -le 999 ] || die "no sequence numbers left (999 campaigns)"
    name=`printf '%03d-%s' $next "$base"`
fi

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
# the archived campaign.conf carries the numbered name
sed "s/^NAME\([ 	]*=[ 	]*\).*/NAME\1$name/" "$tmp/campaign.conf" > "$tmp/campaign.conf.new" \
    && mv "$tmp/campaign.conf.new" "$tmp/campaign.conf" || die "cannot set NAME"

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
    types "$c" > "$tmp.types"
    while read t; do
        printf '%s\t| [%s](%s/README.md) | %s | %s | %s | %s |\n' "$t" "$n" "$n" \
            "`get "$c" DATE`" "`get "$c" MACHINE`" "`get "$c" TITLE`" \
            "`get "$c" SUMMARY`" >> "$rows"
    done < "$tmp.types"
    rm -f "$tmp.types"
done
{
    echo "<!-- BEGIN CAMPAIGNS (generated by make add_benchmark_result) -->"
    echo "$TYPE_ORDER" | tr '|' '\n' | while read t; do
        grep -q "^$t	" "$rows" || continue
        echo
        echo "### $t"
        echo
        echo "| Campaign | Date | Machine | Experiment | Summary |"
        echo "|----------|------|---------|------------|---------|"
        grep "^$t	" "$rows" | cut -f2- | LC_ALL=C sort
    done
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
echo "check the archive with: make check_benchmark_results"
