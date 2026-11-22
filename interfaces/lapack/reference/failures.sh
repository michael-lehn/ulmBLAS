#! /bin/sh
#
#  failures.sh TESTING KNOWN: the failures in the output of LAPACK's test
#  programs (TESTING/*.out), one line each:
#
#      known|new FILE PATH RECORD
#
#  A failure is a result above the threshold (a line with the seed of the
#  test matrix and the ratio) or a routine that returned INFO != 0 (the
#  line with INFO and the next one with the seed).  RECORD is the line
#  (lines) with runs of blanks reduced to one blank and, for a result, the
#  ratio removed: it names the test matrix (seed, size, type) and the test,
#  and is the same on every machine.  PATH: the path of LAPACK's test
#  programs (DES, SBD, ...), taken from the next line with "out of" or
#  "passed the threshold" (the routine, STFSM, ..., if the line names no
#  path).  The tests of the RFP (rectangular full packed) format print no
#  seed; their results are the lines "Failure in ROUTINE, ..., test=
#  RATIO" and "SPFSV , UPLO=..., test(K)= RATIO".  "known": KNOWN
#  (known-failures.txt) contains "FILE PATH RECORD".  Last line: "known
#  failures: K".
#
set -u
TESTING=$1
KNOWN=$2
[ -f "$KNOWN" ] || { echo "error: $KNOWN not found" >&2; exit 2; }
awk '
function norm(s) {
    gsub(/[ \t\r]+/, " ", s)
    sub(/^ /, "", s)
    sub(/ $/, "", s)
    return s
}
#  the records of a section get the path when the section ends
function flush(path,    i, key) {
    for (i = 1; i <= nrec; i++) {
        key = file " " path " " rec[i]
        if (key in known) {
            print "known " key
            nknown++
        } else {
            print "new " key
        }
    }
    nrec = 0
}
FNR == 1 {
    if (FILENAME == KNOWN) {
        inknown = 1
    } else {
        if (inknown == 0 && file != "") {
            if (info != "") rec[++nrec] = norm(info)
            flush("?")
        }
        inknown = 0
        file = FILENAME
        sub(/.*\//, "", file)
        info = ""
    }
}
inknown {
    if ($0 !~ /^[ \t]*(#|$)/) known[norm($0)] = 1
    next
}
info != "" {
    #  the line after INFO: the matrix, if it has the seed
    if ($0 ~ /ISEED/) {
        rec[++nrec] = norm(info " " $0)
        info = ""
        next
    }
    rec[++nrec] = norm(info)
    info = ""
}
index($0, " INFO") {
    info = $0
    next
}
/seed= *[0-9]/ || / Failure in [A-Z]/ || /^ *[SDCZ]PF[A-Z]* *, UPLO=/ {
    s = norm($0)
    sub(/ ?[-+]?[0-9]*\.?[0-9]+([EeDd][-+]?[0-9]+)?$/, "", s)
    rec[++nrec] = s
    next
}
/out of|passed the threshold/ {
    path = "?"
    for (i = 1; i <= NF; i++) {
        t = $i
        sub(/:$/, "", t)
        if (t ~ /^[A-Z][A-Z0-9][A-Z0-9]$/) {
            path = t
            break
        }
    }
    if (path == "?") {
        for (i = 1; i <= NF; i++) {
            t = $i
            sub(/:$/, "", t)
            if (t ~ /^[A-Z][A-Z0-9]+$/) {
                path = t
                break
            }
        }
    }
    if (info != "") {
        rec[++nrec] = norm(info)
        info = ""
    }
    flush(path)
}
END {
    if (info != "") rec[++nrec] = norm(info)
    flush("?")
    print "known failures: " nknown + 0
}
' KNOWN="$KNOWN" "$KNOWN" "$TESTING"/*.out
