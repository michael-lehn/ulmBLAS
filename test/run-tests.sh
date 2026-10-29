#!/bin/sh
#
# Test driver of ulmBLAS.  Invoked by 'make check' (see test/Makefile) in
# the directory test/ after all test programs have been built.
#
# Every test has a stable ID Tnnnn that is never reused.  Known bugs and open
# design questions are listed in test/known_issues.md with issue IDs #nnnn.
# Before the tests are run, the registry is validated: test IDs are unique,
# and the tests referring to an issue are exactly those listed in its entry.
# The validation covers the kinds of tests the run selects (see
# validate_registry): make check-netlib does not build the numeric tests, so
# registry entries of numeric tests are reported as not checked there.
#
# Kinds of tests:
#
#   numeric   C++ tests of the C interfaces (test/numeric, IDs T01nn-T06nn),
#             one executable per configuration.  Results are compared with
#             straightforward reference implementations using error bounds
#             (see test/numeric/testing.h).
#   netlib-F77, netlib-C
#             Netlib BLAS test suites for the Fortran 77 and the C interface
#             (test/F77, test/C, IDs T0001-T0012 and T0021-T0032).  A run
#             passes if the program exits with status 0, its output contains
#             no failure markers and shows that all routines were tested.
#   header    the generated headers can be used from C (IDs T0501, T0502).
#   example   the C++ examples of examples/ build and run with the flags of
#             the configuration and exit with status 0 (IDs T0520-T0529).
#   registry  self-tests of the registry validation (IDs T0510-T0515), run
#             once per invocation.
#
# All tests are run for each test configuration (block sizes, see
# test/config).
#
# Classification of results:
#
#   PASS    expected behavior
#   FAIL    an error
#   XFAIL   fails because of a known bug (OPEN issue of type BUG)
#   XPASS   test of a known bug passes: an error, update test and registry
#   DESIGN  open design question (OPEN issue of type DESIGN), observed
#           behavior as recorded
#   SKIP    not run (e.g. no Fortran compiler); with STRICT=1 an error
#
# Environment:
#
#   TEST_BUILD        build directory (build)
#   CONFIGS           test configurations
#   NUMERIC           yes: run the numeric tests
#   NETLIB_SUITES     Netlib suites to run ("C F77", "C", "F77" or empty)
#   NETLIB_SKIP_LOG   if NETLIB_SUITES is empty because the tools are
#                     missing: file with the explanation (then reported as
#                     SKIP)
#   TESTS             only run tests whose ID or name contains one of these
#                     strings
#   CC                C compiler and flags for the header tests (optional)
#   VERBOSE=1         print all results, not only problems
#   STRICT=1          skipped tests are errors
#
# Exit status: 0 if there was no FAIL, no XPASS, the registry is consistent
# and (with STRICT=1) nothing was skipped.  A log of all results is written to
# $TEST_BUILD/check.log.
#

LC_ALL=C
export LC_ALL

TEST_BUILD=${TEST_BUILD:-build}
registry=known_issues.md
log=$TEST_BUILD/check.log

n_pass=0
n_fail=0
n_xfail=0
n_xpass=0
n_design=0
n_skip=0
problems=

mkdir -p "$TEST_BUILD" || exit 2
: > "$log"

# selected ID NAME: is the test selected by TESTS?
selected()
{
    [ -z "$TESTS" ] && return 0
    for t in $TESTS; do
        case "$1 $2" in
            *"$t"*) return 0 ;;
        esac
    done
    return 1
}

# report STATUS ID CONFIG NAME [MESSAGE]: record and print a result
report()
{
    r_status=$1 r_id=$2 r_config=$3 r_name=$4 r_message=$5
    r_line=`printf '%-6s %s [%s] %s' "$r_status" "$r_id" "$r_config" "$r_name"`
    [ -n "$r_message" ] && r_line="$r_line: $r_message"
    echo "$r_line" >> "$log"
    case $r_status in
        PASS)   n_pass=`expr $n_pass + 1` ;;
        FAIL)   n_fail=`expr $n_fail + 1` ;;
        XFAIL)  n_xfail=`expr $n_xfail + 1` ;;
        XPASS)  n_xpass=`expr $n_xpass + 1` ;;
        DESIGN) n_design=`expr $n_design + 1` ;;
        SKIP)   n_skip=`expr $n_skip + 1` ;;
    esac
    case $r_status in
        PASS|DESIGN)
            [ -n "$VERBOSE" ] && echo "$r_line" ;;
        XFAIL)
            [ -n "$VERBOSE" ] && echo "$r_line"
            problems="$problems
$r_line" ;;
        *)
            echo "$r_line"
            problems="$problems
$r_line" ;;
    esac
    return 0
}

# Netlib test programs: ID suite program
netlib_tests="
T0001 F77 sblat1
T0002 F77 dblat1
T0003 F77 cblat1
T0004 F77 zblat1
T0005 F77 sblat2
T0006 F77 dblat2
T0007 F77 cblat2
T0008 F77 zblat2
T0009 F77 sblat3
T0010 F77 dblat3
T0011 F77 cblat3
T0012 F77 zblat3
T0013 F77 ublat1
T0021 C c_sblat1
T0022 C c_dblat1
T0023 C c_cblat1
T0024 C c_zblat1
T0025 C c_sblat2
T0026 C c_dblat2
T0027 C c_cblat2
T0028 C c_zblat2
T0029 C c_sblat3
T0030 C c_dblat3
T0031 C c_cblat3
T0032 C c_zblat3"

header_tests="
T0501 cblas.h
T0502 clapack.h"

example_tests="
T0520 gemm"

registry_tests="
T0510 registry
T0511 registry
T0512 registry
T0513 registry
T0514 registry
T0515 registry"

#
# Registry validation
#
# Each run validates the registry against the tests of the kinds (scopes) it
# selects: netlib always, numeric and header if the C++ tests are run
# (make check, check-numeric).  The kind of each test listed in the registry
# is given by its annotation 'Tnnnn (kind, CLASS)'.  A listed test of a
# selected kind must exist; a listed test of a kind that is not selected
# (e.g. a numeric test in make check-netlib) is reported as not checked.
#

# registry_ids SCOPES: print the test IDs of the given scopes as lines
# 'ID KIND ISSUE SCOPE' (KIND: TEST, XFAIL or DESIGN)
registry_ids()
{
    first=`echo $CONFIGS | awk '{print $1}'`
    numeric=$TEST_BUILD/$first/numeric/numeric_tests
    echo "$registry_tests" | awk 'NF {print $1, "TEST", 0, "registry"}'
    echo "$netlib_tests" | awk 'NF {print $1, "TEST", 0, "netlib"}'
    case " $1 " in
        *" header "*)
            echo "$header_tests" | awk 'NF {print $1, "TEST", 0, "header"}'
            echo "$example_tests" | awk 'NF {print $1, "TEST", 0, "example"}'
            ;;
    esac
    case " $1 " in
        *" numeric "*)
            if [ ! -x "$numeric" ]; then
                echo "error: numeric tests selected, but $numeric is not built" >&2
                return 1
            fi
            "$numeric" --list | awk '{print $1, $2, $3, "numeric"}' \
                || return 1
            ;;
    esac
}

# check_registry REGISTRY IDS SCOPES: check the registry against the test IDs
# (file written by registry_ids) of the selected scopes.  Prints errors and
# notes; exit status 1 if the registry is inconsistent.
check_registry()
{
    awk -v registry="$1" -v scopes=" $3 " '
        # tests: ID KIND ISSUE SCOPE
        FILENAME != registry {
            if ($1 in kind) {
                printf "duplicate test ID %s\n", $1; bad = 1
            }
            kind[$1] = $2; issue[$1] = $3 + 0
            if ($3 + 0) {
                refs[$3 + 0] = refs[$3 + 0] " " $1
            }
            next
        }
        # registry entries
        /^Issue #[0-9][0-9][0-9][0-9]:/ {
            cur = substr($2, 2, 4) + 0
            if (cur in itype) {
                printf "duplicate issue #%04d\n", cur; bad = 1
            }
            itype[cur] = "?"; istatus[cur] = "?"; covered[cur] = ""
            next
        }
        /^Type:/            { if (cur) itype[cur] = $2; next }
        /^Status:/          { if (cur) istatus[cur] = $2; next }
        # Covered by tests: T0001, T0002 (kind, CLASS), T0003 (kind, ...)
        # the annotation gives the kind of the preceding test IDs
        /^Covered by tests:/ {
            if (!cur) next
            pending = ""
            for (i = 4; i <= NF; ++i) {
                if ($i ~ /^T[0-9][0-9][0-9][0-9]/) {
                    t = substr($i, 1, 5)
                    covered[cur] = covered[cur] " " t
                    pending = pending " " t
                } else if ($i ~ /^\(/ && pending != "") {
                    k = $i; gsub(/[(),;]/, "", k)
                    n = split(pending, p, " ")
                    for (j = 1; j <= n; ++j) {
                        tkind[cur, p[j]] = k
                    }
                    pending = ""
                }
            }
            if (pending != "") {
                printf "issue #%04d: no kind given for test(s)%s\n", cur, \
                       pending
                bad = 1
            }
            next
        }
        END {
            for (t in kind) {
                if (kind[t] == "TEST") continue
                i = issue[t]
                want = (kind[t] == "XFAIL") ? "BUG" : "DESIGN"
                if (!(i in itype)) {
                    printf "test %s refers to unknown issue #%04d\n", t, i
                    bad = 1
                } else if (itype[i] != want || istatus[i] != "OPEN") {
                    printf "test %s (%s) refers to issue #%04d, which is " \
                           "not an OPEN %s\n", t, kind[t], i, want
                    bad = 1
                }
            }
            for (i in itype) {
                if (itype[i] != "BUG" && itype[i] != "DESIGN") {
                    printf "issue #%04d: invalid type\n", i; bad = 1
                }
                if (istatus[i] !~ /^(OPEN|RESOLVED|WONTFIX)$/) {
                    printf "issue #%04d: invalid status\n", i; bad = 1
                }
                n = split(covered[i], c, " ")
                for (k = 1; k <= n; ++k) {
                    tk = tkind[i, c[k]]
                    if (tk != "" && tk !~ /^(numeric|netlib|header|example)$/) {
                        printf "issue #%04d: unknown kind \"%s\" of test " \
                               "%s\n", i, tk, c[k]
                        bad = 1
                    }
                }
                if (istatus[i] != "OPEN") continue
                for (k = 1; k <= n; ++k) {
                    tk = tkind[i, c[k]]
                    if (c[k] in kind) {
                        if (issue[c[k]] != i) {
                            printf "issue #%04d lists %s, which does not " \
                                   "refer to it\n", i, c[k]
                            bad = 1
                        }
                    } else if (index(scopes, " " tk " ") == 0 && tk != "") {
                        printf "note: issue #%04d: test %s (%s) not checked " \
                               "(%s tests not selected)\n", i, c[k], tk, tk
                    } else {
                        printf "issue #%04d lists unknown test %s\n", i, c[k]
                        bad = 1
                    }
                }
                m = split(refs[i], r, " ")
                for (k = 1; k <= m; ++k) {
                    if (index(covered[i] " ", " " r[k] " ") == 0) {
                        printf "test %s refers to issue #%04d, which does " \
                               "not list it\n", r[k], i
                        bad = 1
                    }
                }
            }
            exit bad
        }' "$2" "$1"
}

# validate_registry: validate the registry for the tests selected by this run
validate_registry()
{
    scopes=netlib
    if [ "$NUMERIC" = yes ]; then
        scopes="netlib numeric header example"
    fi
    ids=$TEST_BUILD/registry.ids
    registry_ids "$scopes" > "$ids" || return 1
    check_registry "$registry" "$ids" "$scopes"
}

#
# Self-tests of the registry validation (independent of the configurations):
# the registry must be consistent for the scope of the standalone Netlib
# targets (make check-netlib, check_cblas, check_f77blas), and inconsistent
# registries must be rejected.
#
run_registry_tests()
{
    dir=$TEST_BUILD/registry-tests
    mkdir -p "$dir"
    registry_ids netlib > "$dir/netlib.ids"

    # broken_registry ENTRY: registry with an additional issue
    broken_registry()
    {
        cat "$registry" > "$dir/broken.md"
        printf '\n%s\n' "$1" >> "$dir/broken.md"
    }

    # expect_registry ID NAME RESULT SCOPES [ENTRY]: check_registry with
    # the scope SCOPES must give RESULT (ok or rejected)
    expect_registry()
    {
        e_id=$1 e_name=$2 e_want=$3 e_scopes=$4
        reg=$registry
        if [ -n "$5" ]; then
            broken_registry "$5"
            reg=$dir/broken.md
        fi
        selected "$e_id" "registry/$e_name" || return 0
        if check_registry "$reg" "$dir/netlib.ids" "$e_scopes" \
               > "$dir/$e_id.log" 2>&1
        then
            got=ok
        else
            got=rejected
        fi
        if [ "$got" = "$e_want" ]; then
            report PASS "$e_id" registry "registry/$e_name"
        else
            report FAIL "$e_id" registry "registry/$e_name" \
                "registry $got, expected $e_want, see $dir/$e_id.log"
        fi
    }

    expect_registry T0510 "consistent for the standalone Netlib targets" \
        ok netlib
    expect_registry T0511 "rejects a missing test of a selected kind" \
        rejected netlib \
        "Issue #9999: test
Type: BUG
Status: OPEN
Covered by tests: T0999 (netlib, XFAIL)"
    expect_registry T0512 "tolerates numeric tests if they are not selected" \
        ok netlib \
        "Issue #9999: test
Type: DESIGN
Status: OPEN
Covered by tests: T0998 (numeric, DESIGN)"
    expect_registry T0513 "rejects missing numeric tests if they are selected" \
        rejected "netlib numeric" \
        "Issue #9999: test
Type: DESIGN
Status: OPEN
Covered by tests: T0998 (numeric, DESIGN)"
    expect_registry T0514 "rejects tests without or with unknown kind" \
        rejected netlib \
        "Issue #9999: test
Type: BUG
Status: OPEN
Covered by tests: T0001, T0002 (bogus, XFAIL)"

    # duplicate test IDs are rejected
    if selected T0515 "registry/rejects duplicate test IDs"; then
        cat "$dir/netlib.ids" "$dir/netlib.ids" > "$dir/duplicate.ids"
        if check_registry "$registry" "$dir/duplicate.ids" netlib \
               > "$dir/T0515.log" 2>&1
        then
            report FAIL T0515 registry "registry/rejects duplicate test IDs" \
                "duplicate IDs accepted, see $dir/T0515.log"
        else
            report PASS T0515 registry "registry/rejects duplicate test IDs"
        fi
    fi
}

#
# Numeric tests
#
run_numeric()
{
    config=$1
    exe=$TEST_BUILD/$config/numeric/numeric_tests
    out=$TEST_BUILD/$config/numeric/results.txt

    if [ ! -x "$exe" ]; then
        report FAIL T0100-T0699 "$config" numeric "$exe not built"
        return
    fi
    filter=
    if [ -n "$TESTS" ]; then
        filter=$TESTS
    fi
    "$exe" $filter > "$out" 2>&1
    status=$?

    # result lines: STATUS ID NAME[ (#issue)][: message], continuation lines
    # of messages are indented
    while IFS= read -r line; do
        case $line in
            "PASS "*|"FAIL "*|"XFAIL "*|"XPASS "*|"DESIGN "*)
                st=`echo "$line" | awk '{print $1}'`
                id=`echo "$line" | awk '{print $2}'`
                rest=`echo "$line" | sed 's/^[A-Z]* *T[0-9]* //'`
                name=`echo "$rest" | sed 's/: .*//'`
                msg=`echo "$rest" | sed -n 's/^[^:]*: //p'`
                report "$st" "$id" "$config" "numeric/$name" "$msg"
                ;;
        esac
    done < "$out"

    if ! grep -q '^END$' "$out"; then
        report FAIL T0100-T0699 "$config" "numeric" \
            "test program terminated abnormally (status $status), see $out"
    elif [ $status -ne 0 ] && ! grep -q '^FAIL \|^XPASS ' "$out"; then
        report FAIL T0100-T0699 "$config" "numeric" \
            "unexpected exit status $status, see $out"
    fi
}

#
# Netlib test programs
#
# check_netlib OUTPUT LEVEL: evaluate the output of a Netlib test program,
# print a message if it failed
check_netlib()
{
    out=$1
    level=$2
    if grep -q -E 'FAIL|FATAL|SUSPECT|ILLEGAL|NOT TESTED|ERROR [^E]' "$out"
    then
        grep -E 'FAIL|FATAL|SUSPECT|ILLEGAL|NOT TESTED|ERROR [^E]' "$out" \
            | head -n 1 | sed 's/^ *//'
        return
    fi
    if [ "$level" = 1 ]; then
        tested=`grep -c 'Test of subprogram number' "$out"`
        passed=`grep -c -- '----- PASS -----' "$out"`
        if [ "$tested" -eq 0 ] || [ "$tested" -ne "$passed" ]; then
            echo "$passed of $tested subprograms passed"
        fi
    else
        if ! grep -q 'END OF TESTS' "$out"; then
            echo "incomplete output (no END OF TESTS)"
        elif ! grep -q 'PASSED THE' "$out"; then
            echo "no routine passed"
        fi
    fi
}

run_netlib()
{
    config=$1
    suite=$2
    dir=$TEST_BUILD/$config/netlib-$suite

    echo "$netlib_tests" | while read id s program; do
        [ -n "$id" ] && [ "$s" = "$suite" ] || continue
        selected "$id" "netlib-$suite/$program" || continue
        echo "$id $program"
    done > "$TEST_BUILD/$config.netlib-$suite.list"

    while read id program; do
        name=netlib-$suite/$program
        exe=$dir/$program
        level=`echo "$program" | sed 's/.*blat//'`
        out=$dir/$program.out
        if [ ! -x "$exe" ]; then
            report FAIL "$id" "$config" "$name" "$exe not built"
            continue
        fi
        # level 2 and 3 programs read their parameters from standard input
        # (test/C/[sdcz]in[23], test/F77/*.dat.in); the Fortran 77 programs
        # write their results to a summary file named in the input
        prefix=`echo "$program" | sed 's/c_//; s/blat.*//'`
        case $suite$level in
            C1|F771)  input=/dev/null ;;
            C*)       input=`pwd`/C/${prefix}in$level ;;
            F77*)     input=$program.dat ;;
        esac
        summary=`echo "$program" | tr 'a-z' 'A-Z'`.SUMM
        rm -f "$dir/$summary"
        (cd "$dir" && "./$program" < "$input" > "$program.out" 2>&1)
        status=$?
        if [ -f "$dir/$summary" ]; then
            cat "$dir/$summary" >> "$out"
        fi
        if [ $status -ne 0 ]; then
            report FAIL "$id" "$config" "$name" \
                "exit status $status, see $out"
            continue
        fi
        message=`check_netlib "$out" "$level"`
        if [ -n "$message" ]; then
            report FAIL "$id" "$config" "$name" "$message (see $out)"
        else
            report PASS "$id" "$config" "$name"
        fi
    done < "$TEST_BUILD/$config.netlib-$suite.list"
    rm -f "$TEST_BUILD/$config.netlib-$suite.list"
}

skip_netlib()
{
    config=$1
    reason=$2
    only=${3:-}
    echo "$netlib_tests" | while read id suite program; do
        [ -n "$id" ] || continue
        [ -z "$only" ] || [ "$suite" = "$only" ] || continue
        selected "$id" "netlib-$suite/$program" || continue
        echo "SKIP $id $suite/$program"
    done > "$TEST_BUILD/skip.list"
    while read status id name; do
        report SKIP "$id" "$config" "netlib-$name" "$reason"
    done < "$TEST_BUILD/skip.list"
    rm -f "$TEST_BUILD/skip.list"
}

#
# Header tests: the generated headers compile as C
#
# skip_config CONFIG REASON: report the selected tests of a configuration
# that is not supported here (CONFIG_REQUIRES, see Makefile.config) as SKIP.
# The IDs of the numeric tests are listed by the program of the first
# configuration (as for the registry).
skip_config()
{
    config=$1
    reason=$2
    if [ "$NUMERIC" = yes ]; then
        first=`echo $CONFIGS | awk '{print $1}'`
        numeric=$TEST_BUILD/$first/numeric/numeric_tests
        if [ -x "$numeric" ]; then
            "$numeric" --list | while read id kind issue name; do
                selected "$id" "numeric/$name" && echo "$id numeric/$name"
            done > "$TEST_BUILD/skip.list"
        else
            echo "T0100-T0699 numeric" > "$TEST_BUILD/skip.list"
        fi
        echo "$header_tests" | while read id header; do
            [ -n "$id" ] || continue
            selected "$id" "header/$header" && echo "$id header/$header"
        done >> "$TEST_BUILD/skip.list"
        echo "$example_tests" | while read id example; do
            [ -n "$id" ] || continue
            selected "$id" "example/$example" && echo "$id example/$example"
        done >> "$TEST_BUILD/skip.list"
        while read id name; do
            report SKIP "$id" "$config" "$name" "$reason"
        done < "$TEST_BUILD/skip.list"
        rm -f "$TEST_BUILD/skip.list"
    fi
    for suite in $NETLIB_SUITES; do
        echo "$netlib_tests" | while read id s program; do
            [ "$s" = "$suite" ] || continue
            selected "$id" "netlib-$s/$program" && echo "$id $s/$program"
        done > "$TEST_BUILD/skip.list"
        while read id name; do
            report SKIP "$id" "$config" "netlib-$name" "$reason"
        done < "$TEST_BUILD/skip.list"
        rm -f "$TEST_BUILD/skip.list"
    done
}

run_headers()
{
    config=$1
    lib=$TEST_BUILD/$config/lib
    echo "$header_tests" | while read id header; do
        [ -n "$id" ] || continue
        selected "$id" "header/$header" || continue
        echo "$id $header"
    done > "$TEST_BUILD/headers.list"

    while read id header; do
        name=header/$header
        if [ -z "$CC" ] || ! command -v `echo $CC | awk '{print $1}'` \
                                > /dev/null 2>&1
        then
            report SKIP "$id" "$config" "$name" "no C compiler (CC)"
            continue
        fi
        src=$TEST_BUILD/$config/header_test.c
        printf '#include <%s>\nint main(void) { return 0; }\n' "$header" \
            > "$src"
        if $CC -Wall -I"$lib" -c -o "$src.o" "$src" \
                > "$TEST_BUILD/$config/header_test.log" 2>&1
        then
            report PASS "$id" "$config" "$name"
        else
            report FAIL "$id" "$config" "$name" \
                "does not compile as C, see $TEST_BUILD/$config/header_test.log"
        fi
        rm -f "$src" "$src.o"
    done < "$TEST_BUILD/headers.list"
    rm -f "$TEST_BUILD/headers.list"
}

# run_examples CONFIG: the examples built by Makefile.config must run and
# exit with status 0
run_examples()
{
    config=$1
    echo "$example_tests" | while read id example; do
        [ -n "$id" ] || continue
        selected "$id" "example/$example" || continue
        echo "$id $example"
    done > "$TEST_BUILD/examples.list"

    while read id example; do
        name=example/$example
        exe=$TEST_BUILD/$config/examples/$example
        out=$TEST_BUILD/$config/examples/$example.out
        if [ ! -x "$exe" ]; then
            report FAIL "$id" "$config" "$name" "$exe not built"
        elif "$exe" > "$out" 2>&1; then
            report PASS "$id" "$config" "$name"
        else
            report FAIL "$id" "$config" "$name" \
                "exit status $?, see $out"
        fi
    done < "$TEST_BUILD/examples.list"
    rm -f "$TEST_BUILD/examples.list"
}

#
# Main
#
registry_ok=yes
if ! validate_registry > "$TEST_BUILD/registry.log" 2>&1; then
    registry_ok=no
    echo "error: test registry ($registry) is inconsistent:"
    sed 's/^/  /' "$TEST_BUILD/registry.log"
else
    sed -n 's/^note: /note: registry: /p' "$TEST_BUILD/registry.log"
fi
run_registry_tests

skip_reason=
if [ -z "$NETLIB_SUITES" ] && [ -n "$NETLIB_SKIP_LOG" ] \
   && [ -s "$NETLIB_SKIP_LOG" ]
then
    skip_reason="tools not usable, see $NETLIB_SKIP_LOG"
fi

for config in $CONFIGS; do
    echo "Running tests for configuration $config"
    if [ -f "$TEST_BUILD/$config/unsupported" ]; then
        skip_config "$config" "`cat "$TEST_BUILD/$config/unsupported"`"
        continue
    fi
    if [ "$NUMERIC" = yes ]; then
        run_numeric "$config"
        run_headers "$config"
        run_examples "$config"
    fi
    for suite in $NETLIB_SUITES; do
        if [ -f "$TEST_BUILD/$config/netlib-$suite.skip" ]; then
            skip_netlib "$config" \
                        "`cat "$TEST_BUILD/$config/netlib-$suite.skip"`" "$suite"
        else
            run_netlib "$config" "$suite"
        fi
    done
    if [ -n "$skip_reason" ]; then
        skip_netlib "$config" "$skip_reason"
    fi
done

echo
echo "Summary ($CONFIGS):"
printf '  PASS %d, FAIL %d, XFAIL %d, XPASS %d, DESIGN %d, SKIP %d\n' \
       $n_pass $n_fail $n_xfail $n_xpass $n_design $n_skip
if [ -n "$problems" ] && [ -z "$VERBOSE" ]; then
    echo
    echo "Results other than PASS and DESIGN:"
    echo "$problems" | awk 'NF' | sort -s -k2,2 | sed 's/^/  /'
fi
if [ -n "$skip_reason" ]; then
    echo
    echo "WARNING: the Netlib test suites were SKIPPED: a Fortran compiler,"
    echo "a C compiler and linking of Fortran with C++ are needed."
    sed 's/^/  /' "$NETLIB_SKIP_LOG"
fi
echo
echo "Known issues: test/$registry.  All results: test/$log"

[ $n_fail -eq 0 ] && [ $n_xpass -eq 0 ] && [ $registry_ok = yes ] \
    || exit 1
if [ -n "$STRICT" ] && [ $n_skip -ne 0 ]; then
    echo "error: $n_skip tests skipped (STRICT=1)"
    exit 1
fi
exit 0
