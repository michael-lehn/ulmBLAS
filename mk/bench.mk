#
#  The full check with the ATLAS (Automatically Tuned Linear Algebra
#  Software) testers (make check-large) and the ATLAS timing programs (make
#  bench, long running).
#
#  make check-large: the testers of bench/check, built against the static
#  libraries of the top directory (libcblas.a, libf77blas.a,
#  librefblas.a; needs FC and python3), in CHECK_BUILD (relative to bench/,
#  default check-build).  The runs of bench/check/list.sh whose line
#  matches CHECK_PATTERN (a grep pattern, default all 876) run CHECK_SLOTS
#  at a time (default: the number of CPUs (central processing units)) with
#  CHECK_THREADS threads each (default 1; all: the default of ulmBLAS).
#  Results in CHECK_OUT (relative to bench/, default check-results:
#  summary.txt, the output of every run).  Fails on a FAIL, an error message, a run that did not
#  finish and an incomplete check (bench/check/evaluate.sh).
#
CHECK_BUILD   ?= check-build
CHECK_OUT     ?= check-results
CHECK_THREADS ?= 1
CHECK_SLOTS   ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
CHECK_PATTERN ?= .

check-large : $(top.cblas.all) $(top.f77.all) $(refblas.lib) | tool-ar
	cd bench && CC="$(CC)" FC="$(FC)" $(SHELL) check/build.sh .. $(CHECK_BUILD)
	cd bench && rm -rf $(CHECK_OUT)
	cd bench && $(SHELL) check/run.sh $(CHECK_BUILD) $(CHECK_OUT) \
	    $(CHECK_THREADS) $(CHECK_SLOTS) '$(CHECK_PATTERN)'
	cd bench && $(SHELL) check/evaluate.sh $(CHECK_OUT)

#
#  make bench: the ATLAS timing programs for each BLAS variant
#  (bench/config/make.VARIANT.inc: the libraries they are linked with) and
#  each case (bench/cases/make.CASE.inc: program, sizes, plots).
#
#  USE_BLAS_VARIANTS  BLAS libraries compared.  Default: ulmBLAS and netlib
#                     (Netlib reference BLAS).  The variants atlas and mkl
#                     need ATLAS_LIBDIR and MKLROOT.
#  USE_BENCH_CASES    benchmark cases, default: all
#
#  Results: bench/benchmarks/CASE/VARIANT.summ, plots (if gnuplot is found)
#  bench/benchmarks/CASE/*.svg.  make benchmark-suite: only the programs
#  (bench/bench.VARIANT/).
#
BLAS_VARIANTS = $(patsubst bench/config/make.%.inc,%,\
                           $(wildcard bench/config/make.*.inc))
BENCH_CASES   = $(patsubst bench/cases/make.%.inc,%,$(filter-out \
                    bench/cases/make.inc,$(wildcard bench/cases/make.*.inc)))

USE_BLAS_VARIANTS ?= ulmBLAS netlib
USE_BENCH_CASES   ?= $(BENCH_CASES)

GNUPLOT           ?= gnuplot

bench.clean := bench/bench.* bench/benchmarks bench/libtstatlas.a \
               bench/ATL_*.o bench/$(CHECK_BUILD) bench/$(CHECK_OUT)

bench.goals := $(filter bench benchmark-suite,$(MAKECMDGOALS))

ifneq ($(bench.goals),)

#  the auxiliary functions of the ATLAS testers
bench.atl.o := $(patsubst %.c,%.o,$(wildcard bench/ATL_*.c)) \
               $(patsubst %.f,%.o,$(wildcard bench/ATL_*.f))
bench.libtst := bench/libtstatlas.a

bench/ATL_%.o : bench/ATL_%.c | tool-cc
	$(CC) $(BENCH_CFLAGS) $(ULMBLAS_CFLAGS) $(CFLAGS) -o $@ $<

bench/ATL_%.o : bench/ATL_%.f | tool-fc
	$(FC) $(FFLAGS) -c -o $@ $<

$(bench.libtst) : $(bench.atl.o)
	$(call make_archive,$@,$(bench.atl.o))

#
#  $(eval $(call bench_read,FILE,PREFIX,NAMES)): the variables NAMES set
#  by FILE as PREFIX.NAME; a name the file does not set gets the value
#  bench.unset (variables of the same name from an earlier file do not
#  count)
#
bench.unset := __bench_unset__
define bench_read
$$(foreach v,$3,$$(eval $$v := $$(bench.unset)))
include $1
$$(foreach v,$3,$$(eval $2.$$v := $$(value $$v)))
endef
#  the value of a variable read by bench_read, or $2 if the file did not
#  set it
bench_get = $(if $(filter $(bench.unset),$(firstword $($1))),$2,$($1))
bench_set = $(if $(filter $(bench.unset),$(firstword $($1))),,yes)

#  the variants: their libraries and the lines of the reports that make
#  the summaries (GREP_OPTS)
bench.vnames := ATLAS_BLAS_LIB_NAMES ATLAS_LAPACK_LIB_NAMES REF_BLAS_LIB_NAMES \
                ATLAS_BLAS_LIBS ATLAS_LAPACK_LIBS REF_BLAS_LIBS GREP_OPTS \
                $(foreach v,$(BLAS_VARIANTS),$(v)_NAME)

define bench_variant
$$(eval $$(call bench_read,bench/config/make.$1.inc,bench.v.$1,$$(bench.vnames)))
bench.$1.dir  := bench/bench.$1
bench.$1.libs := $$(call bench_get,bench.v.$1.ATLAS_LAPACK_LIBS,$$(addprefix ./,$$(call bench_get,bench.v.$1.ATLAS_LAPACK_LIB_NAMES))) \
                 $$(call bench_get,bench.v.$1.ATLAS_BLAS_LIBS,$$(addprefix ./,$$(call bench_get,bench.v.$1.ATLAS_BLAS_LIB_NAMES))) \
                 $$(call bench_get,bench.v.$1.REF_BLAS_LIBS,$$(addprefix ./,$$(call bench_get,bench.v.$1.REF_BLAS_LIB_NAMES)))
bench.$1.name := $$(call bench_get,bench.v.$1.$1_NAME,$1)
bench.$1.grep := $$(call bench_get,bench.v.$1.GREP_OPTS)
bench.$1.x    := $$(addprefix $$(bench.$1.dir)/xd,l1blastst l2blastst l3blastst lutst)

$$(bench.$1.dir)/%.o : bench/%.c | tool-cc
	@mkdir -p $$(@D)
	$$(CC) $$(BENCH_CFLAGS) $$(ULMBLAS_CFLAGS) $$(CFLAGS) -o $$@ $$<

$$(bench.$1.x) : $$(bench.$1.dir)/xd% : $$(bench.$1.dir)/%.o $$(bench.libtst) \
                 $$(filter %.a,$$(bench.$1.libs)) | tool-cxx-link tool-fc-link
	$$(CXX) $$(ULMBLAS_CXXFLAGS) $$(CXXFLAGS) $$(LDFLAGS) -o $$@ $$< \
	       $$(bench.libtst) $$(bench.$1.libs) $$(LDLIBS) $$(FLIBS)
endef

#  the cases: program, options, columns of the plots; GPS_* (the values
#  that replace the placeholders of bench/gnuplot.in) also from
#  bench/cases/make.inc, per plot variant P overridden by GPS_P_*
bench.gpsnames := TERMINAL OUTPUT_SUFFIX XLABEL YLABEL YRANGE_MIN YRANGE_MAX \
                  LOGSCALEX LOGSCALEY TITLE
bench.cnames = BENCH_CMD BENCH_OPT EXTRACT_VARIANTS GPS_VARIANTS EXTRACT_X \
               EXTRACT_Y $(foreach n,$(bench.gpsnames),GPS_$n) \
               $(foreach p,$(bench.pvariants),EXTRACT_$p_X EXTRACT_$p_Y \
                   $(foreach n,$(bench.gpsnames),GPS_$p_$n))
#  the plot variants of all cases (their names appear in the variables)
bench.pvariants := $(sort $(shell sed -n 's/^EXTRACT_VARIANTS *[:?]*= *//p' \
                                    bench/cases/make.*.inc))

define bench_case
$$(eval $$(call bench_read,bench/cases/make.inc,bench.c0,$$(bench.cnames)))
$$(eval $$(call bench_read,bench/cases/make.$1.inc,bench.c.$1,$$(bench.cnames)))
bench.$1.dir := bench/benchmarks/$1

$$(foreach v,$$(USE_BLAS_VARIANTS),$$(eval $$(call bench_run,$1,$$v)))

#  plots: one per plot variant P of the case (GPS_VARIANTS)
bench.$1.plots :=
$$(foreach p,$$(call bench_get,bench.c.$1.GPS_VARIANTS),$$(eval $$(call bench_plot,$1,$$p)))
endef

#  $(call bench_run,CASE,VARIANT): the report and the summary.  The
#  benchmark is run again when the program (and thus a library it is
#  linked with) or the configuration has changed.  The ATLAS testers check
#  every result against the reference BLAS (-T 1, the default): a FAIL line
#  fails the benchmark, and the report stays in REPORT.tmp.
define bench_run
bench.summaries += $$(bench.$1.dir)/$2.summ

$$(bench.$1.dir)/$2.report : $$(bench.$2.dir)/$$(call bench_get,bench.c.$1.BENCH_CMD) \
                             bench/cases/make.$1.inc bench/config/make.$2.inc
	@mkdir -p $$(@D)
	set -e; $$< $$(call bench_get,bench.c.$1.BENCH_OPT) > $$@.tmp
	@if grep ' FAIL *$$$$' $$@.tmp; then \
	    echo "error: results above the threshold (above), see $$@.tmp"; \
	    exit 1; \
	fi
	mv $$@.tmp $$@

$$(bench.$1.dir)/$2.summ : $$(bench.$1.dir)/$2.report
	grep $$(bench.$2.grep) $$< > $$@
endef

#  the value of placeholder N of plot variant P of case C: GPS_P_N of the
#  case, GPS_N of the case, GPS_N of cases/make.inc
bench_gps = $(if $(call bench_set,bench.c.$1.GPS_$2_$3),$(bench.c.$1.GPS_$2_$3),$(if \
                $(call bench_set,bench.c.$1.GPS_$3),$(bench.c.$1.GPS_$3),$(call \
                bench_get,bench.c0.GPS_$3)))
bench_extract = $(call bench_get,bench.c.$1.EXTRACT_$2_$3,$(call bench_get,bench.c.$1.EXTRACT_$3))

#  $(call bench_plot,CASE,P)
define bench_plot
bench.$1.$2.svg := $$(bench.$1.dir)/$1_$2.$$(call bench_gps,$1,$2,OUTPUT_SUFFIX)
bench.$1.$2.gps := $$(bench.$1.dir)/$1_$2.gps
bench.$1.$2.data := $$(foreach v,$$(USE_BLAS_VARIANTS),$$(bench.$1.dir)/$$v.summ.$2)
bench.plots += $$(bench.$1.$2.svg)

$$(bench.$1.$2.data) : $$(bench.$1.dir)/%.summ.$2 : $$(bench.$1.dir)/%.summ
	awk ' { print $$$$$$(call bench_extract,$1,$2,X) " " $$$$$$(call bench_extract,$1,$2,Y) }' $$< > $$@

$$(bench.$1.$2.gps) : $$(bench.$1.$2.data) bench/gnuplot.in \
                      bench/cases/make.inc bench/cases/make.$1.inc
	sed -e 's;GPS_TERMINAL;$$(call bench_gps,$1,$2,TERMINAL);g' \
	    -e 's;GPS_OUTPUT;"$$(bench.$1.$2.svg)";g' \
	    -e 's;GPS_XLABEL;$$(call bench_gps,$1,$2,XLABEL);g' \
	    -e 's;GPS_YLABEL;$$(call bench_gps,$1,$2,YLABEL);g' \
	    -e 's;GPS_YRANGE_MIN;$$(call bench_gps,$1,$2,YRANGE_MIN);g' \
	    -e 's;GPS_YRANGE_MAX;$$(call bench_gps,$1,$2,YRANGE_MAX);g' \
	    -e 's;GPS_LOGSCALEX;$$(call bench_gps,$1,$2,LOGSCALEX);g' \
	    -e 's;GPS_LOGSCALEY;$$(call bench_gps,$1,$2,LOGSCALEY);g' \
	    -e 's;GPS_TITLE;$$(call bench_gps,$1,$2,TITLE);g' \
	    -e 's;GPS_PLOT;plot $$(foreach v,$$(USE_BLAS_VARIANTS),"$$(bench.$1.dir)/$$v.summ.$2" using 1:2 with linespoints lw 3 title $$(bench.$$v.name)$$(COMMA));g' \
	    bench/gnuplot.in > $$@

$$(bench.$1.$2.svg) : $$(bench.$1.$2.gps)
	@command -v $$(GNUPLOT) >/dev/null 2>&1 \
	 || { echo "error: '$$(GNUPLOT)' not found, needed for plots" >&2; exit 1; }
	$$(GNUPLOT) $$<
endef

$(foreach v,$(USE_BLAS_VARIANTS),$(eval $(call bench_variant,$v)))
$(foreach c,$(USE_BENCH_CASES),$(eval $(call bench_case,$c)))

HAVE_GNUPLOT := $(shell command -v $(GNUPLOT) >/dev/null 2>&1 && echo yes)
endif

#  the libraries of ulmBLAS the variants use
BENCH_LIBS = cblas atlblas atllapack f77blas refblas

benchmark-suite : $(BENCH_LIBS) $(foreach v,$(USE_BLAS_VARIANTS),$(bench.$v.x))

bench : $(BENCH_LIBS) $(bench.summaries) $(if $(HAVE_GNUPLOT),$(bench.plots))
	@$(if $(HAVE_GNUPLOT),:,echo "note: '$(GNUPLOT)' not found, no plots created (set GNUPLOT=...)."; \
	  echo "      Results: $(bench.summaries)")
