TOPDIR = ../..
include $(TOPDIR)/make.inc

VARIANTS = $(patsubst config/make.%.inc, %, $(wildcard config/make.*.inc))

TESTBASE_SOURCE_FILES = $(wildcard *.f)
TESTBASE_OBJECT_FILES = $(patsubst %.f, %.o, $(TESTBASE_SOURCE_FILES))


all : testbase $(VARIANTS)
	@echo
	@echo "Completed tests for configuration variants: $(VARIANTS).";
	@echo "Results for tests on BLAS level 1, 2 and 3 functions are summerized in:";
	@echo
	@set -e; cd $(TOPDIR); ls -1 *.SUMM.*

testbase: $(TESTBASE_OBJECT_FILES)

$(VARIANTS) :
	make -C . -f Makefile.config $(MAKECMDGOALS) CONFIG=$@

.PHONY : clean
clean : $(VARIANTS)
