TOPDIR = ../..
include $(TOPDIR)/make.inc

VARIANTS       = $(patsubst config/make.%.inc, %, $(wildcard config/make.*.inc))
CMAKEFILE_PATH = $(TOPDIR)/interfaces/blas/C/

all : $(VARIANTS)
	@echo
	@echo "Completed tests for configuration variants: $(VARIANTS).";
	@echo "Results for tests on BLAS level 1, 2 and 3 functions are summerized in:";
	@echo
	@set -e; cd $(TOPDIR); ls -1 C*.SUMM.*

$(VARIANTS) :
	make -C $(CMAKEFILE_PATH) -f Makefile.config $(MAKECMDGOALS) CONFIG_PATH='$(CURDIR)/' CONFIG=$@
	make -C . -f Makefile.config $(MAKECMDGOALS) CONFIG=$@

clean : $(VARIANTS)
