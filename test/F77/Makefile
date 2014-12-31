TOPDIR = ../..
include $(TOPDIR)/make.inc

VARIANTS         = $(patsubst config/make.%.inc, %, $(wildcard config/make.*.inc))
F77MAKEFILE_PATH = $(TOPDIR)/interfaces/blas/F77/

all : $(VARIANTS)
	@echo
	@echo "Completed tests for configuration variants: $(VARIANTS).";
	@echo "Results for tests on BLAS level 1, 2 and 3 functions are summerized in:";
	@echo
	@set -e; cd $(TOPDIR); ls -1 F77*.SUMM.*

$(VARIANTS) :
	make -C $(F77MAKEFILE_PATH) -f Makefile.config $(MAKECMDGOALS) CONFIG_PATH='$(CURDIR)/' CONFIG=$@
	make -C . -f Makefile.config $(MAKECMDGOALS) CONFIG=$@

.PHONY : clean
clean : $(VARIANTS)
