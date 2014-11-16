TOPDIR = ../../..

include $(TOPDIR)/make.inc

VARIANTS = $(patsubst config/make.%.inc, %, $(wildcard config/make.*.inc))

all : $(VARIANTS)

% :
	@echo VARIANTS=$(VARIANTS)
	make -C . -f Makefile.config $(MAKECMDGOALS) CONFIG=$@

.PHONY : clean
clean : $(VARIANTS)
