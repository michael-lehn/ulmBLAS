TOPDIR = ..
include $(TOPDIR)/make.inc
include $(CONFIG_PATH)cases/make.inc
include $(CONFIG_PATH)cases/make.$(CASE).inc

CASE_DIR = $(DIR)/$(CASE)


GPS_MAKE_INC_DEPS = $(CONFIG_PATH)cases/make.inc \
                    $(CONFIG_PATH)cases/make.$(CASE).inc

GPS        = $(foreach variant, $(GPS_VARIANTS),\
                  $(CASE_DIR)/$(CASE)_$(variant).gps)
GPS_OUTPUT = $(foreach variant, $(GPS_VARIANTS),\
                  $(CASE_DIR)/$(CASE)_$(variant).$(GPS_OUTPUT_SUFFIX))

all : $(GPS_OUTPUT)

$(GPS) : gnuplot.in

$(CASE_DIR)/$(CASE)_%.$(GPS_OUTPUT_SUFFIX) : gnuplot.in $(GPS_MAKE_INC_DEPS)
	rm -f $@
	make -C . -f Makefile.variantplot $(MAKECMDGOALS) CASE=$(CASE) VARIANT=$* DIR=${@D} BLAS_VARIANTS="$(BLAS_VARIANTS)"
