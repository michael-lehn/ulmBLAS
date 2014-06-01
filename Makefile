#
#  Note: bench must be the last directory.  That's because building the
#        benchmark suite requires that the BLAS libraries are already built.
#
DIRS = src refblas test bench


.PHONY: all
all: $(DIRS)

.PHONY: $(DIRS)
$(DIRS):
	make -C $@

# bench and test require an up-to-date ulmBLAS and reference BLAS
bench: src refblas
test: src refblas

.PHONY: clean
clean:
	-for dir in $(DIRS); do make -C $$dir clean; done

.PHONY: check
check: test
	make -C test check_ulm
