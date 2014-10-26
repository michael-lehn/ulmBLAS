ATL_C_SOURCEFILES = $(wildcard ATL_*.c)
ATL_F_SOURCEFILES = $(wildcard ATL_*.f)
ATL_OBJECTFILES = $(patsubst %.c, %.o, $(ATL_C_SOURCEFILES)) \
                  $(patsubst %.f, %.o, $(ATL_F_SOURCEFILES))

FC = gfortran
CXX = g++
#CC = gcc

CFLAGS = -c -DL2SIZE=4194304 -DAdd_ -DF77_INTEGER=int -DStringSunStyle \
         -DATL_SSE2 -DDREAL

LDFLAGS = -lm

BRANCH=$(shell git rev-parse --abbrev-ref HEAD)

#
#  Select the ATLAS library (or our faked ATLAS implementation)
#
ATLAS_LIB = ../libatlblas.a
#ATLAS_LIB = $(HOME)/work/ATLAS/build/lib/libatlas.a

ATLAS_LAPACK_LIB = ../libatllapack.a

#
#  Select the reference implementation
#
REF_LIB_DEP   = ../librefblas.a
#REF_LIB_DEP   = ../libblas.a
#REF_LIB_DEP   = $(HOME)/work/eigen-3.2.1/blas/libeigen_blas_static.a
#REF_LIB_DEP   = $(HOME)/work/blis/lib/reference/libblis.a
#REF_LIB_DEP   = $(HOME)/work/blis/lib/dunnington/libblis.a
#REF_LIB_DEP   = $(HOME)/work/OpenBLAS/libopenblas.a
REF_LIB = $(REF_LIB_DEP)

#MKLROOT     = /opt/intel/composer_xe_2013_sp1.3.166/mkl/
#REF_LIB_DEP = $(MKLROOT)/lib/libmkl_intel_lp64.a
#REF_LIB     = $(REF_LIB_DEP) $(MKLROOT)/lib/libmkl_core.a \
              $(MKLROOT)/lib/libmkl_sequential.a -lpthread -lm

all: xdl1blastst xdl2blastst xdl3blastst xdlutst

xdl1blastst : l1blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB_DEP)
	$(CXX) -lstdc++ -o xdl1blastst l1blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)

xdl2blastst : l2blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB_DEP)
	$(CXX) -lstdc++ -o xdl2blastst l2blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)

xdl3blastst : l3blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB_DEP)
	$(CXX) -lstdc++ -o xdl3blastst l3blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)

xdlutst : lutst.o libtstatlas.a $(ATLAS_LIB) $(ATLAS_LAPACK_LIB) $(REF_LIB_DEP)
	$(CXX) -o xdlutst lutst.o libtstatlas.a $(ATLAS_LIB) $(ATLAS_LAPACK_LIB) $(REF_LIB)

libtstatlas.a : $(ATL_OBJECTFILES)
	ar r libtstatlas.a $(ATL_OBJECTFILES)
	ranlib libtstatlas.a

.PHONY: plot
plot: xdl3blastst
	./xdl3blastst -N 100 2000 100 > report
	grep "\ \-\-\-\-\-$$"  report > Netlib-refBLAS.dat
	grep PASS report > $(BRANCH).dat
	# List the branch for comparision
	grep -Fxq "Netlib-refBLAS" branches || echo Netlib-refBLAS >> branches
	grep -Fxq "$(BRANCH)"      branches || echo $(BRANCH)      >> branches
	gnuplot -e "BRANCHES=\"$(shell cat branches)\"" bench.gps
	xdg-open comp.svg || open comp.svg || echo "Benchmark plotted as bench/comp.svg"

clean:
	rm -f xdl[1-3]blastst libtstatlas.a l[1-3]blastst.o $(ATL_OBJECTFILES)
	rm -f report branches *.dat comp.svg

