ATL_C_SOURCEFILES = $(wildcard ATL_*.c)
ATL_F_SOURCEFILES = $(wildcard ATL_*.f)
ATL_OBJECTFILES = $(patsubst %.c, %.o, $(ATL_C_SOURCEFILES)) \
                  $(patsubst %.f, %.o, $(ATL_F_SOURCEFILES))

FC = gfortran
CC = gcc-4.8
#CC = gcc

CFLAGS = -c -DL2SIZE=4194304 -DAdd_ -DF77_INTEGER=int -DStringSunStyle \
         -DATL_SSE2 -DDREAL

LDFLAGS = -lm

BRANCH=$(shell git rev-parse --abbrev-ref HEAD)

#
#  Select the ATLAS library (or our faked ATLAS implementation)
#
ATLAS_LIB = ../libatlulmblas.a
#ATLAS_LIB = $(HOME)/work/ATLAS/build/lib/libatlas.a


#
#  Select the reference implementation
#
REF_LIB   = ../librefblas.a
#REF_LIB   = ../libulmblas.a
#REF_LIB   = $(HOME)/work/EigenBuild/blas/libeigen_blas_static.a
#REF_LIB   = $(HOME)/work/blis/lib/reference/libblis.a
#REF_LIB   = $(HOME)/work/blis/lib/dunnington/libblis.a
#REF_LIB   = $(HOME)/work/OpenBLAS/libopenblas.a

#MKLROOT  = /opt/intel/composer_xe_2011_sp1.8.269/mkl/
#REF_LIB = $(MKLROOT)/lib/libmkl_intel_lp64.a $(MKLROOT)/lib/libmkl_core.a $(MKLROOT)/lib/libmkl_sequential.a -lpthread -lm

all: xdl1blastst xdl3blastst

xdl1blastst : l1blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)
	$(FC) -o xdl1blastst l1blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)

xdl3blastst : l3blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)
	$(FC) -o xdl3blastst l3blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)

libtstatlas.a : $(ATL_OBJECTFILES)
	ar r libtstatlas.a $(ATL_OBJECTFILES)
	ranlib libtstatlas.a

.PHONY: plot
plot: xdl3blastst
	./xdl3blastst -N 100 2000 100 > report
	grep "\ \-\-\-\-\-$$"  report > refBLAS.dat
	grep PASS report > $(BRANCH).dat
	# List the branch for comparision
	grep -Fxq "refBLAS"   branches || echo refBLAS   >> branches
	grep -Fxq "$(BRANCH)" branches || echo $(BRANCH) >> branches
	gnuplot -e "BRANCHES=\"$(shell cat branches)\"" bench.gps
	open bench.png

clean:
	rm -f xdl[1-3]blastst libtstatlas.a l[1-3]blastst.o $(ATL_OBJECTFILES)
	rm -f report branches *.dat bench.png

