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

#
#  Select the ATLAS library (or our faked ATLAS implementation)
#
ATLAS_LIB = ../libatlulmblas.a

#
#  Select the reference implementation
#
REF_LIB   = ../librefblas.a

all: xdl1blastst xdl3blastst

xdl1blastst : l1blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)
	$(FC) -o xdl1blastst l1blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)

xdl3blastst : l3blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)
	$(FC) -o xdl3blastst l3blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)

libtstatlas.a : $(ATL_OBJECTFILES)
	ar r libtstatlas.a $(ATL_OBJECTFILES)
	ranlib libtstatlas.a

clean:
	rm -f xdl1blastst libtstatlas.a l1blastst.o $(ATL_OBJECTFILES)
