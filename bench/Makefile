ATL_C_SOURCEFILES = $(wildcard ATL_*.c)
ATL_F_SOURCEFILES = $(wildcard ATL_*.f)
ATL_OBJECTFILES = $(patsubst %.c, %.o, $(ATL_C_SOURCEFILES)) \
                  $(patsubst %.f, %.o, $(ATL_F_SOURCEFILES))

FC = gfortran
CC = gcc-4.8

CFLAGS = -c -DL2SIZE=4194304 -DAdd_ -DF77_INTEGER=int -DStringSunStyle \
         -DATL_SSE2 -DDREAL

ATLAS_LIB = ../libatlulmblas.a
REF_LIB   = ../librefblas.a

xdl1blastst : l1blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)
	$(CC) -o xdl1blastst l1blastst.o libtstatlas.a $(ATLAS_LIB) $(REF_LIB)

libtstatlas.a : $(ATL_OBJECTFILES)
	ar r libtstatlas.a $(ATL_OBJECTFILES)
	ranlib libtstatlas.a

clean:
	rm -f xdl1blastst libtstatlas.a l1blastst.o $(ATL_OBJECTFILES)
