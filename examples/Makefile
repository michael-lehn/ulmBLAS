CFLAGS  += -I../src
LDFLAGS += -L.. -lulmblas

all: test-dgetrf

test-dgetrf: test-dgetrf.c ../libulmblas.a

clean:
	rm -f test-dgetrf
