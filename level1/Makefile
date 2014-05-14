CC       = gcc-4.8
CFLAGS  += -Wall -I.. -O3

SOURCE_FILES     = $(wildcard *.c)
OBJECT_FILES     = $(patsubst %.c,%.o,$(SOURCE_FILES))
ATL_OBJECT_FILES = $(patsubst %.c,atl_%.o,$(SOURCE_FILES))

ULMBLAS      = ../libulmblas.a
ATLULMBLAS   = ../libatlulmblas.a

all : $(ULMBLAS) $(ATLULMBLAS)

$(ULMBLAS) : $(OBJECT_FILES)
	ar cru $(ULMBLAS) $(OBJECT_FILES)
	ranlib $(ULMBLAS)

$(ATLULMBLAS) : $(ATL_OBJECT_FILES)
	ar cru $(ATLULMBLAS) $(ATL_OBJECT_FILES)
	ranlib $(ATLULMBLAS)

%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

atl_%.o : %.c
	$(CC) $(CFLAGS) -DFAKE_ATLAS -c -o $@ $<

clean :
	rm -f $(OBJECT_FILES)
	rm -f $(ATL_OBJECT_FILES)
	rm -f $(ULMBLAS)
	rm -f $(ATLULMBLAS)
