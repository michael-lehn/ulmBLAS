CC       = gcc-4.8
CFLAGS  += -Wall

SOURCE_FILES = $(wildcard *.c)
OBJECT_FILES = $(patsubst %.c,%.o,$(SOURCE_FILES))
ULMBLAS      = ../libulmblas.a

all : $(ULMBLAS)

$(ULMBLAS) : $(OBJECT_FILES)
	ar cru $(ULMBLAS) $(OBJECT_FILES)
	ranlib $(ULMBLAS)

%.o : %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean :
	rm -f $(OBJECT_FILES)
	rm -f $(ULMBLAS)
