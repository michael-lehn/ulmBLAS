CC       = gcc-4.8
#CC       = clang
#CC       = gcc
CFLAGS  += -Wall -I. -O3

DIRS = auxiliary level1 level3

SOURCE_FILES     = $(foreach DIR, $(DIRS), $(wildcard $(DIR)/*.c))
OBJECT_FILES     = $(foreach DIR, $(DIRS), \
                       $(patsubst $(DIR)/%.c,$(DIR)/%.o, \
                           $(filter $(DIR)/%.c, $(SOURCE_FILES))))
ATL_OBJECT_FILES = $(foreach DIR, $(DIRS), \
                       $(patsubst $(DIR)/%.c,$(DIR)/atl_%.o, \
                           $(filter $(DIR)/%.c, $(SOURCE_FILES))))

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
	$(CC) $(CFLAGS) -DULM_BLOCKED -c -o $@ $<

atl_%.o : %.c
	$(CC) $(CFLAGS) -DULM_BLOCKED -DFAKE_ATLAS -c -o $@ $<

clean :
	rm -f $(OBJECT_FILES)
	rm -f $(ATL_OBJECT_FILES)
	rm -f $(ULMBLAS)
	rm -f $(ATLULMBLAS)
