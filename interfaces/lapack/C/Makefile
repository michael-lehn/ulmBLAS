include ../../../make.inc

SOURCE_FILES     = $(wildcard *.cc)
DEP_FILES        = $(patsubst %.cc, %.d, $(SOURCE_FILES))

ATLAS_PREFIX     = atl_
OBJECT_FILES     = $(patsubst %.cc, %.o,                $(SOURCE_FILES))
ATL_OBJECT_FILES = $(patsubst %.cc, $(ATLAS_PREFIX)%.o, $(SOURCE_FILES))

TOPDIR           = ../../..
CXXFLAGS        += -I $(TOPDIR)

CLAPACK          = $(TOPDIR)/libclapack.a
ATLLAPACK        = $(TOPDIR)/libatllapack.a

CBLAS_HEADER     = $(TOPDIR)/$(CBLAS_HEADER_NAME)
ATLBLAS_HEADER   = $(TOPDIR)/$(ATLBLAS_HEADER_NAME)

CLAPACK_HEADER   = $(TOPDIR)/$(CLAPACK_HEADER_NAME)
ATLLAPACK_HEADER = $(TOPDIR)/$(ATLLAPACK_HEADER_NAME)

CLAPACK_DEFS     = -D'BLAS_HEADER=<$(CBLAS_HEADER_NAME)>' \
                   -D'LAPACK_HEADER=<$(CLAPACK_HEADER_NAME)>' \
                   -D'ULMLAPACK(x)=clapack_ \#\# x' \
                   -D'CLAPACK=ULMBLAS_CLAPACK_H'
ATLLAPACK_DEFS   = -D'BLAS_HEADER=<$(ATLBLAS_HEADER_NAME)>' \
                   -D'LAPACK_HEADER=<$(ATLLAPACK_HEADER_NAME)>' \
                   -D'ULMLAPACK(x)=ATL_ \#\# x' \
                   -D'CLAPACK=ULMBLAS_ATLLAPACK_H'

all : $(CLAPACK) $(ATLLAPACK) $(CLAPACK_HEADER) $(ATLLAPACK_HEADER)

$(CLAPACK) : $(OBJECT_FILES)
	ar cru $(CLAPACK) $(OBJECT_FILES)
	ranlib $(CLAPACK)

$(ATLLAPACK) : $(ATL_OBJECT_FILES)
	ar cru $(ATLLAPACK) $(ATL_OBJECT_FILES)
	ranlib $(ATLLAPACK)

$(CBLAS_HEADER) :
	make -C ../../blas/C $(CBLAS_HEADER)

$(ATLBLAS_HEADER) :
	make -C ../../blas/C $(ATLBLAS_HEADER)

$(CLAPACK_HEADER) : clapack.h.in $(CBLAS_HEADER)
	$(CXX) -x c++ $(CXXFLAGS) $(CLAPACK_DEFS) -E -CC clapack.h.in | grep -v "^#\s\d*\s" | sed  's,^ *#,#,' > $(CLAPACK_HEADER)

$(ATLLAPACK_HEADER) : clapack.h.in $(ATLBLAS_HEADER)
	$(CXX) -x c++ $(CXXFLAGS) $(ATLLAPACK_DEFS) -E -CC clapack.h.in | grep -v "^#\s\d*\s" | sed  's,^ *#,#,' > $(ATLLAPACK_HEADER)

%.o : %.cc $(CLAPACK_HEADER)
	$(CXX) $(CLAPACK_DEFS) $(CXXFLAGS) -c -o $@ $<

atl_%.o : %.cc $(ATLLAPACK_HEADER)
	$(CXX) $(ATLLAPACK_DEFS) $(CXXFLAGS) -c -o $@ $<

%.d : %.cc $(CLAPACK_HEADER)
	@set -e; $(CXX) $(CXXFLAGS) $(CLAPACK_DEFS) -MM $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $(ATLAS_PREFIX)\1.o $@ : ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$

ifneq ($(MAKECMDGOALS),clean)
-include $(DEP_FILES)
endif

clean :
	$(RM) $(DEP_FILES)
	$(RM) $(OBJECT_FILES)
	$(RM) $(ATL_OBJECT_FILES)
	$(RM) $(CLAPACK)
	$(RM) $(ATLLAPACK)
	$(RM) $(CLAPACK_HEADER) $(ATLLAPACK_HEADER)
