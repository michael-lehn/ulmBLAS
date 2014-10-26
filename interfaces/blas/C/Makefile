include ../../../make.inc

SOURCE_FILES     = $(wildcard *.cc)
DEP_FILES        = $(patsubst %.cc, %.d, $(SOURCE_FILES))

ATLAS_PREFIX     = atl_
OBJECT_FILES     = $(patsubst %.cc, %.o,                $(SOURCE_FILES))
ATL_OBJECT_FILES = $(patsubst %.cc, $(ATLAS_PREFIX)%.o, $(SOURCE_FILES))

TOPDIR           = ../../..
CXXFLAGS        += -I $(TOPDIR)

CBLAS            = $(TOPDIR)/libcblas.a
ATLBLAS          = $(TOPDIR)/libatlblas.a

CBLAS_HEADER     = $(TOPDIR)/$(CBLAS_HEADER_NAME)
ATLBLAS_HEADER   = $(TOPDIR)/$(ATLBLAS_HEADER_NAME)

CBLAS_DEFS       = -D'BLAS_HEADER=<$(CBLAS_HEADER_NAME)>' \
                   -D'ULMBLAS(x)=cblas_ \#\# x' \
                   -D'CBLAS=ULMBLAS_CBLAS_H'
ATLBLAS_DEFS     = -D'BLAS_HEADER=<$(ATLBLAS_HEADER_NAME)>' \
                   -D'ULMBLAS(x)=ATL_ \#\# x' \
                   -D'CBLAS=ULMBLAS_ATLBLAS_H'

all : $(CBLAS) $(ATLBLAS) $(CBLAS_HEADER) $(ATLBLAS_HEADER)

$(CBLAS) : $(OBJECT_FILES)
	ar cru $(CBLAS) $(OBJECT_FILES)
	ranlib $(CBLAS)

$(ATLBLAS) : $(ATL_OBJECT_FILES)
	ar cru $(ATLBLAS) $(ATL_OBJECT_FILES)
	ranlib $(ATLBLAS)

$(CBLAS_HEADER) : cblas.h.in
	$(CXX) -x c++ $(CBLAS_DEFS) -E -CC cblas.h.in | grep -v "^#\s\d*\s" | sed  's,^ *#,#,' > $(CBLAS_HEADER)

$(ATLBLAS_HEADER) : cblas.h.in
	$(CXX) -x c++ $(ATLBLAS_DEFS) -E -CC cblas.h.in | grep -v "^#\s\d*\s" | sed  's,^ *#,#,' > $(ATLBLAS_HEADER)

%.o : %.cc $(CBLAS_HEADER)
	$(CXX) $(CBLAS_DEFS) $(CXXFLAGS) -c -o $@ $<

atl_%.o : %.cc $(ATLBLAS_HEADER)
	$(CXX) $(ATLBLAS_DEFS) $(CXXFLAGS) -c -o $@ $<

%.d : %.cc $(CBLAS_HEADER)
	@set -e; $(CXX) $(CXXFLAGS) $(CBLAS_DEFS) -MM $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $(ATLAS_PREFIX)\1.o $@ : ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$

ifneq ($(MAKECMDGOALS),clean)
-include $(DEP_FILES)
endif

clean :
	$(RM) $(DEP_FILES)
	$(RM) $(OBJECT_FILES)
	$(RM) $(ATL_OBJECT_FILES)
	$(RM) $(CBLAS)
	$(RM) $(ATLBLAS)
	$(RM) $(CBLAS_HEADER) $(ATLBLAS_HEADER)
