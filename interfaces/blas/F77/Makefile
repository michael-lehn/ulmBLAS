include ../../../make.inc

CXXFLAGS += -I ../../..
CXXFLASG += -DUSE_TESTPARAM

SOURCE_FILES = $(wildcard *.cc) \
               $(wildcard asum/*.cc)  $(wildcard asum/*.f) \
               $(wildcard dot/*.cc)   $(wildcard dot/*.f) \
               $(wildcard iamax/*.cc) $(wildcard iamax/*.f) \
               $(wildcard nrm2/*.cc)  $(wildcard nrm2/*.f)
DEP_FILES    = $(patsubst %.cc, %.d, $(filter %.cc, $(SOURCE_FILES)))
OBJECT_FILES = $(patsubst %.f, %.o, $(patsubst %.cc, %.o, $(SOURCE_FILES)))

LIBBLAS      = ../../../libblas_check.a

all: $(LIBBLAS)

$(LIBBLAS) : $(OBJECT_FILES)
	ar cru $(LIBBLAS) $(OBJECT_FILES)
	ranlib $(LIBBLAS)

%.d : %.cc
	@set -e; $(CXX) $(CXXFLAGS) -MM $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$

-include $(DEP_FILES)

clean:
	$(RM) $(DEP_FILES)
	$(RM) $(OBJECT_FILES)
	$(RM) $(LIBBLAS)
