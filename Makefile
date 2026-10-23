#
#  ulmBLAS requires GNU Make (3.81 or newer).
#
#  GNU Make reads GNUmakefile and ignores this file.  This file is read by
#  other implementations of make (e.g. BSD make) and only explains how to
#  build ulmBLAS.  It uses POSIX make syntax only.
#

all cblas clapack atlblas atllapack f77blas refblas check check-numeric \
check-netlib check_cblas check_f77blas bench benchmark-suite clean help: \
gnu-make-required

gnu-make-required:
	@echo "error: ulmBLAS must be built with GNU Make (3.81 or newer)." >&2; \
	if command -v gmake >/dev/null 2>&1; then \
	    echo "  GNU Make is installed as 'gmake', run: gmake $(.TARGETS)" >&2; \
	else \
	    echo "  Install GNU Make (often packaged as 'gmake') and run gmake." >&2; \
	fi; \
	exit 1
