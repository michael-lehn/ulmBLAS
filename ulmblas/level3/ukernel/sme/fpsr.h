#ifndef ULMBLAS_LEVEL3_UKERNEL_SME_FPSR_H
#define ULMBLAS_LEVEL3_UKERNEL_SME_FPSR_H 1

#include <cstdint>

//
//  Floating-point exception flags around streaming mode.
//
//  Entering and leaving streaming mode (SMSTART SM, SMSTOP SM) sets the
//  cumulative exception flags of FPSR: FPSR = 0x0800009F, i.e. IOC
//  (invalid), DZC (division by zero), OFC (overflow), UFC (underflow), IXC
//  (inexact), IDC (input denormal) and QC (saturation).  This is required
//  by the architecture (Arm ARM, pseudocode ResetSVEState) and was measured
//  on an Apple M5.  Without care every call of a streaming routine leaves
//  all flags set, and a caller testing them (fetestexcept, NumPy after each
//  matmul) sees exceptions that did not occur.
//
//  Therefore FPSR is saved in ordinary mode and restored after streaming
//  routines have run (FpsrRestore).  Every caller of a streaming routine
//  (the SME micro kernels and packing routines) has a FpsrRestore, and so
//  has every level 3 driver (ulmBLAS::FpsrScope, level3/fpsrscope.h).
//  They nest: only the outermost one of a thread saves and restores, the
//  inner ones only record that a streaming routine runs.  So a level 3
//  call restores FPSR once, not after each of its micro tiles and panels
//  (each restore waits for the floating-point operations in flight, about
//  10 ns: dgemm with n = 32 was 7 to 9% slower with one per streaming
//  call), and an inner FpsrRestore without an outer one (a driver without
//  FpsrScope) still restores.  The outermost one restores only if a
//  streaming routine ran: then the flags of the whole call are dropped,
//  also those of ordinary code in it, otherwise (e.g. the direct NEON path
//  of gemm) the flags are kept.  The streaming routines report no flags, as
//  Apple Accelerate (no flag from dgemm for overflow or Inf * 0 in any
//  tested case).  Reporting them would be neither reliable nor cheap:
//
//  - the outer products (FMOPA) are ZA-targeting operations and raise no
//    flags at all (Arm SME supplement, ZA-targeting floating-point
//    behaviors; on the M5 neither inexact nor overflow nor invalid);
//  - on macOS 26.6 FPSR was 0x0800009F again after an interruption of the
//    thread in streaming mode with live Z registers or ZA (about every
//    second run of 1 ms), so flags read in streaming mode can be wrong;
//  - reading FPSR in streaming mode waits for the SME unit: the kernels
//    were up to 25% slower for n = 32.
//
//  The kernels of the level 3 functions are the only code of ulmBLAS that
//  runs in streaming mode.
//
namespace ulmBLAS { namespace sme {

__attribute__((always_inline)) inline std::uint64_t
fpsr_get()
{
    std::uint64_t r;
    __asm__ volatile("mrs %0, fpsr" : "=r"(r) : : "memory");
    return r;
}

__attribute__((always_inline)) inline void
fpsr_set(std::uint64_t r)
{
    __asm__ volatile("msr fpsr, %0" : : "r"(r) : "memory");
}

//  per thread: an outermost FpsrRestore exists; a streaming routine ran
struct FpsrState
{
    bool active;
    bool streamed;
};

inline FpsrState &
fpsr_state()
{
    static thread_local FpsrState state = { false, false };
    return state;
}

//  In ordinary mode.  streams = true: around a call of a streaming
//  routine; false: around a level 3 driver (FpsrScope), which may or may
//  not call one.  The outermost one saves FPSR and restores it on
//  destruction if a streaming routine ran (see above).
class FpsrRestore
{
    public:
        explicit
        FpsrRestore(bool streams = true)
            : state(fpsr_state()), outermost(!state.active), fpsr(0)
        {
            if (outermost) {
                fpsr           = fpsr_get();
                state.active   = true;
                state.streamed = false;
            }
            state.streamed = state.streamed || streams;
        }

        ~FpsrRestore()
        {
            if (outermost) {
                if (state.streamed) {
                    fpsr_set(fpsr);
                }
                state.active   = false;
                state.streamed = false;
            }
        }

        FpsrRestore(const FpsrRestore &) = delete;
        FpsrRestore &operator=(const FpsrRestore &) = delete;

    private:
        FpsrState      &state;
        const bool     outermost;
        std::uint64_t  fpsr;
};

} } // namespace sme, ulmBLAS

#endif // ULMBLAS_LEVEL3_UKERNEL_SME_FPSR_H
