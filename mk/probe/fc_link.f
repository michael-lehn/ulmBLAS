*     Probe: Fortran main program calling a C++ function, linked by the
*     C++ compiler (as done for the Netlib test drivers).
      PROGRAM PROBE
      INTEGER ULMPRB
      EXTERNAL ULMPRB
      WRITE( *, FMT = '(A)' )'Fortran I/O works'
      IF( ULMPRB( 21 ).NE.42 )STOP 1
      END
