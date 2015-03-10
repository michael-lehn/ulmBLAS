      REAL FUNCTION SCASUM(N,CX,INCX)

      INTEGER           INCX, N
      COMPLEX           CX(*)
      REAL              TEMP


      CALL SCASUM_SUB( N, CX, INCX, TEMP )
      SCASUM = TEMP

      RETURN
      END
