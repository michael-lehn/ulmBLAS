      REAL FUNCTION SDOT( N, X, INCX, Y, INCY )

      INTEGER            INCX, INCY, N
      REAL               X( * ), Y( * )

      REAL               TEMP
      EXTERNAL           SDOT_SUB

      CALL SDOT_SUB( N, X, INCX, Y, INCY, TEMP )
      SDOT = TEMP

      RETURN
      END
