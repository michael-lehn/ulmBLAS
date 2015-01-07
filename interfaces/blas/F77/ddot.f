      DOUBLE PRECISION FUNCTION DDOT( N, X, INCX, Y, INCY )

      INTEGER            INCX, INCY, N
      DOUBLE PRECISION   X( * ), Y( * )

      DOUBLE PRECISION   TEMP
      EXTERNAL           DDOT_SUB

      CALL DDOT_SUB( N, X, INCX, Y, INCY, TEMP )
      DDOT = TEMP

      RETURN
      END
