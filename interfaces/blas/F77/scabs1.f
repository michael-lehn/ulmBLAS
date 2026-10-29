      REAL FUNCTION SCABS1( Z )

      COMPLEX            Z

      REAL               TEMP
      EXTERNAL           SCABS1_SUB

      CALL SCABS1_SUB( Z, TEMP )
      SCABS1 = TEMP

      RETURN
      END
