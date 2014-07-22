#ifndef DMM_H
   #define DMM_H

   #define ATL_mmMULADD
   #define ATL_mmLAT 5
   #define ATL_mmMU  4
   #define ATL_mmNU  2
   #define ATL_mmKU  2
   #define MB 56
   #define NB 56
   #define KB 56
   #define NBNB 3136
   #define MBNB 3136
   #define MBKB 3136
   #define NBKB 3136
   #define NB2 112
   #define NBNB2 6272

   #define ATL_MulByNB(N_) ((N_) * 56)
   #define ATL_DivByNB(N_) ((N_) / 56)
   #define ATL_MulByNBNB(N_) ((N_) * 3136)
   #define NBmm ATL_dJIK56x56x56TN56x56x0_a1_b1
   #define NBmm_b1 ATL_dJIK56x56x56TN56x56x0_a1_b1
   #define NBmm_b0 ATL_dJIK56x56x56TN56x56x0_a1_b0
   #define NBmm_bX ATL_dJIK56x56x56TN56x56x0_a1_bX

#endif
