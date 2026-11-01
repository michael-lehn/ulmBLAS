/* shapes.c: best-of timing of cblas_dgemm for (m,n,k), column major, alpha=beta=1 */
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(void){
  int S[][3]={{8,8,8},{16,16,16},{32,32,32},{64,64,64},{256,256,256},{2048,64,2048},{64,2048,2048},{2048,2048,64}};
  for(unsigned s=0;s<sizeof S/sizeof *S;++s){
    int m=S[s][0],n=S[s][1],k=S[s][2];
    double *A=malloc(sizeof(double)*m*k),*B=malloc(sizeof(double)*k*n),*C=malloc(sizeof(double)*m*n);
    for(long i=0;i<(long)m*k;++i)A[i]=(double)rand()/RAND_MAX-.5;
    for(long i=0;i<(long)k*n;++i)B[i]=(double)rand()/RAND_MAX-.5;
    for(long i=0;i<(long)m*n;++i)C[i]=0;
    double flops=2.0*m*n*k,best=1e30;
    long reps=1; for(;;){double t=now();for(long r=0;r<reps;++r)cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,A,m,B,k,1.0,C,m);t=now()-t;if(t>0.05||reps>(1L<<24))break;reps*=2;}
    for(int a=0;a<7;++a){double t=now();for(long r=0;r<reps;++r)cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,A,m,B,k,1.0,C,m);t=(now()-t)/reps;if(t<best)best=t;}
    printf("%d %d %d %.3f %.3f\n",m,n,k,flops/best*1e-9,best*1e6);
    free(A);free(B);free(C);
  }
  return 0;
}
