/*
 *  bw: single core memory bandwidth (QoS user-interactive): read (sum),
 *  copy (memcpy) and write (memset) of N MiB, median of 7 repetitions in
 *  GB/s (copy: bytes read + written).
 *
 *      bw [MiB ...]
 */
#include <pthread/qos.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return (x>y)-(x<y);}
int main(int argc,char**argv){
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
  for(int a=1;a<argc;++a){
    size_t n=(size_t)atol(argv[a])<<20, nd=n/8;
    double *x=malloc(n),*y=malloc(n),s=0,g[3][7];
    memset(x,1,n);memset(y,1,n);
    for(int r=0;r<7;++r){
      double t0=now(); double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0; for(size_t i=0;i<nd;i+=8){s0+=x[i];s1+=x[i+1];s2+=x[i+2];s3+=x[i+3];s4+=x[i+4];s5+=x[i+5];s6+=x[i+6];s7+=x[i+7];} s+=s0+s1+s2+s3+s4+s5+s6+s7; double t1=now();
      memcpy(y,x,n); __asm__ volatile("" : : "r"(y) : "memory"); double t2=now(); s+=y[r]; memset(x,r,n); double t3=now();
      g[0][r]=n/(t1-t0)*1e-9; g[1][r]=2*n/(t2-t1)*1e-9; g[2][r]=n/(t3-t2)*1e-9;
    }
    for(int k=0;k<3;++k) qsort(g[k],7,sizeof(double),cmp);
    printf("%6ld MiB  read %6.1f  copy %6.1f  write %6.1f GB/s\n",atol(argv[a]),g[0][3],g[1][3],g[2][3]);
    if(s==1.5)puts("");free(x);free(y);
  }
  return 0;}
