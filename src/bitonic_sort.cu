#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

struct timeval st, et;
__device__
void swapPar(int*, int, int);
void swapSer(int*, int, int);
void rng(int*, int);
int getMax(int*, int);
void buildDummy(int*, int, int, int);
__global__
void compareAndSwap(int*, int, int, int);
void impBitonicSortPar(int*, int, int);
void impBitonicSortSer(int*, int);
int getPowTwo(int);
void writeToFile(int*, int, char*);

int main(int argc, char **argv) {
  int n, dummy_n, t = 512;

  if (argc < 2) {
    printf("Usage: %s <n> <p>\nwhere <n> is problem size, <p> is number of thread (optional)\n", argv[0]);
    exit(1);
  }

  if (argc == 3){
    t = atoi(argv[2]);
  }

  n = atoi(argv[1]);
  dummy_n = getPowTwo(n);

  int *arr, *arr_ser, *d_arr;

  arr = (int*) malloc(dummy_n*sizeof(int));
  arr_ser = (int*) malloc(dummy_n*sizeof(int));
  rng(arr,n);
  int max_x = getMax(arr,n);
  buildDummy(arr,n,dummy_n,max_x);
  memcpy(arr_ser, arr, dummy_n*sizeof(int));
  cudaMalloc((void **)&d_arr, dummy_n*sizeof(int));
  cudaMemcpy(d_arr, arr, dummy_n*sizeof(int), cudaMemcpyHostToDevice);

  // write random numbers to input file
  writeToFile(arr,n,"./data/input");

  // execute paralel
  gettimeofday(&st,NULL);
  impBitonicSortPar(d_arr,dummy_n,t);
  gettimeofday(&et,NULL);
  int elapsed_paralel = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
  printf("Execution paralel time: %d micro sec\n",elapsed_paralel);

  // execute serial
  gettimeofday(&st,NULL);
  impBitonicSortSer(arr_ser,dummy_n);
  gettimeofday(&et,NULL);
  int elapsed_serial = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
  printf("Execution serial time: %d micro sec\n",elapsed_serial);

  // calculate speedup
  printf("Speedup : %.3f\n",(float)elapsed_serial/elapsed_paralel);

  cudaMemcpy(arr, d_arr, dummy_n*sizeof(int), cudaMemcpyDeviceToHost);
  writeToFile(arr,n,"./data/output");
  free(arr);
  free(arr_ser);
  cudaFree(d_arr);
  return 0;
}

void writeToFile(int* arr, int n, char* path){
  FILE* f = fopen(path,"w");
  for(int i=0; i<n; i++) {
      fprintf(f, "%d\n", arr[i]);
  }
  fclose(f);
}

void rng(int* arr, int n) {
  int seed = 13515097;
  srand(seed);
  for(long i = 0; i < n; i++) {
      arr[i] = (int)rand();
  }
}

int getMax(int* arr, int n){
  int max_x = arr[0];
  for(int i=0; i<n; i++){
    max_x = ((max_x<arr[i])?arr[i]:max_x);
  }
  return max_x;
}

void buildDummy(int* arr,int N,int dummy_n, int max_x){
  for(long i = N; i < dummy_n; i++) {
    arr[i]=max_x;
  }
}

__device__
void swapPar(int* a, int i, int j) {
  int t;
  t = a[i];
  a[i] = a[j];
  a[j] = t;
}

void swapSer(int* a, int i, int j) {
  int t;
  t = a[i];
  a[i] = a[j];
  a[j] = t;
}

__global__
void compareAndSwap(int* a, int n, int k, int j){
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  while(i<n){
    int ij=i^j;
    if ((ij)>i) {
      // monotonic increasing
      if ((i&k)==0 && a[i] > a[ij]) swapPar(a,i,ij);
      // monotonic decreasing
      if ((i&k)!=0 && a[i] < a[ij]) swapPar(a,i,ij);
    }
    i += blockDim.x;
  }
}

/*
Imperative paralel bitonic sort
*/
void impBitonicSortPar(int* a, int n, int t) {
  int j,k;

  for (k=2; k<=n; k=2*k) {
    for (j=k>>1; j>0; j=j>>1) {
      compareAndSwap<<<n/t,t>>>(a, n, k, j);
      cudaDeviceSynchronize();
    }
  }
}

void impBitonicSortSer(int* a, int n){
  int i,j,k;

  for (k=2; k<=n; k=2*k) {
    for (j=k>>1; j>0; j=j>>1) {
      for (i=0; i<n; i++) {
        int ij=i^j;
        if ((ij)>i) {
          // monotonic increasing
          if ((i&k)==0 && a[i] > a[ij]) swapSer(a,i,ij);
          // monotonic decreasing
          if ((i&k)!=0 && a[i] < a[ij]) swapSer(a,i,ij);
        }
      }
    }
  }
}

int getPowTwo(int n){
  int d=1;
  while (d>0 && d<n) d<<=1;
  return d;
}
