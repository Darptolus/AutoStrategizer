// #include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
// #include <cuda_device_runtime_api.h>

int main()
{
  int num_dev = 2;
  int arr_len = 10;
  int* x_arr = (int*) malloc(arr_len * sizeof(int)); 
  void ** x_ptr = (void **) malloc(sizeof(void**) * num_dev+1); 

  // allocate the memory on the GPUs
  for(int dev=0; dev<2; ++dev) {
    cudaSetDevice(dev);
    cudaMalloc(&x_ptr[dev], arr_len * sizeof(int) );
  }

  // copy the arrays to the GPUs
  for(int dev=0; dev<2; ++dev) {
    cudaSetDevice(dev);
    cudaMemcpy( x_ptr[dev], x_arr, arr_len * sizeof(int), cudaMemcpyHostToDevice);
  }
  return 0;
}


// int main()
// {
//   int arr_len = 10;
//   int* x_arr = (int*) malloc(arr_len * sizeof(int)); 

//   double *dev_a[2], *dev_b[2], *dev_c[2];
//   const int Ns[2] = {N/2, N-(N/2)};

//   // allocate the memory on the GPUs
//   for(int dev=0; dev<2; ++dev) {
//     cudaSetDevice(dev);
//     cudaMalloc( (void**)&dev_a[dev], Ns[dev] * sizeof(double) );
//     cudaMalloc( (void**)&dev_b[dev], Ns[dev] * sizeof(double) );
//     cudaMalloc( (void**)&dev_c[dev], Ns[dev] * sizeof(double) );
//   }

//   // copy the arrays 'a' and 'b' to the GPUs
//   for(int dev=0,pos=0; dev<2; pos+=Ns[dev], dev++) {
//     cudaSetDevice(dev);
//     cudaMemcpy( dev_a[dev], a+pos, Ns[dev] * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy( dev_b[dev], b+pos, Ns[dev] * sizeof(double), cudaMemcpyHostToDevice);
//   }
//   return 0;
// }