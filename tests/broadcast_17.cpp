#include <stdio.h>
#include <omp.h>
#include <cstdlib>
#include <cuda_runtime.h>

// Testing cudaMemcpy -> one example

int arr_len = 10;
// int arr_len = 256; // 1 Kb
// int arr_len = 2560; // 10 Kb
// int arr_len = 25600; // 100 Kb
// int arr_len = 262144; // 1 Mb
// int arr_len = 2621440; // 10 Mb
// int arr_len = 26214400; // 100 Mb
// int arr_len = 268435456; // 1 Gb

bool v_flag = false; // Print verifications True=On False=Off

void verify(void ** x_ptr){
  int num_dev = omp_get_num_devices();  
  for(int i=0; i<num_dev; ++i)
  {
    #pragma omp target device(i) map(x_ptr[i]) 
    {
      int* x = (int*)x_ptr[i];
      printf("%s No. %d ArrX = ", omp_is_initial_device()?"Host":"Device", omp_get_device_num());
      for (int j=0; j<arr_len; ++j){
        printf("%d ", *(x+j));
      }
      printf("\n");
    }
  }
}

void print_hval(int * x_arr){
  printf("Host -> Value of X: ");
  for (int j=0; j<arr_len; ++j)
    printf("%d ", x_arr[j]);
  printf("\n");
}

int main()
{
  double start, end; 
  int num_dev = omp_get_num_devices();  
  int* x_arr = (int*) malloc(arr_len * sizeof(int)); 
  
  // Set device pointers
  void ** x_ptr = (void **) malloc(sizeof(void**) * num_dev+1); 
  if (!x_ptr) {
    printf("Memory Allocation Failed\n");
    exit(1);
  }
  size_t size = sizeof(int) * arr_len;

  // Allocate device memory
  for (int dev = 0; dev < num_dev; ++dev){
    cudaSetDevice(dev);
    cudaMalloc(&x_ptr[dev], size);
  }
  // Add host pointer 
  x_ptr[num_dev]=&x_arr[0];

  printf("[Broadcast Int Array Size = %zu]\n", size);
  printf("No. of Devices: %d\n", omp_get_num_devices());

//**************************************************//
//            Host-to-One -> One-to-All             //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+20;

  // printf("Host-to-One -> One-to-All\n");
  if (v_flag) print_hval(x_arr);

  start = omp_get_wtime(); 
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dependency;
      #pragma omp task depend(out:dependency)
        cudaMemcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      for(int i=1; i<num_dev; ++i)
        #pragma omp task depend(in:dependency) firstprivate(i)
          // cudaSetDevice(dev);
          cudaMemcpy(
            x_ptr[i],                         // dst
            x_ptr[0],                         // src
            size,                             // length 
            cudaMemcpyDeviceToDevice          // kind
          );
    }
  }
  #pragma omp taskwait
  end = omp_get_wtime();

  printf( "Time %f (s)\n", end - start);
  if (v_flag) verify(x_ptr);

  free(x_ptr);

  return 0;
}

// CUDA tests
// PROG=./tests/broadcast_17; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -L/soft/compilers/cuda/cuda-11.0.2/lib64 -L/soft/compilers/cuda/cuda-11.0.2/targets/x86_64-linux/lib/ -I/soft/compilers/cuda/cuda-11.0.2/include -ldl -lcudart -pthread $PROG.cpp
// ./tests/broadcast_17.x 
// nvprof --print-gpu-trace ./tests/broadcast_17.x 