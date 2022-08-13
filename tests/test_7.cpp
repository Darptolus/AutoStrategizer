#include <stdio.h>
#include <omp.h>
#include <cstdlib>
#include <cuda_runtime.h>

// Testing omp_target_memcpy vs cudaMemcpy -> different array sizes

int arr_len = 10;
// int arr_len = 256; // 1 Kb
// int arr_len = 2560; // 10 Kb
// int arr_len = 25600; // 100 Kb
// int arr_len = 262144; // 1 Mb
// int arr_len = 2621440; // 10 Mb
// int arr_len = 26214400; // 100 Mb
// int arr_len = 268435456; // 1 Gb

bool v_flag = true; // Print verifications True=On False=Off

void verify(void ** x_ptr){
  int num_dev = omp_get_num_devices();  
  int num_dev_0 = 2;  

  for(int i=0; i<num_dev_0; ++i)
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
  int num_dev_0 = 2;  
 
  int* x_arr = (int*) malloc(arr_len * sizeof(int)); 

  // Devices
  int* n_dev = (int*) malloc(num_dev * sizeof(int));

  // Set device pointers
  void ** x_ptr = (void **) malloc(sizeof(void**) * num_dev_0+1); 
  if (!x_ptr) {
    printf("Memory Allocation Failed\n");
    exit(1);
  }
  size_t size = sizeof(int) * arr_len;

  // Add host pointer 
  x_ptr[num_dev_0]=&x_arr[0];

  printf("[Broadcast Int Array Size = %zu]\n", size);
  printf("No. of Devices: %d\n", num_dev_0);

//******************************************************//
//     Host-to-D0 / Host-to-D1 (omp_target_memcpy)      //
//******************************************************//
  // Set up devices
  n_dev[0]=0;
  n_dev[1]=0;

  // Allocate device memory
  for (int dev = 0; dev < num_dev_0; ++dev){
    x_ptr[dev] = omp_target_alloc(size, 0);
  }

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+10;

  printf("Host-to-D0 / Host-to-D1\n");
    if (v_flag) print_hval(x_arr);

  start = omp_get_wtime(); 
  #pragma omp parallel num_threads(num_dev_0) shared(x_ptr)
  {
    #pragma omp task
      omp_target_memcpy(
        x_ptr[0],                           // dst ptr (D0)
        x_ptr[num_dev_0],                   // src ptr (Host)
        size,                               // length 
        0,                                  // dst_offset
        0,                                  // src_offset, 
        0,                                  // dst_device_num
        omp_get_initial_device()            // src_device_num
      );
    #pragma omp task
      omp_target_memcpy(
        x_ptr[1],                           // dst ptr (D0)
        x_ptr[num_dev_0],                   // src ptr (Host)
        size,                               // length 
        0,                                  // dst_offset
        0,                                  // src_offset, 
        0,                                  // dst_device_num
        omp_get_initial_device()            // src_device_num
      );
  }
  #pragma omp taskwait
  end = omp_get_wtime();

  printf( "Time %f (s)\n", end - start);
  if (v_flag) verify(x_ptr);

//******************************************************//
//      Host-to-D0 / D0-to-D1 (omp_target_memcpy)       //
//******************************************************//
  
  // Allocate device memory
  for (int dev = 0; dev < num_dev; ++dev){
    x_ptr[dev] = omp_target_alloc(size, dev);
  }

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+10;

  // print_hval(x_arr);

  printf("Host-to-one -> One-to-all\n");
  start = omp_get_wtime(); 
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dependency;
      #pragma omp task depend(out:dependency)
        omp_target_memcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          0,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      for(int i=1; i<num_dev; ++i)
        #pragma omp task depend(in:dependency) firstprivate(i)
          omp_target_memcpy(
            x_ptr[i],                           // dst
            x_ptr[0],                           // src
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
            i,                                  // dst_device_num
            0                                   // src_device_num
          );
    }
  }
  #pragma omp taskwait
  end = omp_get_wtime();
  printf( ">>>>> %f seconds <<<<< Host-to-one -> One-to-all\n", end - start);
  // verify(x_ptr);


//******************************************************//
// Host-to-D0 / Host-to-D1 (cudaMemcpy/cudaMemcpyPeer)  //
//******************************************************//

  // Allocate device memory
  for (int dev = 0; dev < num_dev; ++dev){
    cudaSetDevice(dev);
    cudaMalloc(&x_ptr[dev], size);
  }

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