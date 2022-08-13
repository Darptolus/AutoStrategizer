#include <stdio.h>
#include <omp.h>
#include <cstdlib>
#include <cuda_runtime.h>

// Testing cudaMemcpyPeer -> Used for testing different array sizes -> no validation 
// ToDo: Add verify print cond (On/Off)

int arr_len = 10;
// int arr_len = 256; // 1 Kb
// int arr_len = 2560; // 10 Kb
// int arr_len = 25600; // 100 Kb
// int arr_len = 262144; // 1 Mb
// int arr_len = 2621440; // 10 Mb
// int arr_len = 26214400; // 100 Mb
// int arr_len = 268435456; // 1 Gb

void verify(void ** x_ptr){
  int num_dev = omp_get_num_devices();  
  bool verify = true; // Print verifications True=On False=Off
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
//                   Host-to-All                    //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+10;

  // printf("Host-to-All\n");
  // print_hval(x_arr);
  start = omp_get_wtime(); 
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    cudaMemcpy(
      x_ptr[omp_get_thread_num()],        // dst ptr
      x_ptr[omp_get_initial_device()],    // src ptr
      size,                               // length 
      cudaMemcpyHostToDevice              // kind
    );
  }
  end = omp_get_wtime();
  // printf( "%f seconds <<<<< Host-to-All\n", end - start);
  printf( "%f\n", end - start);
  // verify(x_ptr);


//**************************************************//
//            Host-to-One -> One-to-All             //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+20;

  // printf("Host-to-One -> One-to-All\n");
  // print_hval(x_arr);
  start = omp_get_wtime(); 
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dependency;
      #pragma omp task depend(out:dependency)
        cudaMemcpy(
          x_ptr[0],                           // dst ptr
          x_ptr[omp_get_initial_device()],    // src ptr
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      for(int i=1; i<num_dev; ++i)
        #pragma omp task depend(in:dependency) firstprivate(i)
          // cudaSetDevice(dev);
          cudaMemcpyPeer(
            x_ptr[i],                         // dst ptr
            i,                                // dst_device_num
            x_ptr[0],                         // src ptr
            0,                                // src ptr_device_num
            size                              // length 
          );
    }
  }
  #pragma omp taskwait
  end = omp_get_wtime();
  // printf( "%f seconds <<<<< Host-to-One -> One-to-All\n", end - start);
  printf( "%f\n", end - start);
  // verify(x_ptr);


//**************************************************//
//            Host-to-One -> Binary tree            //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+30;

  // printf("Host-to-One -> Binary tree\n");
  // print_hval(x_arr);
  start = omp_get_wtime(); 
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
        cudaMemcpy(
          x_ptr[0],                           // dst ptr
          x_ptr[omp_get_initial_device()],    // src ptr
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      for(int i=1; i<num_dev; ++i)
        #pragma omp task depend(in:dep_arr[(i-1)/2]) depend(out:dep_arr[i]) firstprivate(i)
          // cudaSetDevice(dev);
          cudaMemcpyPeer(
            x_ptr[i],                         // dst ptr
            i,                                // dst_device_num
            x_ptr[(i-1)/2],                   // src ptr
            i,                                // dst_device_num
            size                              // length 
          );
    }
  }
  #pragma omp taskwait
  end = omp_get_wtime();
  // printf( "%f seconds <<<<< Host-to-one -> Binary tree\n", end - start);
  printf( "%f\n", end - start);
  // verify(x_ptr);


//**************************************************//
//               Host-to-Binary Tree                //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+40;
  
  // printf("Host-to-Binary Tree\n");
  // print_hval(x_arr);
  start = omp_get_wtime();  
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
        cudaMemcpy(
          x_ptr[0],                           // dst ptr
          x_ptr[omp_get_initial_device()],    // src ptr
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      #pragma omp task depend(out:dep_arr[1])
        // cudaSetDevice(dev);
        cudaMemcpy(
          x_ptr[1],                           // dst ptr
          x_ptr[omp_get_initial_device()],    // src ptr
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      for(int i=2; i<num_dev; ++i)
        #pragma omp task depend(in:dep_arr[(i-1)/2]) depend(out:dep_arr[i]) firstprivate(i)
        // cudaSetDevice(dev);
        cudaMemcpyPeer(
          x_ptr[i],                           // dst ptr
          i,                                  // dst_device_num
          x_ptr[(i-1)/2],                     // src ptr
          (i-1)/2,                            // src_device_num
          size                                // length 
        );
    }
  }
  #pragma omp taskwait
  end = omp_get_wtime();
  // printf( "%f seconds <<<<< Host-to-Binary Tree\n", end - start);
  printf( "%f\n", end - start);
  // verify(x_ptr);

//**************************************************//
//            Host-to-One -> Linked List            //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+50;
  
  // printf("Host-to-one -> Linked List\n");
  // print_hval(x_arr);
  start = omp_get_wtime();  
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
        cudaMemcpy(
          x_ptr[0],                           // dst ptr
          x_ptr[omp_get_initial_device()],    // src ptr
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      for(int i=1; i<num_dev; ++i)
        #pragma omp task depend(in:dep_arr[i-1]) depend(out:dep_arr[i]) firstprivate(i)
        cudaMemcpyPeer(
          x_ptr[i],                           // dst ptr
          i,                                  // dst_device_num
          x_ptr[i-1],                         // src ptr
          i-1,                                // src_device_num
          size                                // length 
        );
    }
  }
  #pragma omp taskwait
  end = omp_get_wtime();
  // printf("%f seconds <<<<< Host-to-one -> Linked List\n", end - start);
  printf( "%f\n", end - start);
  // verify(x_ptr);

//**************************************************//
//           Host-to-Splited Linked List            //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+60;
  
  // printf("Host-to-Splited Linked List\n");
  // print_hval(x_arr);
  start = omp_get_wtime();  
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
        cudaMemcpy(
          x_ptr[0],                           // dst ptr
          x_ptr[omp_get_initial_device()],    // src ptr
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      #pragma omp task depend(out:dep_arr[num_dev/2])
        cudaMemcpy(
          x_ptr[num_dev/2],                   // dst ptr
          x_ptr[omp_get_initial_device()],    // src ptr
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      for(int i=1; i<num_dev/2; ++i){
        #pragma omp task depend(in:dep_arr[i-1]) depend(out:dep_arr[i]) firstprivate(i)
        cudaMemcpyPeer(
          x_ptr[i],                           // dst ptr
          i,                                  // dst_device_num
          x_ptr[i-1],                         // src ptr
          i-1,                                // src_device_num
          size                                // length 
        );
        #pragma omp task depend(in:dep_arr[num_dev/2+i-1]) depend(out:dep_arr[num_dev/2+i]) firstprivate(i)
        cudaMemcpyPeer(
          x_ptr[num_dev/2+i],                 // dst ptr
          num_dev/2+i,                        // dst_device_num
          x_ptr[num_dev/2+i-1],               // src ptr
          num_dev/2+i-1,                      // src_device_num
          size                                // length 
        );
      }
    }
  }
  #pragma omp taskwait
  end = omp_get_wtime();
  // printf("%f seconds <<<<< Host-to-one -> Splited Linked List\n", end - start);
  printf( "%f\n", end - start);
  // verify(x_ptr);

  free(x_ptr);

  return 0;
}

// CUDA memory copy types

// Enumerator:
// cudaMemcpyHostToHost 	Host -> Host
// cudaMemcpyHostToDevice 	Host -> Device
// cudaMemcpyDeviceToHost 	Device -> Host
// cudaMemcpyDeviceToDevice 	Device -> Device
// cudaMemcpyDefault 	Default based unified virtual address space

// cudaMemcpyPeer ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count )
// Copies memory between two devices.
// Parameters
// dst
// - Destination device pointer
// dstDevice
// - Destination device
// src
// - Source device pointer
// srcDevice
// - Source device
// count
// - Size of memory copy in bytes


//gcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

// PROG=./tests/broadcast_15; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -L/soft/compilers/cuda/cuda-11.0.2/lib64 -L/soft/compilers/cuda/cuda-11.0.2/targets/x86_64-linux/lib/ -I/soft/compilers/cuda/cuda-11.0.2/include -ldl -lcudart -pthread $PROG.cpp
// ./tests/broadcast_15.x 
// nvprof --print-gpu-trace ./tests/broadcast_15.x 