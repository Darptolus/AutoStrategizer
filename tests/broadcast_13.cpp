#include <stdio.h>
#include <omp.h>
#include <cstdlib>
// #include <cuda.h>
#include <cuda_runtime.h>
// #include <device_launch_parameters.h>

// Testing cudaMemcpy -> Used for testing different strategies -> printed validation 

int arr_len = 10;

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

  printf("[Broadcast Int Array]\n");
  printf("No. of Devices: %d\n", omp_get_num_devices());
  
//**************************************************//
//                   Host-to-all                    //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+10;

  printf("Host-to-all\n");
  print_hval(x_arr);

  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    cudaMemcpy(
      x_ptr[omp_get_thread_num()],        // dst
      x_ptr[omp_get_initial_device()],    // src
      size,                               // length 
      cudaMemcpyHostToDevice              // kind
    );
  }
  
  verify(x_ptr);


//**************************************************//
//            Host-to-one -> One-to-all             //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+20;

  printf("Host-to-one -> One-to-all\n");
  print_hval(x_arr);
  
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
  verify(x_ptr);


//**************************************************//
//            Host-to-one -> Binary tree            //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+30;

  printf("Host-to-one -> Binary tree\n");
  print_hval(x_arr);
  
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
        cudaMemcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      for(int i=1; i<num_dev; ++i)
        #pragma omp task depend(in:dep_arr[(i-1)/2]) depend(out:dep_arr[i]) firstprivate(i)
          // cudaSetDevice(dev);
          cudaMemcpy(
            x_ptr[i],                         // dst
            x_ptr[(i-1)/2],                   // src
            size,                             // length 
            cudaMemcpyDeviceToDevice          // kind
          );
    }
  }
  #pragma omp taskwait
  verify(x_ptr);


//**************************************************//
//               Host-to-Binary tree                //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+40;
  
  printf("Host-to-Binary tree\n");
  print_hval(x_arr);
  
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
        cudaMemcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      #pragma omp task depend(out:dep_arr[1])
        // cudaSetDevice(dev);
        cudaMemcpy(
          x_ptr[1],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      for(int i=2; i<num_dev; ++i)
        #pragma omp task depend(in:dep_arr[(i-1)/2]) depend(out:dep_arr[i]) firstprivate(i)
        // cudaSetDevice(dev);
        cudaMemcpy(
          x_ptr[i],                           // dst
          x_ptr[(i-1)/2],                     // src
          size,                               // length 
          cudaMemcpyDeviceToDevice            // kind
        );
    }
  }
  #pragma omp taskwait
  verify(x_ptr);


//**************************************************//
//            Host-to-one -> Linked List            //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+50;
  
  printf("Host-to-one -> Linked List\n");
  print_hval(x_arr);
  
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
        cudaMemcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      for(int i=1; i<num_dev; ++i)
        #pragma omp task depend(in:dep_arr[i-1]) depend(out:dep_arr[i]) firstprivate(i)
        cudaMemcpy(
          x_ptr[i],                           // dst
          x_ptr[i-1],                         // src
          size,                               // length 
          cudaMemcpyDeviceToDevice            // kind
        );
    }
  }
  #pragma omp taskwait
  verify(x_ptr);


//**************************************************//
//           Host-to-Splited Linked List            //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+60;
  
  printf("Host-to-Splited Linked List\n");
  print_hval(x_arr);
  
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
        cudaMemcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      #pragma omp task depend(out:dep_arr[num_dev/2])
        cudaMemcpy(
          x_ptr[num_dev/2],                   // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          cudaMemcpyHostToDevice              // kind
        );
      for(int i=1; i<num_dev/2; ++i){
        #pragma omp task depend(in:dep_arr[i-1]) depend(out:dep_arr[i]) firstprivate(i)
        cudaMemcpy(
          x_ptr[i],                           // dst
          x_ptr[i-1],                         // src
          size,                               // length 
          cudaMemcpyDeviceToDevice            // kind
        );
        #pragma omp task depend(in:dep_arr[num_dev/2+i-1]) depend(out:dep_arr[num_dev/2+i]) firstprivate(i)
        cudaMemcpy(
          x_ptr[num_dev/2+i],                 // dst
          x_ptr[num_dev/2+i-1],               // src
          size,                               // length 
          cudaMemcpyDeviceToDevice            // kind
        );
      }
    }
  }
  #pragma omp taskwait
  verify(x_ptr);

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


//gcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart