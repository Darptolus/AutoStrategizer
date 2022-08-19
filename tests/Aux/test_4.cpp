#include <stdio.h>
#include <omp.h>
#include <cstdlib>

void verify(void ** x_ptr){
  int num_dev = omp_get_num_devices();  
  for(int i; i<num_dev; ++i)
  {
    #pragma omp target device(i) map(x_ptr[0:num_dev]) 
    {
      int* x = (int*)x_ptr[i];
      printf("Value of X is %d on the: %s No. %d\n", *x, omp_is_initial_device()?"Host":"Device", omp_get_device_num());
    }
  }
}

int main(){

  int x = 10;
  int num_dev = omp_get_num_devices();  
  printf("Host -> Value of X: %d \n", x);
  printf("No. of Devices: %d\n", num_dev);

  // Set device pointers
  void ** x_ptr = (void **) malloc(sizeof(void**) * num_dev+1); 
  if (!x_ptr) {
    printf("Memory Allocation Failed\n");
    exit(1);
  }
  size_t size = sizeof(x);
  
  // Allocate device pointer
  for (int dev = 0; dev < num_dev; ++dev){
    x_ptr[dev] = omp_target_alloc(size, dev);
  }
  // Add host pointer 
  x_ptr[num_dev]=&x;
 
  //**********//
  //          //
  //**********//
  // Host-to-all
  printf("Host-to-all\n");
  #pragma omp parallel num_threads(omp_get_num_devices())
  {
    omp_target_memcpy(
      x_ptr[omp_get_thread_num()],        // dst
      x_ptr[omp_get_initial_device()],    // src
      size,                               // length 
      0,                                  // dst_offset
      0,                                  // src_offset, 
      omp_get_thread_num(),               // dst_device_num
      omp_get_initial_device()            // src_device_num
    );
  }

  verify(x_ptr);
 
  //**********//
  //          //
  //**********//
  // Host-to-one -> One-to-all
  x = 20;
  printf("Host-to-one -> One-to-all\n");
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
  verify(x_ptr);


  //**********//
  //          //
  //**********//
  // Host-to-one -> Binary tree
  x = 30;
  
  printf("Host-to-one -> Binary tree\n");
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
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
        #pragma omp task depend(in:dep_arr[(i-1)/2]) depend(out:dep_arr[i]) firstprivate(i)
          omp_target_memcpy(
            x_ptr[i],                           // dst
            x_ptr[(i-1)/2],                           // src
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
            i,                                  // dst_device_num
            (i-1)/2                                   // src_device_num
          );
    }
  }
  #pragma omp taskwait
  verify(x_ptr);

//**********//
  //          //
  //**********//
  // Host-to-Binary tree
  x = 40;
  
  printf("Host-to-Binary tree\n");
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
        omp_target_memcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          0,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      #pragma omp task depend(out:dep_arr[1])
        omp_target_memcpy(
          x_ptr[1],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          1,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      for(int i=2; i<num_dev; ++i)
        #pragma omp task depend(in:dep_arr[(i-1)/2]) depend(out:dep_arr[i]) firstprivate(i)
        omp_target_memcpy(
          x_ptr[i],                           // dst
          x_ptr[(i-1)/2],                           // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          i,                                  // dst_device_num
          (i-1)/2                                   // src_device_num
        );
    }
  }
  #pragma omp taskwait
  verify(x_ptr);

  //**********//
  //          //
  //**********//
  // Host-to-one -> Linked List
  x = 50;
  
  printf("Host-to-one -> Linked List\n");
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
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
        #pragma omp task depend(in:dep_arr[i-1]) depend(out:dep_arr[i]) firstprivate(i)
        omp_target_memcpy(
          x_ptr[i],                           // dst
          x_ptr[i-1],                           // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          i,                                  // dst_device_num
          i-1                                   // src_device_num
        );
    }
  }
  #pragma omp taskwait
  verify(x_ptr);

    //**********//
  //          //
  //**********//
  // Host-to-one -> Splited Linked List
  x = 60;
  
  printf("Host-to-one -> Splited Linked List\n");
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
        omp_target_memcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          0,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      #pragma omp task depend(out:dep_arr[num_dev/2])
        omp_target_memcpy(
          x_ptr[num_dev/2],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          num_dev/2,                          // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      for(int i=1; i<num_dev/2; ++i){
        #pragma omp task depend(in:dep_arr[i-1]) depend(out:dep_arr[i]) firstprivate(i)
        omp_target_memcpy(
          x_ptr[i],                           // dst
          x_ptr[i-1],                           // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          i,                                  // dst_device_num
          i-1                                   // src_device_num
        );
        #pragma omp task depend(in:dep_arr[num_dev/2+i-1]) depend(out:dep_arr[num_dev/2+i]) firstprivate(i)
        omp_target_memcpy(
          x_ptr[num_dev/2+i],                           // dst
          x_ptr[num_dev/2+i-1],                           // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          num_dev/2+i,                                  // dst_device_num
          num_dev/2+i-1                                   // src_device_num
        );
      }
    }
  }
  #pragma omp taskwait
  verify(x_ptr);

  free(x_ptr);

  return 0;
}

