#include <stdio.h>
#include <omp.h>
#include <cstdlib>

int main(){
  //**********//
  //          //
  //**********//
  int x = 10;
  int num_dev = omp_get_num_devices();
  
  printf("Host -> Value of X: %d \n", x);
  printf("No. of Devices: %d\n", num_dev);

  // Host-to-one -> one-to-many
  // Set device pointers
  // int** x_ptr = (int**) malloc(num_dev * sizeof(int)); 
  void ** x_ptr = (void **) malloc(sizeof(void**) * num_dev); 
  void * x_ptr_aux;
  if (!x_ptr) {
    printf("Memory Allocation Failed");
    exit(1);
  }
  size_t size = sizeof(x);
  
  // Allocate device pointer
  for (int dev = 0; dev < num_dev; ++dev){
    x_ptr_aux = omp_target_alloc(size, dev);
    x_ptr[dev]=x_ptr_aux;
  }
    // #pragma omp parallel num_threads(omp_get_num_devices()){
      // #pragma omp task if(omp_get_thread_num()==0)
      // {
        #pragma omp target device(0) map(to:x) depend(out:x)
        {
          printf("Value of X is %d on the: %s No. %d\n", x, omp_is_initial_device()?"Host":"Device", omp_get_device_num());
          for (int dev = 1; dev < num_dev; ++dev){
            // Coppy value to devices
            omp_target_memcpy(
              x_ptr[dev], // dst
              &x,         // src
              size,       // length 
              0,          // dst_offset
              0,          // src_offset, 
              dev,        // dst_device_num
              0           // src_device_num
            );
          }
        }
    // } //end task
  // } // end parallel

  free(x_ptr);


  return 0;
}


  // }
  
// 

// int num_dev = omp_get_num_devices();
// for (int dev = 0; dev < num_dev; ++dev){
//     #pragma omp target device(dev)
//       printf("Device %d is: %s\n", omp_get_device_num(), omp_is_initial_device()?"Host":"Device");
// }