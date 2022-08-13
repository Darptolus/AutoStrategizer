#include <stdio.h>
#include <omp.h>

int main(){
  //**********//
  //          //
  //**********//
  printf("No. of Devices: %d\n", omp_get_num_devices());
  // int device_id;

  #pragma omp parallel num_threads(omp_get_num_devices())
  {
    int device_id = omp_get_thread_num();
    #pragma omp target device(device_id)
    {
      printf("Device %d is: %s\n", omp_get_device_num(), omp_is_initial_device()?"Host":"Device");
      // printf("Target region executed on the: %s\n", omp_is_initial_device()?"Host":"Device");   
    }  
  }
  
printf("Host\n");

int num_dev = omp_get_num_devices();
for (int dev = 0; dev < num_dev; ++dev){
    #pragma omp target device(dev)
      printf("Device %d is: %s\n", omp_get_device_num(), omp_is_initial_device()?"Host":"Device");
}

  return 0;
}
