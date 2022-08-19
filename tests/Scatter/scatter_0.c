#include <stdio.h>
#include <omp.h>

int main(){
  printf("[Gather]\n");
  printf("No. of Devices: %d\n", omp_get_num_devices());
  int device_id;
  int x = 10;

  #pragma omp parallel num_threads(omp_get_num_devices()) private(device_id)
  {
    device_id = omp_get_thread_num();
    #pragma omp target enter data map(to:x) device(device_id)
    #pragma omp target device(device_id) 
    {
      printf("%s[%d]: value of x: %d\n", omp_is_initial_device()?"Host":"Device", omp_get_device_num(), x);
      // printf("Device %d is: %s\n", omp_get_device_num(), omp_is_initial_device()?"Host":"Device");
      // printf("Target region executed on the: %s\n", omp_is_initial_device()?"Host":"Device");   
    }  
  }


  return 0;
}
