#include <stdio.h>
#include <omp.h>

int main(){
  //**************************************************//
  //                    Host-2-All                    //
  //**************************************************//
  
  printf("[Broadcast]\n");
  printf("No. of Devices: %d\n", omp_get_num_devices());

  int x=10;
  
  #pragma omp parallel num_threads(omp_get_num_devices())
    #pragma omp target device(omp_get_thread_num()) map(to:x)
    {
      printf("Value of X is %d on the: %s\n", x, omp_is_initial_device()?"Host":"Device");
      // printf("Target region executed on the: %s\n", omp_is_initial_device()?"Host":"Device");
      
    }  
  return 0;
}
