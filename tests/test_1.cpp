// #include <stdio.h>
// #include <omp.h>

int main(){

  // printf("No. of Devices: %d\n", omp_get_num_devices());
  // // int device_id;

  // int num_dev = omp_get_num_devices();
  int dev_arr[8] = { 1, 2, 3, 4, 5, 6, 7, 8};
  int * dev_ptr = dev_arr;

  // #pragma omp parallel num_threads(omp_get_num_devices())
  // {
  //   int device_id = omp_get_thread_num();
    #pragma omp target map(dev_arr[0:8]) device(dev_ptr, num_dev)
    {
      // printf("Device %d is: %s\n", omp_get_device_num(), omp_is_initial_device()?"Host":"Device");
      // printf("Target region executed on the: %s\n", omp_is_initial_device()?"Host":"Device");   
    }  
  // }

  
// for (int dev = 0; dev < num_dev; ++dev){
//     #pragma omp target device(dev)
//     {
//       // #pragma omp target update to(num_dev) device(dev+1)
//       printf("Device %d is: %s\n", omp_get_device_num(), omp_is_initial_device()?"Host":"Device");
//     }
// }
// printf("This is the Host\n");
  return 0;
}


