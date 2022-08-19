#include <stdio.h>
#include <omp.h>

int main(){
  printf("[Scather]\n");
  int n_dev = omp_get_num_devices();
  printf("No. of Devices: %d\n", n_dev);
  int device_id;
  int x = 10;
  int * x_arr;
  x_arr = (int*)malloc(n_dev * sizeof(int));

  #pragma omp parallel num_threads(omp_get_num_devices()) private(device_id)
  {
    device_id = omp_get_thread_num();
    #pragma omp target enter data map(to:x)_arr device(device_id)
    #pragma omp target device(device_id) 
    {
      x_arr[device_id] = omp_get_device_num();
      printf("%s[%d]: value of x: %d\n", omp_is_initial_device()?"Host":"Device", omp_get_device_num(), x_arr[device_id]);
    }  
    #pragma omp target exit data map(from:x_arr[device_id]) device(device_id)
  }

  return 0;
}
