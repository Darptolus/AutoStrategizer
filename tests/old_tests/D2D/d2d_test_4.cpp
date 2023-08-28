#include <stdio.h>
#include <omp.h>
#include <cstdlib>
// #include <vector>

// Testing device multicopy -> Used for testing different strategies -> printed validation 
// socket spread

const bool v_flag = false; // Print verifications True=On False=Off

void verify(void ** x_ptr, int arr_len, int n_dev){
  // for(int i=1; i<=n_dev; ++i)
  // {
    #pragma omp target device(0) map(x_ptr[0], arr_len) 
    {
      int* x = (int*)x_ptr[0];
      printf("%s No. %d ArrX = ", omp_is_initial_device()?"Host":"Device", omp_get_device_num());
      for (int j=0; j<arr_len; ++j){
        printf("%d ", *(x+j));
      }
      printf("\n");
    }
  // }
}

void validate(void ** x_ptr, int arr_len, int host_id){
  // for(int i=1; i<=n_dev; ++i)
  // {
    int* y = (int*)x_ptr[host_id];
    #pragma omp target device(0) map(x_ptr[0], arr_len) 
    {
      int* x = (int*)x_ptr[0];
      int ok = 0;
      for (int j=0; j<arr_len; ++j){
        if (*(x+j) != *(y+j))
          {
            printf("FAIL %d \n", *(x+j));
            int ok = 1;
            break;
          }
      }
    if (ok == 0)
      printf("OK \n");
    }
  // }
}

void print_hval(int * x_arr, int arr_len){
  // Print value at host for validation
  printf("Host -> Value of X: ");
  for (int j=0; j<arr_len; ++j)
    printf("%d ", x_arr[j]);
  printf("\n");
}

int init_arr(int * x_arr, int v_no, int arr_len){
  // Initialize Array Values
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+v_no;
  return (v_no+=10);
}

int main()
{
  // int arr_len = 60;
  // int arr_len = 15360; // 60 Kb
  // int arr_len = 153600; // 600 Kb
  // int arr_len = 1572864; // 6 Mb
  // int arr_len = 15728640; // 60 Mb
  // int arr_len = 157286400; // 600 Mb
  int arr_len = 1610612736; // 6 Gb
  
  // setenv("OMP_PROC_BIND", "true", 1);
  // omp_set_dynamic(true);
  // omp_set_max_active_levels(2);

  if(const char* arr_size = std::getenv("ARR_SZ"))
    arr_len = atoi(arr_size);

  double start, end;
  int v_no = 0;
  int dest_no;
  // int num_dev = omp_get_num_devices();  
  int num_dev = 6;
  int host_id = num_dev;

  size_t size = sizeof(int) * arr_len;
  int* x_arr = (int*) malloc(size); 
  
  // Set device pointers
  void ** x_ptr = (void **) malloc(sizeof(void**) * num_dev+1); 

  // Add host pointer 
  x_ptr[host_id]=&x_arr[0];

  
  // Allocate device memory
  x_ptr[0] = omp_target_alloc(size, 0);

  printf("Broadcast NDev: %d Int ASize = %zu \n", omp_get_num_devices(), size);

//**************************************************//
//                      H -> D0                     //
//**************************************************//

  if (v_flag) printf("H -> D0\n");

  dest_no = 1;

  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);
   
  start = omp_get_wtime(); 

  // printf("To Main %d\n", main_arr[i]);
  omp_target_memcpy(
    x_ptr[0],                           // dst
    x_ptr[host_id],                     // src
    size,                               // length 
    0,                                  // dst_offset
    0,                                  // src_offset, 
    0,                                  // dst_device_num
    omp_get_initial_device()            // src_device_num
  );
  end = omp_get_wtime();
  
  if (v_flag) verify(x_ptr, arr_len, dest_no);

  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  // for (int dev = 1; dev < dest_no; ++dev)
  //   omp_target_free(x_ptr[dev], 0);

//**************************************************//
//                 H -> D0-1 -> D0                  //
//**************************************************//

  if (v_flag) printf("H -> D0-1 -> D0\n");

  // Allocate device memory
  dest_no = 2;
  x_ptr[1] = omp_target_alloc(size/dest_no, 4);

  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);
  
  #pragma omp parallel num_threads(2) proc_bind(spread) shared(x_ptr)
  {
    #pragma omp single
    {
      start = omp_get_wtime();
    }
    if (omp_get_thread_num() == 0){
      // printf("To Main %d\n", main_arr[i]);
      omp_target_memcpy(
        x_ptr[0],                           // dst
        x_ptr[host_id],                     // src
        size/dest_no,                       // length 
        0,                                  // dst_offset
        0*size/dest_no,                     // src_offset, 
        0,                                  // dst_device_num
        omp_get_initial_device()            // src_device_num
      );
    }
      
    if (omp_get_thread_num() == 1){
      #pragma omp parallel proc_bind(close)
      {
        int dep_0;
        #pragma omp single
        {
          #pragma omp task depend(out:dep_0) shared(x_ptr)
          {
            // printf("To Main %d\n", main_arr[i]);
            omp_target_memcpy(
              x_ptr[1],                           // dst
              x_ptr[host_id],                     // src
              size/dest_no,                       // length 
              0,                                  // dst_offset
              1*size/dest_no,                     // src_offset, 
              4,                                  // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
          }
          #pragma omp task depend(in:dep_0) shared(x_ptr)
          {
            // printf("To Main %d\n", main_arr[i]);
            omp_target_memcpy(
              x_ptr[0],                           // dst
              x_ptr[1],                           // src
              size/dest_no,                       // length 
              1*size/dest_no,                     // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              4                                   // src_device_num
            );
          }
        }
      }
    }
    #pragma omp barrier
    #pragma omp single
    {
      end = omp_get_wtime();
    }
    #pragma omp barrier
  }
  
  if (v_flag) verify(x_ptr, arr_len, dest_no);
  // if (v_flag) validate(x_ptr, arr_len, host_id);
  
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  for (int dev = 1; dev < dest_no; ++dev)
    omp_target_free(x_ptr[dev], dev);

//**************************************************//
//                 H -> D0-2 -> D0                  //
//**************************************************//

  if (v_flag) printf("H -> D0-2 -> D0\n");

  // Allocate device memory
  dest_no = 3;
  x_ptr[1] = omp_target_alloc(size/dest_no, 3);
  x_ptr[2] = omp_target_alloc(size/dest_no, 4);

  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);
  
  #pragma omp parallel num_threads(2) proc_bind(spread) shared(x_ptr)
  {
    #pragma omp single
    {
      start = omp_get_wtime();
    }
    if (omp_get_thread_num() == 0){
      // printf("To Main %d\n", main_arr[i]);
      #pragma omp parallel proc_bind(close)
      {
        int dep_0;
        #pragma omp single
        {
          #pragma omp task
          {
            omp_target_memcpy(
              x_ptr[0],                         // dst
              x_ptr[host_id],                     // src
              size/dest_no,                       // length 
              0,                                  // dst_offset
              0*size/dest_no,                   // src_offset, 
              0,                                // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
          }
          #pragma omp task depend(out:dep_0)
          {
            // printf("To Main %d\n", main_arr[i]);
            omp_target_memcpy(
              x_ptr[1],                         // dst
              x_ptr[host_id],                     // src
              size/dest_no,                       // length 
              0,                                  // dst_offset
              1*size/dest_no,                   // src_offset, 
              4,                                // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
          }
          #pragma omp task depend(in:dep_0)
          {
            // printf("To Main %d\n", main_arr[i]);
            omp_target_memcpy(
              x_ptr[0],                           // dst
              x_ptr[1],                           // src
              size/dest_no,                       // length 
              1*size/dest_no,                     // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              4                                   // src_device_num
            );
          }
        }
      }
    }
    if (omp_get_thread_num() == 1){
      #pragma omp parallel proc_bind(close)
      {
      int dep_1;
      #pragma omp single
        {
          #pragma omp task depend(out:dep_1) 
          {
            // printf("To Main %d\n", main_arr[i]);
            omp_target_memcpy(
              x_ptr[2],                         // dst
              x_ptr[host_id],                     // src
              size/dest_no,                       // length 
              0,                                  // dst_offset
              2*size/dest_no,                   // src_offset, 
              3,                                // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
          }
          #pragma omp task depend(in:dep_1)
          {
            // printf("To Main %d\n", main_arr[i]);
            omp_target_memcpy(
              x_ptr[0],                           // dst
              x_ptr[2],                           // src
              size/dest_no,                       // length 
              2*size/dest_no,                     // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              3                                   // src_device_num
            );
          }
        }
      }
    }
    #pragma omp barrier
    #pragma omp single
    {
      end = omp_get_wtime();
    }
    #pragma omp barrier
  }
  
  if (v_flag) verify(x_ptr, arr_len, dest_no);

  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  for (int dev = 1; dev < dest_no; ++dev)
    omp_target_free(x_ptr[dev], dev);


  free(x_ptr);

  unsetenv("OMP_PROC_BIND");

  return 0;
}

// PROG=d2d_test_4; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -L/soft/compilers/cuda/cuda-11.0.2/lib64 -L/soft/compilers/cuda/cuda-11.0.2/targets/x86_64-linux/lib/ -I/soft/compilers/cuda/cuda-11.0.2/include -ldl -lcudart -pthread $PROG.cpp
// OMP_PROC_BIND=true OMP_DYNAMIC=true OMP_MAX_ACTIVE_LEVELS=2 OMP_DISPLAY_ENV=VERBOSE
// ./d2d_test_4.x 
// ./d2d_test_4.x>out_test.o  2>&1
// nsys profile -o test_prof --stats=true ./d2d_test_4.x
// nsys profile -o d2d_test_4_6G_0 ./d2d_test_4.x

// OMP_PROC_BIND=true nsys profile -o d2d_test_4_6G_1 ./d2d_test_4.x
// OMP_DYNAMIC=true nsys profile -o d2d_test_4_6G_2 ./d2d_test_4.x
// OMP_MAX_ACTIVE_LEVELS=2 nsys profile -o d2d_test_4_6G_3 ./d2d_test_4.x
// OMP_PROC_BIND=true OMP_DYNAMIC=true nsys profile -o d2d_test_4_6G_4 ./d2d_test_4.x
// OMP_PROC_BIND=true OMP_DYNAMIC=true OMP_MAX_ACTIVE_LEVELS=2 nsys profile -o d2d_test_4_6G_5 ./d2d_test_4.x
// OMP_PROC_BIND=spread,close OMP_DYNAMIC=true nsys profile -o d2d_test_4_6G_5 ./d2d_test_4.x

// nsys profile -o d2d_test_4_6G ./d2d_test_4.x
// taskset -c 0,32,33 ./test_16.x

