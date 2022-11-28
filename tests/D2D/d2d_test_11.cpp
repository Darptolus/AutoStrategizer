#include <stdio.h>
#include <omp.h>
#include <cstdlib>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <numaif.h>

// Testing device multicopy -> Used for testing different strategies -> printed validation 
// socket spread

const bool v_flag = false; // Print verifications True=On False=Off

int init_arr_1(int * x_arr, int v_no, int arr_len){
  // Initialize Array Values
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+v_no;
  return (v_no+=10);
}

int init_arr_2(int * x_arr_0, int * x_arr_1, int v_no, int arr_len_0, int arr_len_1){
  // Initialize Array Values
  for (int i=0; i<arr_len_0; ++i)
    x_arr_0[i]=i+v_no;
  for (int i=0; i<arr_len_1; ++i)
    x_arr_1[i]=i+arr_len_0+v_no;
  return (v_no+=10);
}

void print_hval_1(int * x_arr, int arr_len){
  // Print value at host for validation
  printf("Host -> Value of X: ");
  for (int j=0; j<arr_len; ++j)
    printf("%d ", x_arr[j]);
  printf("\n");
}

void print_hval_2(int * x_arr_0, int * x_arr_1, int arr_len_0, int arr_len_1){
  // Print value at host for validation
  printf("Host -> Value of X: ");
  for (int i=0; i<arr_len_0; ++i)
    printf("%d ", x_arr_0[i]);
  for (int i=0; i<arr_len_1; ++i)
    printf("%d ", x_arr_1[i]);
  printf("\n");
}

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
  unsigned cpu, node;
  int v_no = 0; 
  int dest_no, cnk_len, dest_id;
  // int num_dev = omp_get_num_devices();  
  int num_dev = 4;
  int num_thr = 4;
  int host_id = num_dev;

  // NUMA check
  int status[1];
  int ret_code;
  status[0]=-1;
  void ** ptr_to_check;

  omp_memspace_handle_t  x_memspace = omp_default_mem_space;
  omp_alloctrait_t       x_traits[1]={omp_atk_alignment, 64};
  omp_allocator_handle_t x_alloc    = omp_init_allocator(x_memspace,1,x_traits);
  
  omp_memspace_handle_t  y_memspace = omp_default_mem_space;
  omp_alloctrait_t       y_traits[1]={omp_atk_alignment, 64};
  omp_allocator_handle_t y_alloc    = omp_init_allocator(y_memspace,1,y_traits);

  size_t size = sizeof(int) * arr_len;
  size_t size_c;
  // int* x_arr = (int*) malloc(size); 
  int *x_arr, *x_arr_0,  *x_arr_1;

  // Set device pointers
  void ** x_ptr = (void **) malloc(sizeof(void**) * num_dev+1); 

  // Allocate device memory
  x_ptr[0] = omp_target_alloc(size, 0);

  printf("Broadcast NDev: %d Int ASize = %zu \n", omp_get_num_devices(), size);
  syscall(SYS_getcpu, &cpu, &node, NULL);
  if (v_flag) printf("Main: CPU core %u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());

//**************************************************//
//                      H -> D0                     //
//**************************************************//

  if (v_flag) printf("H -> D0\n");

  // Allocate memory
  x_arr = (int*) malloc(size); 
  // Add host pointer 
  x_ptr[host_id]=&x_arr[0];
  
  dest_no = 1;

  // Initialize Array Values
  if (v_flag) v_no = init_arr_1(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval_1(x_arr, arr_len);
   
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
  free(x_arr);

//**************************************************//
//                 H -> 2D_0 -> D0                  //
//**************************************************//

  if (v_flag) printf("H -> 2D_0 -> D0\n");
  
  dest_no = 2;
  // Allocate memory
  cnk_len = arr_len/dest_no;
  size_c = sizeof(int) * cnk_len;
  
  #pragma omp parallel num_threads(num_thr) private(cpu, node, status, ret_code, ptr_to_check)
  {
    syscall(SYS_getcpu, &cpu, &node, NULL);
    switch(cpu){
      case 0:
        if (v_flag) printf("Alloc Host_0: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
        x_arr_0 = (int *)omp_alloc(size_c, x_alloc);
        ptr_to_check = (void**) &x_arr_0;
        ret_code=move_pages(0 /*self memory */, 1, ptr_to_check, NULL, status, 0);
        if (v_flag) printf("Memory at %p is at %d node (retcode %d)\n", ptr_to_check, status[0], ret_code);
      break;
      case 22:
        if (v_flag) printf("Alloc Host_1: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
        x_arr_1 = (int *)omp_alloc(size_c, y_alloc);
        ptr_to_check = (void**) &x_arr_1;
        ret_code=move_pages(0 /*self memory */, 1, ptr_to_check, NULL, status, 0);
        if (v_flag) printf("Memory at %p is at %d node (retcode %d)\n", ptr_to_check, status[0], ret_code);
      break;
      // default:
        // if (v_flag) printf("Default: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
    }
    #pragma omp barrier
  }

  // Allocate device memory
  x_ptr[2] = omp_target_alloc(size_c, 2);

  // Initialize Array Values
  if (v_flag) v_no = init_arr_2(x_arr_0, x_arr_1, v_no, cnk_len, cnk_len);
  
  if (v_flag) print_hval_2(x_arr_0, x_arr_1, cnk_len, cnk_len);

  #pragma omp parallel num_threads(num_thr) private(dest_id, cpu, node)
  {
    #pragma omp single
    {
      start = omp_get_wtime();
    }
    syscall(SYS_getcpu, &cpu, &node, NULL);
    switch(cpu){
      case 0:
        if (v_flag) printf("MemCpy Dev_0: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
          omp_target_memcpy
          (
            x_ptr[0],                                 // dst
            x_arr_0,                                  // src
            size_c,                                   // length 
            0,                                        // dst_offset
            0,                                        // src_offset, 
            0,                                        // dst_device_num
            omp_get_initial_device()                  // src_device_num
          );
      break;
      case 22:
        dest_id = 2;
        if (v_flag) printf("MemCpy Dev_1: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
          omp_target_memcpy
          (
            x_ptr[2],                     // dst
            x_arr_1,                            // src
            size_c,                             // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
            2,                            // dst_device_num
            omp_get_initial_device()            // src_device_num
          );
          omp_target_memcpy
          (
            x_ptr[0],                           // dst
            x_ptr[2],                     // src
            size_c,                             // length 
            size_c,                             // dst_offset
            0,                                  // src_offset, 
            0,                                  // dst_device_num
            2                             // src_device_num
          );
      break;
      // default:
      //   if (v_flag) printf("Default: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
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

  omp_target_free(x_ptr[2], 2);

  omp_free (x_arr_0, x_alloc);
  omp_free (x_arr_1, y_alloc);

//**************************************************//
//                  H -> 3D_1 -> D0                 //
//**************************************************//

  if (v_flag) printf("H -> 3D_1 -> D0\n");

  dest_no = 3;

  // Allocate device memory
  cnk_len = arr_len/dest_no;
  size_c = sizeof(int) * cnk_len;

  #pragma omp parallel num_threads(num_thr) private(cpu, node)
  {
    syscall(SYS_getcpu, &cpu, &node, NULL);
    switch(cpu){
      case 0:
        if (v_flag) printf("Alloc Host_0: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
        x_arr_0 = (int *)omp_alloc(2*size_c, x_alloc);
        ptr_to_check = (void**) &x_arr_0;
        ret_code=move_pages(0 /*self memory */, 1, ptr_to_check, NULL, status, 0);
        if (v_flag) printf("Memory at %p is at %d node (retcode %d)\n", ptr_to_check, status[0], ret_code);
      break;
      case 22:
        if (v_flag) printf("Alloc Host_1: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
        x_arr_1 = (int *)omp_alloc(size_c, y_alloc);
        ptr_to_check = (void**) &x_arr_1;
        ret_code=move_pages(0 /*self memory */, 1, ptr_to_check, NULL, status, 0);
        if (v_flag) printf("Memory at %p is at %d node (retcode %d)\n", ptr_to_check, status[0], ret_code);
      break;
      // default:
        // if (v_flag) printf("Default: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
    }
    #pragma omp barrier
  }

  for (int dev = 1; dev < dest_no; ++dev)
    x_ptr[dev] = omp_target_alloc(size_c, dev);


  // Initialize Array Values
  if (v_flag) v_no = init_arr_2(x_arr_0, x_arr_1, v_no, 2*cnk_len, cnk_len);
  
  if (v_flag) print_hval_2(x_arr_0, x_arr_1, 2*cnk_len, cnk_len);
  
  #pragma omp parallel num_threads(num_thr) private(dest_id, cpu, node)
  {
    #pragma omp single
    {
      start = omp_get_wtime();
    }
    syscall(SYS_getcpu, &cpu, &node, NULL);
    switch(cpu){
      case 0:
        if (v_flag) printf("MemCpy Dev_0: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
        omp_target_memcpy(
              x_ptr[0],                           // dst
              x_arr_0,                            // src
              size_c,                             // length 
              0,                                  // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
      break;
      case 1:
        dest_id = 1;
        if (v_flag) printf("MemCpy Dev_1: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
        omp_target_memcpy(
              x_ptr[1],                     // dst
              x_arr_0,                            // src
              size_c,                             // length 
              0,                                  // dst_offset
              size_c,                             // src_offset, 
              1,                            // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
            omp_target_memcpy(
              x_ptr[0],                           // dst
              x_ptr[1],                     // src
              size_c,                             // length 
              size_c,                             // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              1                             // src_device_num
            );
      break;
      case 22:
      dest_id = 2;
        if (v_flag) printf("MemCpy Dev_2: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
        omp_target_memcpy(
              x_ptr[2],                           // dst
              x_arr_1,                            // src
              size_c,                             // length 
              0,                                  // dst_offset
              0,                                  // src_offset, 
              2,                            // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
        omp_target_memcpy(
              x_ptr[0],                           // dst
              x_ptr[2],                           // src
              size_c,                             // length 
              2*size_c,                           // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              2                             // src_device_num
            );
      break;

      // default:
        // if (v_flag) printf("Default: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
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

  omp_free (x_arr_0, x_alloc);
  omp_free (x_arr_1, y_alloc);

//**************************************************//
//                  H -> 3D_2 -> D0                 //
//**************************************************//

  if (v_flag) printf("H -> 3D_2 -> D0\n");

  dest_no = 3;

  // Allocate device memory
  cnk_len = arr_len/dest_no;
  size_c = sizeof(int) * cnk_len;

  #pragma omp parallel num_threads(num_thr) private(cpu, node)
  {
    syscall(SYS_getcpu, &cpu, &node, NULL);
    switch(cpu){
      case 0:
        if (v_flag) printf("Alloc Host_0: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
        // x_arr_0 = (int *)omp_alloc(2*size_c, x_alloc);
        x_arr_0 = (int *)omp_alloc(size_c, x_alloc);
        ptr_to_check = (void**) &x_arr_0;
        ret_code=move_pages(0 /*self memory */, 1, ptr_to_check, NULL, status, 0);
        if (v_flag) printf("Memory at %p is at %d node (retcode %d)\n", ptr_to_check, status[0], ret_code);
      break;
      case 22:
        if (v_flag) printf("Alloc Host_1: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
        // x_arr_1 = (int *)omp_alloc(size_c, x_alloc);
        x_arr_1 = (int *)omp_alloc(2*size_c, y_alloc);
        ptr_to_check = (void**) &x_arr_1;
        ret_code=move_pages(0 /*self memory */, 1, ptr_to_check, NULL, status, 0);
        if (v_flag) printf("Memory at %p is at %d node (retcode %d)\n", ptr_to_check, status[0], ret_code);
      break;
      // default:
        // if (v_flag) printf("Default: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
    }
    #pragma omp barrier
  }

  // for (int dev = 1; dev < dest_no; ++dev)
  //   x_ptr[dev] = omp_target_alloc(size_c, dev);

  // x_ptr[1] = omp_target_alloc(size_c, 1);
  x_ptr[2] = omp_target_alloc(size_c, 2);
  x_ptr[3] = omp_target_alloc(size_c, 3);

  // Initialize Array Values
  if (v_flag) v_no = init_arr_2(x_arr_0, x_arr_1, v_no, cnk_len, 2*cnk_len);
  
  if (v_flag) print_hval_2(x_arr_0, x_arr_1, cnk_len, 2*cnk_len);
  
  #pragma omp parallel num_threads(num_thr) private(dest_id, cpu, node)
  {
    #pragma omp single
    {
      start = omp_get_wtime();
    }
    syscall(SYS_getcpu, &cpu, &node, NULL);
    switch(cpu){
      case 0:
        if (v_flag) printf("MemCpy Dev_0: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
        omp_target_memcpy(
              x_ptr[0],                           // dst
              x_arr_0,                            // src
              size_c,                             // length 
              0,                                  // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
      break;
      case 22:
        dest_id = 2;
        if (v_flag) printf("MemCpy Dev_1: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
        omp_target_memcpy(
              x_ptr[2],                           // dst
              x_arr_1,                            // src
              size_c,                             // length 
              0,                                  // dst_offset
              0,                             // src_offset, 
              2,                            // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
            omp_target_memcpy(
              x_ptr[0],                           // dst
              x_ptr[2],                           // src
              size_c,                             // length 
              size_c,                             // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              2                             // src_device_num
            );
      break;
      case 23:
      dest_id = 3;
        if (v_flag) printf("MemCpy Dev_2: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
        omp_target_memcpy(
              x_ptr[3],                           // dst
              x_arr_1,                            // src
              size_c,                             // length 
              0,                                  // dst_offset
              size_c,                                  // src_offset, 
              3,                            // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
        omp_target_memcpy(
              x_ptr[0],                           // dst
              x_ptr[3],                           // src
              size_c,                             // length 
              2*size_c,                           // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              3                             // src_device_num
            );
      break;
      // default:
        // if (v_flag) printf("Default: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
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

  // for (int dev = 1; dev < dest_no; ++dev)
  //   omp_target_free(x_ptr[dev], dev);
  omp_target_free(x_ptr[2], 2);
  omp_target_free(x_ptr[3], 3);

  omp_free (x_arr_0, x_alloc);
  omp_free (x_arr_1, y_alloc);

//**************************************************//
//                 H -> 4D_0 -> D0                  //
//**************************************************//

  if (v_flag) printf("H -> 4D_0 -> D0\n");

  dest_no = 4;

  // Allocate device memory
  cnk_len = arr_len/dest_no;
  size_c = sizeof(int) * cnk_len;

  #pragma omp parallel num_threads(num_thr) private(cpu, node)
  {
    syscall(SYS_getcpu, &cpu, &node, NULL);
    switch(cpu){
      case 0:
        if (v_flag) printf("Alloc Host_0: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
        x_arr_0 = (int *)omp_alloc(2*size_c, x_alloc);
        ptr_to_check = (void**) &x_arr_0;
        ret_code=move_pages(0 /*self memory */, 1, ptr_to_check, NULL, status, 0);
        if (v_flag) printf("Memory at %p is at %d node (retcode %d)\n", ptr_to_check, status[0], ret_code);
      break;
      case 22:
        if (v_flag) printf("Alloc Host_1: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
        x_arr_1 = (int *)omp_alloc(2*size_c, y_alloc);
        ptr_to_check = (void**) &x_arr_1;
        ret_code=move_pages(0 /*self memory */, 1, ptr_to_check, NULL, status, 0);
        if (v_flag) printf("Memory at %p is at %d node (retcode %d)\n", ptr_to_check, status[0], ret_code);
      break;
      // default:
        // if (v_flag) printf("Default: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
    }
    #pragma omp barrier
  }

  for (int dev = 1; dev < dest_no; ++dev)
    x_ptr[dev] = omp_target_alloc(size_c, dev);

  // Initialize Array Values
  if (v_flag) v_no = init_arr_2(x_arr_0, x_arr_1, v_no, 2*cnk_len, 2*cnk_len);
  
  if (v_flag) print_hval_2(x_arr_0, x_arr_1, 2*cnk_len, 2*cnk_len);
  
  #pragma omp parallel num_threads(num_thr)  private(dest_id, cpu, node)
  {
    #pragma omp single
    {
      start = omp_get_wtime();
    }
    syscall(SYS_getcpu, &cpu, &node, NULL);
    switch(cpu){
      case 0:
        if (v_flag) printf("MemCpy Dev_0: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
        omp_target_memcpy(
              x_ptr[0],                           // dst
              x_arr_0,                            // src
              size_c,                             // length 
              0,                                  // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
      break;
      case 1:
        dest_id = 1;
        if (v_flag) printf("MemCpy Dev_1: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
        omp_target_memcpy(
              x_ptr[1],                     // dst
              x_arr_0,                            // src
              size_c,                             // length 
              0,                                  // dst_offset
              size_c,                             // src_offset, 
              1,                            // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
            omp_target_memcpy(
              x_ptr[0],                           // dst
              x_ptr[1],                     // src
              size_c,                             // length 
              size_c,                             // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              1                             // src_device_num
            );
      break;
      case 22:
      dest_id = 2;
        if (v_flag) printf("MemCpy Dev_2: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
        omp_target_memcpy(
              x_ptr[2],                     // dst
              x_arr_1,                            // src
              size_c,                             // length 
              0,                                  // dst_offset
              0,                                  // src_offset, 
              2,                            // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
        omp_target_memcpy(
              x_ptr[0],                           // dst
              x_ptr[2],                     // src
              size_c,                             // length 
              2*size_c,                     // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              2                             // src_device_num
            );
      break;
      case 23:
      dest_id = 3;
        if (v_flag) printf("MemCpy Dev_3: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
        omp_target_memcpy(
              x_ptr[3],                     // dst
              x_arr_1,                            // src
              size_c,                             // length 
              0,                                  // dst_offset
              size_c,                             // src_offset, 
              3,                            // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
        omp_target_memcpy(
              x_ptr[0],                           // dst
              x_ptr[3],                     // src
              size_c,                             // length 
              3*size_c,                     // dst_offset
              0,                                  // src_offset, 
              0,                                  // dst_device_num
              3                             // src_device_num
            );
      break;
      default:
        if (v_flag) printf("Default: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
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

  for (int dev = 0; dev < dest_no; ++dev)
    omp_target_free(x_ptr[dev], dev);

  omp_free (x_arr_0, x_alloc);
  omp_free (x_arr_1, y_alloc);

  free(x_ptr);

  // unsetenv("OMP_PROC_BIND");

  return 0;
}

//**************************************************//
// qsub -I -n 1 -t 6:00 -q gpu_v100_smx2
// cd ~/collective_ops/tests/D2D; source ~/source/LLVM/exports.sh
// module load llvm
//**************************************************//

//**************************************************//
//                      COMPILE                     //
//**************************************************//

// PROG=d2d_test_9; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -L/soft/compilers/cuda/cuda-11.0.2/lib64 -L/soft/compilers/cuda/cuda-11.0.2/targets/x86_64-linux/lib/ -I/soft/compilers/cuda/cuda-11.0.2/include -ldl -lcudart -pthread $PROG.cpp

//**************************************************//
//                        RUN                       //
//**************************************************//

// OMP_PROC_BIND=spread taskset -c 0,1,22,23 ./d2d_test_9.x

// nsys profile -o d2d_test_9_6G_v0 ./run_one_2.sh


// nsys profile -o d2d_test_9_6G ./d2d_test_9.x
// taskset -c 0,1,20 ./d2d_test_9.x

// OMP_PROC_BIND=true OMP_DYNAMIC=true OMP_MAX_ACTIVE_LEVELS=2 OMP_DISPLAY_ENV=VERBOSE

// ./d2d_test_9.x 
// ./d2d_test_9.x>out_test.o  2>&1
// nsys profile -o test_prof --stats=true ./d2d_test_9.x
// nsys profile -o d2d_test_9_6G_0 ./d2d_test_9.x

// OMP_PROC_BIND=true nsys profile -o d2d_test_9_6G_1 ./d2d_test_9.x
// OMP_DYNAMIC=true nsys profile -o d2d_test_9_6G_2 ./d2d_test_9.x
// OMP_MAX_ACTIVE_LEVELS=2 nsys profile -o d2d_test_9_6G_3 ./d2d_test_9.x
// OMP_PROC_BIND=true OMP_DYNAMIC=true nsys profile -o d2d_test_9_6G_4 ./d2d_test_9.x
// OMP_PROC_BIND=true OMP_DYNAMIC=true OMP_MAX_ACTIVE_LEVELS=2 nsys profile -o d2d_test_9_6G_5 ./d2d_test_9.x
// OMP_PROC_BIND=spread,close OMP_DYNAMIC=true nsys profile -o d2d_test_9_6G_5 ./d2d_test_9.x
