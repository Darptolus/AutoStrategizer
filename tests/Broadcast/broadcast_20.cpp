#include <stdio.h>
#include <omp.h>
#include <cstdlib>

// Testing omp_target_memcpy -> Used for testing different strategies -> printed validation 

int arr_len = 10;
// int arr_len = 256; // 1 Kb
// int arr_len = 2560; // 10 Kb
// int arr_len = 25600; // 100 Kb
// int arr_len = 262144; // 1 Mb
// int arr_len = 2621440; // 10 Mb
// int arr_len = 26214400; // 100 Mb
// int arr_len = 268435456; // 1 Gb

const bool v_flag = true; // Print verifications True=On False=Off

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
  // Print value at host for validation
  printf("Host -> Value of X: ");
  for (int j=0; j<arr_len; ++j)
    printf("%d ", x_arr[j]);
  printf("\n");
}

void print_hval(int * x_arr, int v_no){
  // Initialize Array Values
  v_no+=10;

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+v_no;
}

int main()
{
  double start, end;
  int v_no = 0;
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
    x_ptr[dev] = omp_target_alloc(size, dev);
  }
  // Add host pointer 
  x_ptr[num_dev]=&x_arr[0];

  printf("[Broadcast Int Array]\n");
  printf("No. of Devices: %d\n", omp_get_num_devices());


//**************************************************//
//            Host-to-all (Sequential)              //
//**************************************************//

  if (v_flag) printf("Host-to-all (Sequential)\n");

  // Initialize Array Values
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+v_no;
  
  if (v_flag) print_hval(x_arr);
  
  start = omp_get_wtime(); 
  for (int dev = 0; dev < num_dev; ++dev)
  {
    omp_target_memcpy(
      x_ptr[dev],                         // dst
      x_ptr[omp_get_initial_device()],    // src
      size,                               // length 
      0,                                  // dst_offset
      0,                                  // src_offset, 
      dev,                                // dst_device_num
      omp_get_initial_device()            // src_device_num
    );
  }

  end = omp_get_wtime();
  printf( "Time %f (s)\n", end - start);
  if (v_flag) verify(x_ptr);


//**************************************************//
//                   Host-to-all                    //
//**************************************************//

  if (v_flag) printf("Host-to-all\n");

  // Initialize Array Values
  v_no+=10;
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+v_no;
  
  if (v_flag) print_hval(x_arr);

  start = omp_get_wtime(); 
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
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
  
  end = omp_get_wtime();
  printf( "Time %f (s)\n", end - start);
  if (v_flag) verify(x_ptr);


//**************************************************//
//          Host-to-Main -> Main-to-local           //
//**************************************************//

  // Initialize Array Values
  v_no+=10;
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+v_no;

  // Set up Main - Secondary 
  main_no = 2;
  int* main_arr = (int*) malloc(main_no * sizeof(int)); 
  main_arr= {0, 4};
  sec_no = 4;
  int* main_dep_arr = (int*) malloc(main_no * sec_no * sizeof(int)); 

  // Set up dependencies
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+v_no;


  if (v_flag) print_hval(x_arr);

  if (v_flag) printf("Host-to-Main -> Main-to-local\n");

  start = omp_get_wtime(); 
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_0, dep_1;
      #pragma omp task depend(out:dep_0)
        omp_target_memcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          0,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      #pragma omp task depend(out:dep_1)
        omp_target_memcpy(
          x_ptr[4],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          0,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      for(int i=1; i<num_dev/2; ++i){
        #pragma omp task depend(in:dep_0) firstprivate(i)
          omp_target_memcpy(
            x_ptr[i],                           // dst
            x_ptr[0],                           // src
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
            i,                                  // dst_device_num
            0                                   // src_device_num
          );
        #pragma omp task depend(in:dep_1) firstprivate(i)
          omp_target_memcpy(
            x_ptr[i+num_dev/2],                 // dst
            x_ptr[4],                           // src
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
            i,                                  // dst_device_num
            4                                   // src_device_num
          );
      }
    }
  }
  #pragma omp taskwait

  end = omp_get_wtime();
  printf( "Time %f (s)\n", end - start);
  if (v_flag) verify(x_ptr);

 free (main_arr);
//**************************************************//
//          Host-to-Main -> Main-to-local(Half & Half)           //
//**************************************************//
/*
  // main_no = 2;
  // int* main_arr = (int*) malloc(main_no * sizeof(int)); 
  // main_arr= {0, 4};

  v_no+=10;
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+v_no;
  
  if (v_flag) print_hval(x_arr);

  if (v_flag) printf("Host-to-Main -> Main-to-local\n");

  start = omp_get_wtime(); 
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dep_0, dep_1;
      #pragma omp task depend(out:dep_0)
        omp_target_memcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          0,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      #pragma omp task depend(out:dep_1)
        omp_target_memcpy(
          x_ptr[4],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          0,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      for(int i=1; i<num_dev/2; ++i){
        #pragma omp task depend(in:dep_0) firstprivate(i)
          omp_target_memcpy(
            x_ptr[i],                           // dst
            x_ptr[0],                           // src
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
            i,                                  // dst_device_num
            0                                   // src_device_num
          );
        #pragma omp task depend(in:dep_1) firstprivate(i)
          omp_target_memcpy(
            x_ptr[i+num_dev/2],                 // dst
            x_ptr[4],                           // src
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
            i,                                  // dst_device_num
            4                                   // src_device_num
          );
      }
    }
  }
  #pragma omp taskwait

  end = omp_get_wtime();
  printf( "Time %f (s)\n", end - start);
  if (v_flag) verify(x_ptr);
*/
//**************************************************//
//            Host-to-one -> Linked List            //
//**************************************************//
/*
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+50;
  
  print_hval(x_arr);

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

*/
//**************************************************//
//        Host-to-one -> Splited Linked List        //
//**************************************************//
/*
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+60;
  
  print_hval(x_arr);

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

*/
  return 0;
}

