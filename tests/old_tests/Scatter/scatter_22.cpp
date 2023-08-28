#include <stdio.h>
#include <omp.h>
#include <cstdlib>
#include <math.h>

const bool v_flag = false; // Print verifications True=On False=Off

int num_dev = omp_get_num_devices();

void verify(void ** x_ptr, int chnk_len){
  for(int i=0; i<num_dev; ++i)  
  {
    #pragma omp target device(i) map(x_ptr[i]) 
    {
      int* x = (int*)x_ptr[i];
      printf("%s No. %d ArrX = ", omp_is_initial_device()?"Host":"Device", omp_get_device_num());
      for (int j=0; j<chnk_len; ++j){
        printf("%d ", *(x+j));
      }
      printf("\n");
    }
  }
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
  int chnk_len = 4;
  // int chnk_len = 256; // 1 Kb
  // int chnk_len = 2560; // 10 Kb
  // int chnk_len = 25600; // 100 Kb
  // int chnk_len = 262144; // 1 Mb
  // int chnk_len = 2621440; // 10 Mb
  // int chnk_len = 26214400; // 100 Mb
  // int chnk_len = 268435456; // 1 Gb

  if(const char* chnk_size = std::getenv("CHK_SZ"))
    chnk_len = atoi(chnk_size);

  int arr_len = chnk_len*num_dev;

  double start, end;
  int v_no = 0;
  int n_id, main_no, sec_no, n_deps;
  int *main_arr, *sec_dep_arr;

  int* x_arr = (int*) malloc(arr_len * sizeof(int)); 
  
  // Set device pointers
  void ** x_ptr = (void **) malloc(sizeof(void**) * num_dev+1); 
  if (!x_ptr) {
    printf("Memory Allocation Failed\n");
    exit(1);
  }

  size_t a_size = sizeof(int) * arr_len;
  size_t c_size = sizeof(int) * chnk_len;
  // size_t * ct_size = (size_t*) malloc(num_dev * 2 * sizeof(size_t)); // Chunk size and offsets

  // Add host pointer 
  x_ptr[num_dev]=&x_arr[0];

  printf("[Scatter No. Dev: %d Int Array Size = %zu] \n", omp_get_num_devices(), c_size);

//**************************************************//
//            Host-to-all (Sequential)              //
//**************************************************//

  if (v_flag) printf("Host-to-all (Sequential)\n");

  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);
  
  // Allocate device memory -> Allocating different sizes for each device based on the strategy
  // ToDo: Consider arrays remainder 
  for (int dev = 0; dev < num_dev; ++dev){
    x_ptr[dev] = omp_target_alloc(c_size, dev);
  }
  
  start = omp_get_wtime(); 
  for (int dev = 0; dev < num_dev; ++dev)
  {
    omp_target_memcpy(
      x_ptr[dev],                         // dst
      x_ptr[omp_get_initial_device()],    // src
      c_size,                             // length 
      0,                                  // dst_offset
      c_size*dev,                         // src_offset, 
      dev,                                // dst_device_num
      omp_get_initial_device()            // src_device_num
    );
  }

  end = omp_get_wtime();
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);
  if (v_flag) verify(x_ptr, chnk_len);

  for (int dev = 0; dev < num_dev; ++dev)
    omp_target_free(x_ptr[dev], dev);


//**************************************************//
//              Host-to-all (Parallel)              //
//**************************************************//

  if (v_flag) printf("Host-to-all (Parallel)\n");
  
  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);

  // Allocate device memory -> Allocating different sizes for each device based on the strategy
  // ToDo: Consider arrays remainder 
  for (int dev = 0; dev < num_dev; ++dev){
    x_ptr[dev] = omp_target_alloc(c_size, dev);
  }

  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      start = omp_get_wtime();
    }
    int dev = omp_get_thread_num();
    omp_target_memcpy(
      x_ptr[dev],                         // dst
      x_ptr[omp_get_initial_device()],    // src
      c_size,                             // length 
      0,                                  // dst_offset
      c_size*dev,                         // src_offset, 
      dev,                                // dst_device_num
      omp_get_initial_device()            // src_device_num
    );
    #pragma omp barrier
    #pragma omp single
    {
      end = omp_get_wtime();
    }
    #pragma omp barrier
  }

  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  
  if (v_flag) verify(x_ptr, chnk_len);

  for (int dev = 0; dev < num_dev; ++dev)
    omp_target_free(x_ptr[dev], dev);


//**************************************************//
//         Host-to-Main -> Main-to-local v1         //
//**************************************************//
  
  if (v_flag) printf("Host-to-Main -> Main-to-local v1\n");
  
  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Allocate device memory -> Allocating different sizes for each device based on the strategy
  // ToDo: Consider arrays remainder 
  // for (int dev = 0; dev < num_dev; ++dev){
  //   x_ptr[dev] = omp_target_alloc(c_size, dev);
  // }

  // Set up Main - Secondary 
  main_no = 2;
  main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  main_arr[0] = 0;
  x_ptr[main_arr[0]] = omp_target_alloc(c_size*4, main_arr[0]);
  main_arr[1] = 4;
  x_ptr[main_arr[1]] = omp_target_alloc(c_size*4, main_arr[1]);
  
  sec_no = 3;
  n_deps = main_no * sec_no;
  sec_dep_arr = (int*) malloc(n_deps * 2 * sizeof(int));

  // Set up dependencies
  n_id = 1;
  for (int i=0; i<main_no; ++i)
  {
    for (int j=0; j<sec_no; ++j)
    {
      sec_dep_arr[i*sec_no+j] = n_id;
      sec_dep_arr[i*sec_no+j+n_deps] = main_arr[i];
      // printf("Main %d -> Local %d \n", sec_dep_arr[i*sec_no+j+n_deps], sec_dep_arr[i*sec_no+j]);
      x_ptr[n_id] = omp_target_alloc(c_size, n_id);
      n_id++;
    }
    n_id++;
  }

  if (v_flag) 
  {
    printf("Dependencies check \n");
    for (int i=0; i<main_no; ++i){
      for (int j=0; j<sec_no; ++j){
        printf("Main %d -> Local %d \n", sec_dep_arr[i*sec_no+j+n_deps], sec_dep_arr[i*sec_no+j]);
      }
    }
  }

  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr, main_arr, sec_dep_arr)
  {
    #pragma omp single
    {
      start = omp_get_wtime(); 
      for (int i=0; i<main_no; ++i){
        #pragma omp task depend(out:main_arr[i]) firstprivate(i) shared(x_ptr, main_arr, sec_dep_arr)
        {
          // printf("To Main %d\n", main_arr[i]);
          omp_target_memcpy(
            x_ptr[main_arr[i]],                 // dst
            x_ptr[omp_get_initial_device()],    // src
            c_size*4,                           // length 
            0,                                  // dst_offset
            c_size*i*4,                         // src_offset, 
            main_arr[i],                        // dst_device_num
            omp_get_initial_device()            // src_device_num
          );
        }
      }
      for (int i=0; i<main_no; ++i){
        for (int j=0; j<sec_no; ++j){
          int m_n = sec_dep_arr[i*sec_no+j+n_deps];
          int s_n = sec_dep_arr[i*sec_no+j];
          #pragma omp task depend(in:main_arr[i]) firstprivate(m_n, s_n) shared(x_ptr, main_arr, sec_dep_arr)
          {
            // printf("Main %d -> Local %d \n", m_n, s_n);
            omp_target_memcpy(
              x_ptr[s_n],                       // dst
              x_ptr[m_n],                       // src
              c_size,                           // length 
              0,                                // dst_offset
              c_size*(j+1),                     // src_offset, 
              s_n,                              // dst_device_num
              m_n                               // src_device_num
            );
          }
        }
      }
      #pragma omp taskwait
      end = omp_get_wtime();
    }
  }
  
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, chnk_len);

  free (main_arr);
  free (sec_dep_arr);


//**************************************************//
//         Host-to-Main -> Main-to-local v2         //
//**************************************************//
  
  if (v_flag) printf("Host-to-Main -> Main-to-local v2\n");
  
  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 2;
  main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  main_arr[0] = 2;
  x_ptr[main_arr[0]] = omp_target_alloc(c_size*4, main_arr[0]);
  main_arr[1] = 6;
  x_ptr[main_arr[1]] = omp_target_alloc(c_size*4, main_arr[1]);

  sec_no = 3;
  n_deps = main_no * sec_no;
  sec_dep_arr = (int*) malloc(n_deps * 2 * sizeof(int));

  // Set up dependencies
  n_id = 0;
  for (int i=0; i<main_no; ++i)
  {
    for (int j=0; j<sec_no; ++j)
    {
      if (n_id == main_arr[0] || n_id == main_arr[1])
        n_id++;
      sec_dep_arr[i*sec_no+j] = n_id;
      sec_dep_arr[i*sec_no+j+n_deps] = main_arr[i];
      // printf("Main %d -> Local %d \n", sec_dep_arr[i*sec_no+j+n_deps], sec_dep_arr[i*sec_no+j]);
      x_ptr[n_id] = omp_target_alloc(c_size, n_id);
      n_id++;
    }
  }

  if (v_flag) 
  {
    printf("Dependencies check \n");
    for (int i=0; i<main_no; ++i){
      for (int j=0; j<sec_no; ++j){
        printf("Main %d -> Local %d \n", sec_dep_arr[i*sec_no+j+n_deps], sec_dep_arr[i*sec_no+j]);
      }
    }
  }

  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr, main_arr, sec_dep_arr)
  {
    #pragma omp single
    {
      start = omp_get_wtime(); 
      for (int i=0; i<main_no; ++i){
        n_id = 0;
        #pragma omp task depend(out:main_arr[i]) firstprivate(i, n_id) shared(x_ptr, main_arr)
        {
          // printf("To Main %d\n", main_arr[i]);
          for (int j=0; j<sec_no+1; ++j){
            if (i*(sec_no+1)+j == main_arr[i]){
              omp_target_memcpy(
                x_ptr[main_arr[i]],                 // dst
                x_ptr[omp_get_initial_device()],    // src
                c_size,                             // length 
                0,                                  // dst_offset
                c_size*(main_arr[i]),               // src_offset
                main_arr[i],                        // dst_device_num
                omp_get_initial_device()            // src_device_num
              );
          // }
        // }
        // #pragma omp task firstprivate(i) shared(x_ptr, main_arr, sec_dep_arr)
        // {
          // printf("To Main %d\n", main_arr[i]);
            }else{
              omp_target_memcpy(
                x_ptr[main_arr[i]],                 // dst
                x_ptr[omp_get_initial_device()],    // src
                c_size,                             // length 
                c_size*(n_id+1),                    // dst_offset
                c_size*(j+i*(sec_no+1)),            // src_offset j+i*sec_no
                main_arr[i],                        // dst_device_num
                omp_get_initial_device()            // src_device_num
              );
              n_id++;
            }
          }
        }
      }
      for (int i=0; i<main_no; ++i){
        for (int j=0; j<sec_no; ++j){
          int m_n = sec_dep_arr[i*sec_no+j+n_deps];
          int s_n = sec_dep_arr[i*sec_no+j];
          #pragma omp task depend(in:main_arr[i]) firstprivate(m_n, s_n) shared(x_ptr, main_arr, sec_dep_arr)
          {
            // printf("Main %d -> Local %d \n", m_n, s_n);
            omp_target_memcpy(
              x_ptr[s_n],                       // dst
              x_ptr[m_n],                       // src
              c_size,                           // length 
              0,                                // dst_offset
              c_size*(j+1),                     // src_offset, 
              s_n,                              // dst_device_num
              m_n                               // src_device_num
            );
          }
        }
      }
      #pragma omp taskwait
      end = omp_get_wtime();
    }
  }
  
  // end time was here 
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, chnk_len);

  free (main_arr);
  free (sec_dep_arr);

  for (int dev = 0; dev < num_dev; ++dev)
    omp_target_free(x_ptr[dev], dev);

//**************************************************//
//         Host-to-Main -> Main-to-local v3         //
//**************************************************//

  if (v_flag) printf("Host-to-Main -> Main-to-local v3\n");
  
  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Allocate device memory -> Allocating different sizes for each device based on the strategy
  // ToDo: Consider arrays remainder 
  // Set up Main - Secondary 
  main_no = 2;
  main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  sec_no = 3;
  n_deps = main_no * sec_no;
  sec_dep_arr = (int*) malloc(n_deps * 2 * sizeof(int));

  main_arr[0] = 3;
  x_ptr[main_arr[0]] = omp_target_alloc(c_size*(sec_no+1), main_arr[0]);
  main_arr[1] = 7;
  x_ptr[main_arr[1]] = omp_target_alloc(c_size*(sec_no+1), main_arr[1]);
  


  // Set up dependencies
  n_id = 0;
  for (int i=0; i<main_no; ++i)
  {
    for (int j=0; j<sec_no; ++j)
    {
      if (n_id == main_arr[0] || n_id == main_arr[1])
        n_id++;
      sec_dep_arr[i*sec_no+j] = n_id;
      sec_dep_arr[i*sec_no+j+n_deps] = main_arr[i];
      // printf("Main %d -> Local %d \n", sec_dep_arr[i*sec_no+j+n_deps], sec_dep_arr[i*sec_no+j]);
      x_ptr[n_id] = omp_target_alloc(c_size, n_id);
      n_id++;
    }
    n_id++;
  }

  if (v_flag) 
  {
    printf("Dependencies check \n");
    for (int i=0; i<main_no; ++i){
      for (int j=0; j<sec_no; ++j){
        printf("Main %d -> Local %d \n", sec_dep_arr[i*sec_no+j+n_deps], sec_dep_arr[i*sec_no+j]);
      }
    }
  }

  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr, main_arr, sec_dep_arr)
  {
    #pragma omp single
    {
      start = omp_get_wtime(); 
      for (int i=0; i<main_no; ++i){
        n_id = 0;
        #pragma omp task depend(out:main_arr[i]) firstprivate(i, n_id) shared(x_ptr, main_arr, sec_dep_arr)
        {
          // printf("To Main %d\n", main_arr[i]);
          for (int j=0; j<sec_no+1; ++j){
            if (i*(sec_no+1)+j == main_arr[i]){
              omp_target_memcpy(
                x_ptr[main_arr[i]],                 // dst
                x_ptr[omp_get_initial_device()],    // src
                c_size,                             // length 
                0,                                  // dst_offset
                c_size*(main_arr[i]),               // src_offset
                main_arr[i],                        // dst_device_num
                omp_get_initial_device()            // src_device_num
              );
          // }
        // }
        // #pragma omp task firstprivate(i) shared(x_ptr, main_arr, sec_dep_arr)
        // {
          // printf("To Main %d\n", main_arr[i]);
            }else{
              omp_target_memcpy(
                x_ptr[main_arr[i]],                 // dst
                x_ptr[omp_get_initial_device()],    // src
                c_size,                             // length 
                c_size*(n_id+1),                    // dst_offset
                c_size*(j+i*(sec_no+1)),            // src_offset j+i*sec_no
                main_arr[i],                        // dst_device_num
                omp_get_initial_device()            // src_device_num
              );
              n_id++;
            }
          }
        }
      }
      for (int i=0; i<main_no; ++i){
        for (int j=0; j<sec_no; ++j){
          int m_n = sec_dep_arr[i*sec_no+j+n_deps];
          int s_n = sec_dep_arr[i*sec_no+j];
          #pragma omp task depend(in:main_arr[i]) firstprivate(m_n, s_n) shared(x_ptr, main_arr, sec_dep_arr)
          {
            // printf("Main %d -> Local %d \n", m_n, s_n);
            omp_target_memcpy(
              x_ptr[s_n],                       // dst
              x_ptr[m_n],                       // src
              c_size,                           // length 
              0,                                // dst_offset
              c_size*(j+1),                     // src_offset, 
              s_n,                              // dst_device_num
              m_n                               // src_device_num
            );
          }
        }
      }
      #pragma omp taskwait
      end = omp_get_wtime();
    }
  }
  
  // end time was here 
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  // if (v_flag) verify(x_ptr, arr_len);
  if (v_flag) verify(x_ptr, chnk_len);


  free (main_arr);
  free (sec_dep_arr);

  for (int dev = 0; dev < num_dev; ++dev)
    omp_target_free(x_ptr[dev], dev);


//**************************************************//
//               Splitted Linked List               //
//**************************************************//

  if (v_flag) printf("Splitted Linked List\n");

  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);

  // Set up dependencies
  main_no = 2;
  sec_no = 3;
  main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  main_arr[0] = 0;
  x_ptr[main_arr[0]] = omp_target_alloc(c_size*4, main_arr[0]);
  main_arr[1] = 3;
  x_ptr[main_arr[1]] = omp_target_alloc(c_size*3, main_arr[1]);
  main_arr[2] = 2;
  x_ptr[main_arr[2]] = omp_target_alloc(c_size*2, main_arr[2]);
  main_arr[3] = 1;
  x_ptr[main_arr[3]] = omp_target_alloc(c_size, main_arr[3]);

  sec_dep_arr = (int*) malloc(main_no * sizeof(int)); 

  sec_dep_arr[0] = 4;
  x_ptr[sec_dep_arr[0]] = omp_target_alloc(c_size*4, sec_dep_arr[0]);
  sec_dep_arr[1] = 7;
  x_ptr[sec_dep_arr[1]] = omp_target_alloc(c_size*3, sec_dep_arr[1]);
  sec_dep_arr[2] = 6;
  x_ptr[sec_dep_arr[2]] = omp_target_alloc(c_size*2, sec_dep_arr[2]);
  sec_dep_arr[3] = 5;
  x_ptr[sec_dep_arr[3]] = omp_target_alloc(c_size, sec_dep_arr[3]);

  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      start = omp_get_wtime();

      #pragma omp task depend(out:main_arr[0]) shared(x_ptr, main_arr)
      {
        // printf("To Main %d\n", main_arr[i]);
        for (int i=0; i<sec_no+1; ++i){
          n_id = 0;
          if (i == main_arr[0]){
            omp_target_memcpy(
              x_ptr[main_arr[0]],                 // dst
              x_ptr[omp_get_initial_device()],    // src
              c_size,                             // length 
              0,                                  // dst_offset
              c_size*(main_arr[i]),               // src_offset
              main_arr[0],                        // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
          }else{
            omp_target_memcpy(
              x_ptr[main_arr[0]],                 // dst
              x_ptr[omp_get_initial_device()],    // src
              c_size,                             // length 
              c_size*(i),                         // dst_offset
              c_size*(main_arr[i]),               // src_offset j+i*sec_no
              main_arr[0],                        // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
          }
          n_id++;
        }
      }
      #pragma omp task depend(out:sec_dep_arr[0]) shared(x_ptr, sec_dep_arr)
      {
        // printf("To Main %d\n", main_arr[i]);
        for (int i=0; i<sec_no+1; ++i){
          n_id = 0;
          if (i == sec_dep_arr[0]){
            omp_target_memcpy(
              x_ptr[sec_dep_arr[0]],              // dst
              x_ptr[omp_get_initial_device()],    // src
              c_size,                             // length 
              0,                                  // dst_offset
              c_size*(sec_dep_arr[i]),            // src_offset
              sec_dep_arr[0],                     // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
          }else{
            omp_target_memcpy(
              x_ptr[sec_dep_arr[0]],              // dst
              x_ptr[omp_get_initial_device()],    // src
              c_size,                             // length 
              c_size*(i),                         // dst_offset
              c_size*(sec_dep_arr[i]),            // src_offset j+i*sec_no
              sec_dep_arr[0],                     // dst_device_num
              omp_get_initial_device()            // src_device_num
            );
          }
          n_id++;
        }
      }
      
      for(int i=1; i<num_dev/2; ++i){
        #pragma omp task depend(in:main_arr[i-1]) depend(out:main_arr[i]) firstprivate(i)
        omp_target_memcpy(
          x_ptr[main_arr[i]],                 // dst
          x_ptr[main_arr[i-1]],               // src
          c_size*(sec_no+1-i),                // length 
          0,                                  // dst_offset
          c_size,                             // src_offset, 
           main_arr[i],                       // dst_device_num FIX DEST main_arr[i]
           main_arr[i-1]                      // src_device_num FIX SRC main_arr[i-1]
        );
        #pragma omp task depend(in:sec_dep_arr[i-1]) depend(out:sec_dep_arr[i]) firstprivate(i)
        omp_target_memcpy(
          x_ptr[sec_dep_arr[i]],              // dst
          x_ptr[sec_dep_arr[i-1]],            // src
          c_size*(sec_no+1-i),                // length 
          0,                                  // dst_offset
          c_size,                             // src_offset, 
          sec_dep_arr[i],                     // dst_device_num FIX DEST sec_dep_arr[i]
          sec_dep_arr[i-1]                    // src_device_num FIX SRC sec_dep_arr[i-1]
        );
      }
      #pragma omp taskwait
      end = omp_get_wtime();
    }
  }

  // end = omp_get_wtime();
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  // if (v_flag) verify(x_ptr, arr_len);
  if (v_flag) verify(x_ptr, chnk_len);
  
  free (main_arr);
  free (sec_dep_arr);

  for (int dev = 0; dev < num_dev; ++dev)
    omp_target_free(x_ptr[dev], dev);

//**************************************************//
//         Splitted Linked List Not Ordered         //
//**************************************************//
/*
  if (v_flag) printf("Splitted Linked List\n");

  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);

  // Set up dependencies

  main_no = 2;
  sec_no = 3;
  main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  main_arr[0] = 0;
  x_ptr[main_arr[0]] = omp_target_alloc(c_size*4, main_arr[0]);
  main_arr[1] = 3;
  x_ptr[main_arr[1]] = omp_target_alloc(c_size*3, main_arr[1]);
  main_arr[2] = 2;
  x_ptr[main_arr[2]] = omp_target_alloc(c_size*2, main_arr[2]);
  main_arr[3] = 1;
  x_ptr[main_arr[3]] = omp_target_alloc(c_size, main_arr[3]);

  sec_dep_arr = (int*) malloc(main_no * sizeof(int)); 

  sec_dep_arr[0] = 4;
  x_ptr[sec_dep_arr[0]] = omp_target_alloc(c_size*4, sec_dep_arr[0]);
  sec_dep_arr[1] = 7;
  x_ptr[sec_dep_arr[1]] = omp_target_alloc(c_size*3, sec_dep_arr[1]);
  sec_dep_arr[2] = 6;
  x_ptr[sec_dep_arr[2]] = omp_target_alloc(c_size*2, sec_dep_arr[2]);
  sec_dep_arr[3] = 5;
  x_ptr[sec_dep_arr[3]] = omp_target_alloc(c_size, sec_dep_arr[3]);

  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      start = omp_get_wtime();
      int dep_arr[num_dev];
      #pragma omp task depend(out:main_arr[0])
        omp_target_memcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          c_size*(sec_no+1),                  // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          0,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      #pragma omp task depend(out:sec_dep_arr[0])
        omp_target_memcpy(
          x_ptr[num_dev/2],                   // dst
          x_ptr[omp_get_initial_device()],    // src
          c_size*(sec_no+1),                  // length 
          0,                                  // dst_offset
          c_size*(sec_no+1),                  // src_offset, 
          num_dev/2,                          // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      for(int i=1; i<num_dev/2; ++i){
        #pragma omp task depend(in:main_arr[i-1]) depend(out:main_arr[i]) firstprivate(i)
        omp_target_memcpy(
          x_ptr[main_arr[i]],                 // dst
          x_ptr[main_arr[i-1]],               // src
          c_size*(sec_no+1-i),                // length 
          0,                                  // dst_offset
          c_size,                             // src_offset, 
           main_arr[i],                       // dst_device_num FIX DEST main_arr[i]
           main_arr[i-1]                      // src_device_num FIX SRC main_arr[i-1]
        );
        #pragma omp task depend(in:sec_dep_arr[i-1]) depend(out:sec_dep_arr[i]) firstprivate(i)
        omp_target_memcpy(
          x_ptr[sec_dep_arr[i]],              // dst
          x_ptr[sec_dep_arr[i-1]],            // src
          c_size*(sec_no+1-i),                // length 
          0,                                  // dst_offset
          c_size,                             // src_offset, 
          sec_dep_arr[i],                     // dst_device_num FIX DEST sec_dep_arr[i]
          sec_dep_arr[i-1]                    // src_device_num FIX SRC sec_dep_arr[i-1]
        );
      }
      #pragma omp taskwait
      end = omp_get_wtime();
    }
  }

  // end = omp_get_wtime();
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  // if (v_flag) verify(x_ptr, arr_len);
  if (v_flag) verify(x_ptr, chnk_len);
  
  free (main_arr);
  free (sec_dep_arr);

  for (int dev = 0; dev < num_dev; ++dev)
    omp_target_free(x_ptr[dev], dev);
*/

  free(x_ptr);

  return 0;
}

// PROG=scatter_22; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -L/soft/compilers/cuda/cuda-11.0.2/lib64 -L/soft/compilers/cuda/cuda-11.0.2/targets/x86_64-linux/lib/ -I/soft/compilers/cuda/cuda-11.0.2/include -ldl -lcudart -pthread $PROG.cpp
// ./scatter_22.x 
// nvprof --print-gpu-trace ./scatter_22.x 
// ./scatter_22.x>scatter_22_val.o  2>&1
// ./scatter_22.x 2>&1 | tee scatter_22_test.o
// nsys profile -o scatter_22_prof --stats=true ./scatter_22.x
// PROG=scatter_22; ASIZE=100M; nsys profile -o $PROG\_$ASIZE\_0 --stats=true ./$PROG.x
// paste -d"\t" results_s22_2/* 2>&1 | tee test.o
// paste -d"\t" results_s22_2/*>test.o 2>&1 
// paste -d"\t"  out_test_256.o out_test_2560.o out_test_25600.o out_test_262144.o out_test_2621440.o out_test_26214400.o out_test_209715200.o out_test_235929600.o>test.o 2>&1

