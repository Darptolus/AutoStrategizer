#include <stdio.h>
#include <omp.h>
#include <cstdlib>
#include <vector>

// Testing omp_target_memcpy -> Used for testing different strategies -> printed validation 

const bool v_flag = false; // Print verifications True=On False=Off

void verify(void ** x_ptr, int arr_len){
  int num_dev = omp_get_num_devices();  
  for(int i=0; i<num_dev; ++i)
  {
    #pragma omp target device(i) map(x_ptr[i], arr_len) 
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

// Dependencies
struct Dep{
  int main;
  int sec;
  Dep (int a_m, int a_s){this->main = a_m; this->sec = a_s;};
};

class Depends{
  private:
    Dep * a_dep;
    std::vector<Dep*> all_deps;
    std::vector<Dep*>::iterator deps_ptr;
  public:
    Depends(){};
    ~Depends(){};
    void add_dep(int main, int sec)
    {
      Dep * a_dep = new Dep(main,sec);
      all_deps.push_back(a_dep);
    };
    void clear_dep(){all_deps.clear();};
    // int get_main(){return this->main;};
    // int get_sec(){return this->sec;};
    // int get_beg(){return this->main;};
    // int get_end(){return this->sec;};
};

int main()
{
  int arr_len = 10;
  // int arr_len = 256; // 1 Kb
  // int arr_len = 2560; // 10 Kb
  // int arr_len = 25600; // 100 Kb
  // int arr_len = 262144; // 1 Mb
  // int arr_len = 2621440; // 10 Mb
  // int arr_len = 26214400; // 100 Mb
  // int arr_len = 268435456; // 1 Gb

  if(const char* arr_size = std::getenv("ARR_SZ"))
    arr_len = atoi(arr_size);

  double start, end;
  int v_no = 0;
  int n_id, main_no, sec_no, n_deps;
  int *main_arr, *sec_dep_arr;
  int num_dev = omp_get_num_devices();  
  int *x_arr = (int*) malloc(arr_len * sizeof(int)); 

  // Set device pointers
  void ** x_ptr = (void **) malloc(sizeof(void**) * num_dev+1); 
  if (!x_ptr) {
    printf("Memory Allocation Failed\n");
    exit(1);
  }
  size_t size = sizeof(int) * arr_len;

  // Allocate device memory
  for (int dev = 0; dev < num_dev; ++dev)
    x_ptr[dev] = omp_target_alloc(size, dev);

  // Add host pointer 
  x_ptr[num_dev]=&x_arr[0];

  // Dependencies
  // Depends deps;
  Dep * dep_arr;

  printf("[Broadcast No. Dev: %d Int Array Size = %zu] \n", omp_get_num_devices(), size);

//**************************************************//
//            Host-to-all (Sequential)              //
//**************************************************//

  if (v_flag) printf("Host-to-all (Sequential)\n");

  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);
  
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
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);
  if (v_flag) verify(x_ptr, arr_len);


//**************************************************//
//              Host-to-all (Parallel)              //
//**************************************************//

  if (v_flag) printf("Host-to-all (Parallel)\n");

  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);

  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      start = omp_get_wtime();
    }
    omp_target_memcpy(
      x_ptr[omp_get_thread_num()],        // dst
      x_ptr[omp_get_initial_device()],    // src
      size,                               // length 
      0,                                  // dst_offset
      0,                                  // src_offset, 
      omp_get_thread_num(),               // dst_device_num
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

  if (v_flag) verify(x_ptr, arr_len);


//**************************************************//
//         Host-to-Main -> Main-to-local v1         //
//**************************************************//
  
  if (v_flag) printf("Host-to-Main -> Main-to-local v1\n");
  
  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 2;
  main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  main_arr[0] = 0;
  main_arr[1] = 4;
  
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

  // start time was here 
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
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
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
              size,                             // length 
              0,                                // dst_offset
              0,                                // src_offset, 
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

  if (v_flag) verify(x_ptr, arr_len);

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
  main_arr[1] = 6;
  
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

  // start time was here 
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
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
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
              size,                             // length 
              0,                                // dst_offset
              0,                                // src_offset, 
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

  if (v_flag) verify(x_ptr, arr_len);

  free (main_arr);
  free (sec_dep_arr);


//**************************************************//
//         Host-to-Main -> Main-to-local v3         //
//**************************************************//
  
  if (v_flag) printf("Host-to-Main -> Main-to-local v3\n");
  
  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 2;
  main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  main_arr[0] = 3;
  main_arr[1] = 7;
  
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

  // start time was here 
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
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
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
              size,                             // length 
              0,                                // dst_offset
              0,                                // src_offset, 
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

  if (v_flag) verify(x_ptr, arr_len);

  free (main_arr);
  free (sec_dep_arr);


//**************************************************//
//               Splitted Linked List               //
//**************************************************//

  if (v_flag) printf("Splited Linked List\n");

  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);

  // Set up dependencies

  main_no = 2;
  sec_no = 3;
  main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  main_arr[0] = 0;
  main_arr[1] = 3;
  main_arr[2] = 2;
  main_arr[3] = 1;

  sec_dep_arr = (int*) malloc(main_no * sizeof(int)); 

  sec_dep_arr[0] = 4;
  sec_dep_arr[1] = 7;
  sec_dep_arr[2] = 6;
  sec_dep_arr[3] = 5;
  
  // start = omp_get_wtime();
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
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          0,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      #pragma omp task depend(out:sec_dep_arr[0])
        omp_target_memcpy(
          x_ptr[num_dev/2],                   // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          num_dev/2,                          // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      for(int i=1; i<num_dev/2; ++i){
        #pragma omp task depend(in:main_arr[i-1]) depend(out:main_arr[i]) firstprivate(i)
        omp_target_memcpy(
          x_ptr[main_arr[i]],                           // dst
          x_ptr[main_arr[i-1]],                         // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          main_arr[i],                        // dst_device_num FIX DEST main_arr[i]
          main_arr[i-1]                       // src_device_num FIX SRC main_arr[i-1]
        );
        #pragma omp task depend(in:sec_dep_arr[i-1]) depend(out:sec_dep_arr[i]) firstprivate(i)
        omp_target_memcpy(
          x_ptr[sec_dep_arr[i]],              // dst x_ptr[num_dev/2+i]
          x_ptr[sec_dep_arr[i-1]],            // src x_ptr[num_dev/2+i-1]
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          sec_dep_arr[i],                     // dst_device_num FIX DEST sec_dep_arr[num_dev/2+i]
          sec_dep_arr[i-1]                    // src_device_num FIX SRC sec_dep_arr[num_dev/2+i-1]
        );
      }
      #pragma omp taskwait
      end = omp_get_wtime();
    }
  }

  // end = omp_get_wtime();
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, arr_len);
  
  free (main_arr);
  free (sec_dep_arr);

  for (int dev = 0; dev < num_dev; ++dev)
    omp_target_free(x_ptr[dev], dev);

  free(x_ptr);

  return 0;
}

// PROG=broadcast_22; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -L/soft/compilers/cuda/cuda-11.0.2/lib64 -L/soft/compilers/cuda/cuda-11.0.2/targets/x86_64-linux/lib/ -I/soft/compilers/cuda/cuda-11.0.2/include -ldl -lcudart -pthread $PROG.cpp
// ./broadcast_22.x 
// ./broadcast_22.x>broadcast_22_val.o  2>&1
// ./broadcast_22.x 2>&1 | tee out_test.o
// nsys profile -o broadcast_22_prof --stats=true ./broadcast_22.x
