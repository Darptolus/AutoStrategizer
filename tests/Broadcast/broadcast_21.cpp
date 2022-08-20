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
  for (int dev = 0; dev < num_dev; ++dev)
    x_ptr[dev] = omp_target_alloc(size, dev);

  // Add host pointer 
  x_ptr[num_dev]=&x_arr[0];

  // Dependencies
  // Depends deps;
  Dep * dep_arr;

  printf("[Broadcast NDev: %d Int ASize = %zu] \n", omp_get_num_devices(), size);

//**************************************************//
//            Host-to-all (Sequential)              //
//**************************************************//

  if (v_flag) printf("Host-to-all (Sequential)\n");

  // Initialize Array Values
  v_no = init_arr(x_arr, v_no, arr_len);
  
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
  v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);

  // start = omp_get_wtime(); 
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
  }
  
  // end = omp_get_wtime();
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, arr_len);


//**************************************************//
//          Host-to-Main -> Main-to-local           //
//**************************************************//
  
  if (v_flag) printf("Host-to-Main -> Main-to-local\n");
  
  // Initialize Array Values
  v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 2;
  int* main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  main_arr[0] = 0;
  main_arr[1] = 4;
  
  sec_no = 3;
  n_deps = main_no * sec_no;
  int* sec_dep_arr = (int*) malloc(n_deps * 2 * sizeof(int));

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

  // start = omp_get_wtime(); 
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr, main_arr, sec_dep_arr)
  {
    #pragma omp single
    start = omp_get_wtime(); 
    {
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

  // end = omp_get_wtime();
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, arr_len);

  free (main_arr);
  free (sec_dep_arr);

//**************************************************//
//                   Half & Half                    //
//**************************************************//
  
  if (v_flag) printf("Half & Half\n");
  
  // Initialize Array Values
  v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 4;
  main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  for (int i=0; i<main_no; ++i)
    main_arr[i] = i*2;
  
  sec_no = 1;
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

  // start = omp_get_wtime(); 
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

  // end = omp_get_wtime();
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, arr_len);

  free (main_arr);
  free (sec_dep_arr);

//**************************************************//
//            Host to Mid -> Mid to Mid             //
//**************************************************//
  
  if (v_flag) printf("Host to Mid -> Mid to Mid\n");
  
  // Initialize Array Values
  v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 4;
  main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  main_arr[0] = 0;
  main_arr[1] = 1;
  main_arr[2] = 4;
  main_arr[3] = 5;

  sec_no = 1;
  n_deps = main_no * sec_no;
  sec_dep_arr = (int*) malloc(n_deps * 2 * sizeof(int));

  // Set up dependencies

  for (int i=0; i<main_no; ++i)
  {
    for (int j=0; j<sec_no; ++j)
    {
      sec_dep_arr[i*sec_no+j] = main_arr[i]+2;
      sec_dep_arr[i*sec_no+j+n_deps] = main_arr[i];
      // printf("Main %d -> Local %d \n", sec_dep_arr[i*sec_no+j+n_deps], sec_dep_arr[i*sec_no+j]);
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

  // start = omp_get_wtime(); 
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

  // end = omp_get_wtime();
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, arr_len);

  free (main_arr);
  free (sec_dep_arr);
  

//**************************************************//
//        Host to Half -> Half to Secondary         //
//**************************************************//
  
  if (v_flag) printf("Host to Half -> Half to Secondary\n");
  
  // Initialize Array Values
  v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 4;
  main_arr = (int*) malloc(main_no * sizeof(int)); 
  
  main_arr[0] = 0;
  main_arr[1] = 1;
  main_arr[2] = 6;
  main_arr[3] = 7;

  sec_no = 1;
  int* sec_arr = (int*) malloc(main_no * sec_no * sizeof(int)); 
  
  sec_arr[0] = 4;
  sec_arr[1] = 5;
  sec_arr[2] = 2;
  sec_arr[3] = 3;

  n_deps = main_no * sec_no;
  sec_dep_arr = (int*) malloc(n_deps * 2 * sizeof(int));

  // Set up dependencies

  for (int i=0; i<main_no; ++i)
  {
    sec_dep_arr[i] = sec_arr[i];
    sec_dep_arr[i+n_deps] = main_arr[i];
    // printf("Main %d -> Local %d \n", sec_dep_arr[i+n_deps], sec_dep_arr[i]);
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

  // start = omp_get_wtime(); 
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

  // end = omp_get_wtime();
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, arr_len);

  free (main_arr);
  free (sec_arr);
  free (sec_dep_arr);


//**************************************************//
//        Host to Main -> Main to Secondary         //
//**************************************************//

  if (v_flag) printf("Host to Main -> Main to Secondary\n");
  
  // Initialize Array Values
  v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 4;
  sec_no = 4;

  // Set up dependencies
  // ToDo: use class
  // for (int i=0; i<main_no; ++i)
  //   deps.add_dep(i, i+4);
  // for (auto it = myvector.begin(); it != myvector.end(); ++it)
  //   printf("Main %d -> Secondary %d \n", it->get_main(), it->get_sec());

  dep_arr = (Dep*) malloc((num_dev) * sizeof(Dep));

  for (int dev = 0; dev < num_dev/2; ++dev){
    dep_arr[dev].main = omp_get_initial_device();
    dep_arr[dev].sec = dev;
    dep_arr[dev+num_dev/2].main = dev;
    dep_arr[dev+num_dev/2].sec = dev+4;
  }
  
  if (v_flag) 
  {
    printf("Dependencies check \n");
    for (int i=0; i<num_dev; ++i){
      printf("Main %d -> Local %d \n", dep_arr[i].main, dep_arr[i].sec);
    }
  }

  // start = omp_get_wtime(); 
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      start = omp_get_wtime(); 
      for(int i=0; i<num_dev/2; ++i){
        #pragma omp task depend(out:dep_arr[i])
          omp_target_memcpy(
            x_ptr[dep_arr[i].sec],              // dst
            x_ptr[dep_arr[i].main],             // src
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
            dep_arr[i].sec,                     // dst_device_num
            dep_arr[i].main                     // src_device_num
          );
        #pragma omp task depend(in:dep_arr[i])
          omp_target_memcpy(
            x_ptr[dep_arr[i+num_dev/2].sec],    // dst
            x_ptr[dep_arr[i+num_dev/2].main],   // src
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
            dep_arr[i+num_dev/2].sec,           // dst_device_num
            dep_arr[i+num_dev/2].main           // src_device_num
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

free (dep_arr);


//**************************************************//
//            Host-to-one -> Linked List            //
//**************************************************//

  if (v_flag) printf("Host-to-one -> Linked List\n");
  
  // Initialize Array Values
  v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);

  dep_arr = (Dep*) malloc((num_dev) * sizeof(Dep));

  n_id = 0;
  dep_arr[n_id].main = omp_get_initial_device();
  dep_arr[n_id].sec = 0;

  n_id++;
  dep_arr[n_id].main = 0;
  dep_arr[n_id].sec = 1;

  n_id++;
  dep_arr[n_id].main = 1;
  dep_arr[n_id].sec = 2;

  n_id++;
  dep_arr[n_id].main = 2;
  dep_arr[n_id].sec = 3;

  n_id++;
  dep_arr[n_id].main = 3;
  dep_arr[n_id].sec = 7;
  
  n_id++;
  dep_arr[n_id].main = 7;
  dep_arr[n_id].sec = 6;
  
  n_id++;
  dep_arr[n_id].main = 6;
  dep_arr[n_id].sec = 5;

  n_id++;
  dep_arr[n_id].main = 5;
  dep_arr[n_id].sec = 4;

  if (v_flag) 
  {
    printf("Dependencies check \n");
    for (int i=0; i<num_dev; ++i){
      printf("Main %d -> Local %d \n", dep_arr[i].main, dep_arr[i].sec);
    }
  }

  // start = omp_get_wtime();
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr, dep_arr)
  {
    #pragma omp single
    {
      start = omp_get_wtime();
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
        #pragma omp task depend(in:dep_arr[i-1]) depend(out:dep_arr[i]) firstprivate(i) shared(x_ptr, dep_arr)
        omp_target_memcpy(
          x_ptr[dep_arr[i].sec],              // dst
          x_ptr[dep_arr[i].main],             // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          dep_arr[i].sec,                     // dst_device_num
          dep_arr[i].main                     // src_device_num
        );
      #pragma omp taskwait
      end = omp_get_wtime();
    }
  }
  
  // end = omp_get_wtime();
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, arr_len);

  free (dep_arr);


//**************************************************//
//               Splited Linked List                //
//**************************************************//

  if (v_flag) printf("Splited Linked List\n");

  // Initialize Array Values
  v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);
  
  // start = omp_get_wtime();
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      start = omp_get_wtime();
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
      #pragma omp taskwait
      end = omp_get_wtime();
    }
  }

  // end = omp_get_wtime();
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, arr_len);


  free(x_ptr);


  return 0;
}

// PROG=broadcast_20; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -L/soft/compilers/cuda/cuda-11.0.2/lib64 -L/soft/compilers/cuda/cuda-11.0.2/targets/x86_64-linux/lib/ -I/soft/compilers/cuda/cuda-11.0.2/include -ldl -lcudart -pthread $PROG.cpp
// ./broadcast_20.x 
// ./broadcast_20.x>out_test.o  2>&1


/*
// main_no = 2;
  // int* main_arr = (int*) malloc(main_no * sizeof(int)); 
  // main_arr= {0, 4};

  v_no+=10;
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+v_no;
  
  if (v_flag) print_hval(x_arr, arr_len);

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
  if (v_flag) verify(x_ptr, arr_len);

*/