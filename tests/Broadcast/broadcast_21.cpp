#include <stdio.h>
#include <omp.h>
#include <cstdlib>
// #include <vector>

// Testing omp_target_memcpy -> Used for testing different strategies -> printed validation 

const bool v_flag = false; // Print verifications True=On False=Off

// Dependencies
class Dep{
  private:
    int orig;
    int dest;
    int orig_id;
  public:
    Dep ();
    Dep (int a_o, int a_d):orig {a_o}, dest {a_d}{};
    Dep (int a_o, int a_d, int o_id):orig {a_o}, dest {a_d}, orig_id {o_id}{};
    ~Dep(){};
    int get_orig(){return this->orig;};
    int get_dest(){return this->dest;};
    int get_oid(){return this->orig_id;};
    void set_orig_id(int o_id){this->orig_id = o_id;};
    void set_dep(int a_o, int a_d)
    {
      this->orig =a_o;
      this->dest =a_d;
    };
    void set_dep(int a_o, int a_d, int o_id)
    {
      this->orig =a_o;
      this->dest =a_d;
      this->orig_id =o_id;
    };
};
    
// typedef std::vector<Dep*> all_deps;

void verify(void ** x_ptr, int arr_len)
{
  // Print value at devices for validation
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

void print_hval(int * x_arr, int arr_len)
{
  // Print value at host for validation
  printf("Host -> Value of X: ");
  for (int j=0; j<arr_len; ++j)
    printf("%d ", x_arr[j]);
  printf("\n");
}

int init_arr(int * x_arr, int v_no, int arr_len)
{
  // Initialize Array Values
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+v_no;
  return (v_no+=10);
}

void check_deps(int main_no, int sec_no, Dep * main_dep_arr, Dep * sec_dep_arr)
{
  int n_deps = main_no * sec_no;
  printf("Dependencies check \n");
  for (int i=0; i<main_no; ++i)
    printf("Origin %d -> Dest %d \n", main_dep_arr[i].get_orig(), main_dep_arr[i].get_dest());

  if (sec_no > 0)
  for (int j=0; j<n_deps; ++j)
    printf("Origin %d -> Dest %d \n", sec_dep_arr[j].get_orig(), sec_dep_arr[j].get_dest());
}



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
  // int arr_len = 1342177280; // 5 Gb

  if(const char* arr_size = std::getenv("ARR_SZ"))
    arr_len = atoi(arr_size);

  double start, end;
  int v_no = 0;
  int n_id, o_id, d_id, o_indx, main_id, sec_id, main_no, sec_no, n_deps;
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
  // all_deps deps;
  // deps_itr deps_it;
  // Dep * dep_arr;
  Dep * main_dep_arr, * sec_dep_arr;

  printf("Broadcast NDev: %d Int ASize = %zu \n", omp_get_num_devices(), size);

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
    #pragma omp barrier
  }
  
  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, arr_len);


//**************************************************//
//          Host-to-Main -> Main-to-local           //
//**************************************************//
  
  if (v_flag) printf("Host-to-Main -> Main-to-local\n");
  
  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 2;
  main_dep_arr = (Dep*) malloc(main_no * sizeof(Dep)); 
  // printf("size of dep= %zu \n", sizeof(Dep)); 

  sec_no = 3;
  n_deps = main_no * sec_no;
  sec_dep_arr = (Dep*) malloc(n_deps * sizeof(Dep));

  // Set up dependencies
  n_id = 0;
  sec_id = 0;
  for (int i=0; i<main_no; ++i)
  {
    main_id = n_id;
    main_dep_arr[i].set_dep(omp_get_initial_device(), main_id);
    n_id++;
    for (int j=0; j<sec_no; ++j)
    {
      sec_dep_arr[sec_id].set_dep(main_id, n_id, i);
      n_id++;
      sec_id++;
    }
  }

  // Check Dependencies
  if (v_flag) check_deps(main_no, sec_no, main_dep_arr, sec_dep_arr);

  #pragma omp parallel shared(x_ptr, main_dep_arr, sec_dep_arr)
  {
    #pragma omp single
    {
    start = omp_get_wtime(); 
    // Host to Main
    for (int i=0; i<main_no; ++i)
    {
      o_id = main_dep_arr[i].get_orig();
      d_id = main_dep_arr[i].get_dest();
      #pragma omp task depend(out:main_dep_arr[i]) firstprivate(o_id, d_id) shared(x_ptr)
      {
        // printf("To Main %d\n", d_id);
        omp_target_memcpy(
          x_ptr[d_id],                        // dst
          x_ptr[o_id],                        // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          d_id,                               // dst_device_num
          o_id                                // src_device_num
        );
      }
    }
    for (int j=0; j<n_deps; ++j)
    {
      o_indx = sec_dep_arr[j].get_oid();
      o_id = sec_dep_arr[j].get_orig();
      d_id = sec_dep_arr[j].get_dest();
      // Main to Secondary
      #pragma omp task depend(in:main_dep_arr[o_indx]) firstprivate(o_id, d_id) shared(x_ptr)
      {
        // printf("Main %d -> Local %d \n", o_id, d_id);
        omp_target_memcpy(
          x_ptr[d_id],                        // dst
          x_ptr[o_id],                        // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          d_id,                               // dst_device_num
          o_id                                // src_device_num
        );
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

free (main_dep_arr);
free (sec_dep_arr);

//**************************************************//
//      (Spread) Host to Half -> Half to Half       //
//**************************************************//

  if (v_flag) printf("(Spread) Host to Half -> Half to Half\n");
  
  // Initialize Array Values
  if (v_flag) if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 4;
  main_dep_arr = (Dep*) malloc(main_no * sizeof(Dep)); 

  sec_no = 1;
  n_deps = main_no * sec_no;
  sec_dep_arr = (Dep*) malloc(n_deps * sizeof(Dep));

  // Set up dependencies
  n_id = 0;
  sec_id = 0;
  for (int i=0; i<main_no; ++i)
  {
    main_id = n_id;
    main_dep_arr[i].set_dep(omp_get_initial_device(), main_id);
    n_id++;
    for (int j=0; j<sec_no; ++j)
    {
      sec_dep_arr[sec_id].set_dep(main_id, n_id, i);
      n_id++;
      sec_id++;
    }
  }

  // Check Dependencies
  if (v_flag) check_deps(main_no, sec_no, main_dep_arr, sec_dep_arr);

  // start = omp_get_wtime(); 
  #pragma omp parallel shared(x_ptr, main_dep_arr, sec_dep_arr)
  {
    #pragma omp single
    {
    start = omp_get_wtime(); 
    // Host to Main
    for (int i=0; i<main_no; ++i)
    {
      #pragma omp task depend(out:main_dep_arr[i]) firstprivate(i) shared(main_dep_arr)
      {
        // printf("To Main %d\n", main_dep_arr[i].get_dest());
        omp_target_memcpy(
          x_ptr[main_dep_arr[i].get_dest()],    // dst
          x_ptr[main_dep_arr[i].get_orig()],    // src
          size,                                 // length 
          0,                                    // dst_offset
          0,                                    // src_offset, 
          main_dep_arr[i].get_dest(),           // dst_device_num
          main_dep_arr[i].get_orig()            // src_device_num
        );
      }
    }
    for (int j=0; j<n_deps; ++j)
    {
      int orig_id = sec_dep_arr[j].get_oid();
      // Main to Secondary
      #pragma omp task depend(in:main_dep_arr[orig_id]) firstprivate(j) shared(x_ptr, main_dep_arr, sec_dep_arr)
      {
        // printf("Main %d -> Local %d \n", sec_dep_arr[j].get_orig(), sec_dep_arr[j].get_dest());
        omp_target_memcpy(
          x_ptr[sec_dep_arr[j].get_dest()],   // dst
          x_ptr[sec_dep_arr[j].get_orig()],   // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          sec_dep_arr[j].get_dest(),          // dst_device_num
          sec_dep_arr[j].get_orig()           // src_device_num
        );
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

  free (main_dep_arr);
  free (sec_dep_arr);

//**************************************************//
//      (Compact) Host to Half -> Half to Half      //
//**************************************************//

  if (v_flag) printf("(Compact) Host to Half -> Half to Half\n");
  
  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 4;
  main_dep_arr = (Dep*) malloc(main_no * sizeof(Dep)); 

  sec_no = 1;
  n_deps = main_no * sec_no;
  sec_dep_arr = (Dep*) malloc(n_deps * sizeof(Dep));

  // Set up dependencies
  n_id = 0;
  sec_id = 0;
  for (int i=0; i<main_no; ++i)
  {
    main_id = n_id;
    main_dep_arr[i].set_dep(omp_get_initial_device(), main_id);
    n_id++;
    ++i;
    main_id = n_id;
    main_dep_arr[i].set_dep(omp_get_initial_device(), main_id);
    n_id++;
    for (int j=0; j<sec_no; ++j)
    {
      sec_dep_arr[sec_id].set_dep(main_id-1, n_id, i-1);
      n_id++;
      sec_id++;
      sec_dep_arr[sec_id].set_dep(main_id, n_id, i);
      n_id++;
      sec_id++;
    }
  }

  // Check Dependencies
  if (v_flag) check_deps(main_no, sec_no, main_dep_arr, sec_dep_arr);

  #pragma omp parallel shared(x_ptr, main_dep_arr, sec_dep_arr)
  {
    #pragma omp single
    {
    start = omp_get_wtime(); 
    // Host to Main
    for (int i=0; i<main_no; ++i)
    {
      #pragma omp task depend(out:main_dep_arr[i]) firstprivate(i) shared(main_dep_arr)
      {
        // printf("To Main %d\n", main_dep_arr[i].get_dest());
        omp_target_memcpy(
          x_ptr[main_dep_arr[i].get_dest()],    // dst
          x_ptr[main_dep_arr[i].get_orig()],    // src
          size,                                 // length 
          0,                                    // dst_offset
          0,                                    // src_offset, 
          main_dep_arr[i].get_dest(),           // dst_device_num
          main_dep_arr[i].get_orig()            // src_device_num
        );
      }
    }
    for (int j=0; j<n_deps; ++j)
    {
      int orig_id = sec_dep_arr[j].get_oid();
      // Main to Secondary
      #pragma omp task depend(in:main_dep_arr[orig_id]) firstprivate(j) shared(x_ptr, main_dep_arr, sec_dep_arr)
      {
        // printf("Main %d -> Local %d \n", sec_dep_arr[j].get_orig(), sec_dep_arr[j].get_dest());
        omp_target_memcpy(
          x_ptr[sec_dep_arr[j].get_dest()],   // dst
          x_ptr[sec_dep_arr[j].get_orig()],   // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          sec_dep_arr[j].get_dest(),          // dst_device_num
          sec_dep_arr[j].get_orig()           // src_device_num
        );
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

free (main_dep_arr);
free (sec_dep_arr);
  

//**************************************************//
//        Host to Main -> Main to Secondary         //
//**************************************************//

  if (v_flag) printf("Host to Main -> Main to Secondary\n");
  
  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 4;
  main_dep_arr = (Dep*) malloc(main_no * sizeof(Dep)); 

  sec_no = 1;
  n_deps = main_no * sec_no;
  sec_dep_arr = (Dep*) malloc(n_deps * sizeof(Dep));

  // Set up dependencies
  n_id = 0;
  sec_id = 0;
  for (int i=0; i<main_no; ++i)
  {
    main_id = n_id;
    main_dep_arr[i].set_dep(omp_get_initial_device(), main_id);
    n_id++;
  }
  
  for (int j=0; j<n_deps; ++j)
  {
    sec_dep_arr[sec_id].set_dep(main_dep_arr[j].get_dest(), n_id, main_dep_arr[j].get_oid());
    n_id++;
    sec_id++;
  }

  // Check Dependencies
  if (v_flag) check_deps(main_no, sec_no, main_dep_arr, sec_dep_arr);

  #pragma omp parallel shared(x_ptr, main_dep_arr, sec_dep_arr)
  {
    #pragma omp single
    {
    start = omp_get_wtime(); 
    // Host to Main
    for (int i=0; i<main_no; ++i)
    {
      #pragma omp task depend(out:main_dep_arr[i]) firstprivate(i) shared(main_dep_arr)
      {
        // printf("To Main %d\n", main_dep_arr[i].get_dest());
        omp_target_memcpy(
          x_ptr[main_dep_arr[i].get_dest()],    // dst
          x_ptr[main_dep_arr[i].get_orig()],    // src
          size,                                 // length 
          0,                                    // dst_offset
          0,                                    // src_offset, 
          main_dep_arr[i].get_dest(),           // dst_device_num
          main_dep_arr[i].get_orig()            // src_device_num
        );
      }
    }
    for (int j=0; j<n_deps; ++j)
    {
      int orig_id = sec_dep_arr[j].get_oid();
      // Main to Secondary
      #pragma omp task depend(in:main_dep_arr[orig_id]) firstprivate(j) shared(x_ptr, main_dep_arr, sec_dep_arr)
      {
        // printf("Main %d -> Local %d \n", sec_dep_arr[j].get_orig(), sec_dep_arr[j].get_dest());
        omp_target_memcpy(
          x_ptr[sec_dep_arr[j].get_dest()],   // dst
          x_ptr[sec_dep_arr[j].get_orig()],   // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          sec_dep_arr[j].get_dest(),          // dst_device_num
          sec_dep_arr[j].get_orig()           // src_device_num
        );
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

free (main_dep_arr);
free (sec_dep_arr);


//**************************************************//
//        Host to Half -> Half to Secondary         //
//**************************************************//

  if (v_flag) printf("Host to Half -> Half to Secondary\n");
  
  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);

  if (v_flag) print_hval(x_arr, arr_len);

  // Set up Main - Secondary 
  main_no = 4;
  main_dep_arr = (Dep*) malloc(main_no * sizeof(Dep)); 
  
  sec_no = 1;
  n_deps = main_no * sec_no;
  sec_dep_arr = (Dep*) malloc(n_deps * sizeof(Dep));

  // // Set up dependencies
  main_dep_arr[0].set_dep(omp_get_initial_device(), 0);
  main_dep_arr[1].set_dep(omp_get_initial_device(), 1);
  main_dep_arr[2].set_dep(omp_get_initial_device(), 6);
  main_dep_arr[3].set_dep(omp_get_initial_device(), 7);

  sec_dep_arr[0].set_dep(0, 4, 0);
  sec_dep_arr[1].set_dep(1, 5, 1);
  sec_dep_arr[2].set_dep(6, 2, 2);
  sec_dep_arr[3].set_dep(7, 3, 3);

  // Set up dependencies
  // n_id = 0;
  // sec_id = 0;
  // for (int i=0; i<main_no; ++i)
  // {
  //   main_id = n_id;
  //   main_dep_arr[i].set_dep(omp_get_initial_device(), main_id);
  //   n_id++;
  //   for (int j=0; j<sec_no; ++j)
  //   {
  //     sec_dep_arr[sec_id].set_dep(main_id, n_id, i);
  //     n_id++;
  //     sec_id++;
  //   }
  // }

  // Check Dependencies
  if (v_flag) check_deps(main_no, sec_no, main_dep_arr, sec_dep_arr);

  #pragma omp parallel shared(x_ptr, main_dep_arr, sec_dep_arr)
  {
    #pragma omp single
    {
    start = omp_get_wtime(); 
    // Host to Main
    for (int i=0; i<main_no; ++i)
    {
      #pragma omp task depend(out:main_dep_arr[i]) firstprivate(i) shared(main_dep_arr)
      {
        // printf("To Main %d\n", main_dep_arr[i].get_dest());
        omp_target_memcpy(
          x_ptr[main_dep_arr[i].get_dest()],    // dst
          x_ptr[main_dep_arr[i].get_orig()],    // src
          size,                                 // length 
          0,                                    // dst_offset
          0,                                    // src_offset, 
          main_dep_arr[i].get_dest(),           // dst_device_num
          main_dep_arr[i].get_orig()            // src_device_num
        );
      }
    }
    for (int j=0; j<n_deps; ++j)
    {
      int orig_id = sec_dep_arr[j].get_oid();
      // Main to Secondary
      #pragma omp task depend(in:main_dep_arr[orig_id]) firstprivate(j) shared(x_ptr, main_dep_arr, sec_dep_arr)
      {
        // printf("Main %d -> Local %d \n", sec_dep_arr[j].get_orig(), sec_dep_arr[j].get_dest());
        omp_target_memcpy(
          x_ptr[sec_dep_arr[j].get_dest()],   // dst
          x_ptr[sec_dep_arr[j].get_orig()],   // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          sec_dep_arr[j].get_dest(),          // dst_device_num
          sec_dep_arr[j].get_orig()           // src_device_num
        );
      }
    }
     
    #pragma omp taskwait
    end = omp_get_wtime();
    }
  }

  if (v_flag) printf( "Time %f (s)\n", end - start);
  else printf( "%f\n", end - start);

  if (v_flag) verify(x_ptr, arr_len);

free (main_dep_arr);
free (sec_dep_arr);


//**************************************************//
//            Host-to-one -> Linked List            //
//**************************************************//

  if (v_flag) printf("Host-to-one -> Linked List\n");
  
  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
  if (v_flag) print_hval(x_arr, arr_len);

  main_no = 8;
  main_dep_arr = (Dep*) malloc((num_dev) * sizeof(Dep));
  
  sec_no = 0;
  n_deps = 8;
  // sec_dep_arr = (Dep*) malloc(n_deps * sizeof(Dep));

  n_id = 0;
  main_dep_arr[0].set_dep(omp_get_initial_device(), 0);
  n_id++;
  main_dep_arr[n_id].set_dep(n_id-1,  n_id, n_id-1);
  n_id++;
  main_dep_arr[n_id].set_dep(n_id-1,  n_id, n_id-1);
  n_id++;
  main_dep_arr[n_id].set_dep(n_id-1,  n_id, n_id-1);
  n_id++;
  main_dep_arr[n_id].set_dep(n_id-1,  7, n_id-1);
  n_id++;
  main_dep_arr[n_id].set_dep(7,  6, n_id-1);
  n_id++;
  main_dep_arr[n_id].set_dep(6,  5, n_id-1);
  n_id++;
  main_dep_arr[n_id].set_dep(5,  4, n_id-1);

  // Check Dependencies
  if (v_flag) check_deps(main_no, sec_no, main_dep_arr, sec_dep_arr);

  #pragma omp parallel shared(x_ptr, main_dep_arr, sec_dep_arr)
  {
    #pragma omp single
    {
    start = omp_get_wtime(); 
    // Host to Main
    #pragma omp task depend(out:main_dep_arr[0]) shared(main_dep_arr)
    {
      // printf("To Main %d\n", main_dep_arr[i].get_dest());
      omp_target_memcpy(
        x_ptr[main_dep_arr[0].get_dest()],    // dst
        x_ptr[main_dep_arr[0].get_orig()],    // src
        size,                                 // length 
        0,                                    // dst_offset
        0,                                    // src_offset, 
        main_dep_arr[0].get_dest(),           // dst_device_num
        main_dep_arr[0].get_orig()            // src_device_num
      );
    }
    for (int j=1; j<n_deps; ++j)
    {
      int orig_id = main_dep_arr[j].get_oid();
      // Main to Secondary
      #pragma omp task depend(in:main_dep_arr[orig_id]) depend(out:main_dep_arr[j]) firstprivate(j) shared(x_ptr, main_dep_arr)
      {
        // printf("Main %d -> Local %d \n", main_dep_arr[j].get_orig(), main_dep_arr[j].get_dest());
        omp_target_memcpy(
          x_ptr[main_dep_arr[j].get_dest()],   // dst
          x_ptr[main_dep_arr[j].get_orig()],   // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          main_dep_arr[j].get_dest(),          // dst_device_num
          main_dep_arr[j].get_orig()           // src_device_num
        );
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

free (main_dep_arr);
// free (sec_dep_arr);

//**************************************************//
//               Splited Linked List                //
//**************************************************//

  if (v_flag) printf("Splited Linked List\n");

  // Initialize Array Values
  if (v_flag) v_no = init_arr(x_arr, v_no, arr_len);
  
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

// PROG=broadcast_21; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -L/soft/compilers/cuda/cuda-11.0.2/lib64 -L/soft/compilers/cuda/cuda-11.0.2/targets/x86_64-linux/lib/ -I/soft/compilers/cuda/cuda-11.0.2/include -ldl -lcudart -pthread $PROG.cpp
// ./broadcast_21.x 
// ./broadcast_21.x>out_test.o  2>&1
// nsys profile -o test_prof --stats=true ./broadcast_21.x
// nsys profile -o broadcast21_100M ./broadcast_21.x

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


  // main_dep_arr.push_back(new Dep(omp_get_initial_device(), 0));
  // main_dep_arr.push_back(new Dep(omp_get_initial_device(), 4));

    // new(main_dep_arr +i) Dep(omp_get_initial_device(), main_id);
      // new(sec_dep_arr +i) Dep(main_id, n_id, i);


// void print_deps(Dep * deps)
// {
//   // Check Dependencies
//   printf("Dependencies check \n");
//   for (auto deps_it = deps.begin(); deps_it != deps.end(); ++deps_it)
//     printf("Main %d -> Secondary %d \n", (*deps_it)->get_main(), (*deps_it)->get_sec());
// }

  // for (int i=1; i<=sec_no; ++i)
  // {
  //   deps.push_back(new Dep(0, i, 0)); // Origin , Dest, origin_id
  //   deps.push_back(new Dep(4, i+4, 1)); // Origin , Dest, origin_id
  // }

  /*
  int* main_arr = (int*) malloc(main_no * sizeof(int)); 
  main_arr[0] = 0;
  main_arr[1] = 4;
  
  int* sec_dep_arr = (int*) malloc(n_deps * 2 * sizeof(int));

  
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


  free (main_arr);
  free (sec_dep_arr);
*/