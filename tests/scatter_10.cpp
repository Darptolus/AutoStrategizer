#include <stdio.h>
#include <omp.h>
#include <cstdlib>
#include <math.h>

int num_dev = omp_get_num_devices();
int chnk_len = 2;
int arr_len = chnk_len*num_dev;

void verify(void ** x_ptr){
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

void print_hval(int * x_arr){
  printf("Host -> Value of X: ");
  for (int j=0; j<arr_len; ++j)
    printf("%d ", x_arr[j]);
  printf("\n");
}

int get_desc(int id, int n_nodes){
  int n_desc = 0;
  if(id*2+1 > n_nodes || id*2+2 > n_nodes){
    return n_desc;
  }else{
    if(id*2+1 < n_nodes){
      n_desc++;
      n_desc = n_desc + get_desc(id*2+1, n_nodes);
    }  
    if(id*2+2 < n_nodes){
      n_desc++;
      n_desc = n_desc + get_desc(id*2+2, n_nodes);
    }
    return n_desc;
  }
}

int main()
{  
  int* x_arr = (int*) malloc(arr_len * sizeof(int)); 
  
  // Set device pointers
  void ** x_ptr = (void **) malloc(sizeof(void**) * num_dev+1); 
  if (!x_ptr) {
    printf("Memory Allocation Failed\n");
    exit(1);
  }

  size_t a_size = sizeof(int) * arr_len;
  size_t c_size = sizeof(int) * chnk_len;
  size_t * ct_size = (size_t*) malloc(num_dev * 2 * sizeof(size_t)); // Chunk size and offsets

  // Add host pointer 
  x_ptr[num_dev]=&x_arr[0];

  printf("[Broadcast Int Array]\n");
  printf("No. of Devices: %d\n", num_dev);
  
//**************************************************//
//                   Host-to-all                    //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+10;
  
  print_hval(x_arr);

  // Allocate device memory -> Allocating different sizes for each device based on the strategy
  // ToDo: Consider arrays remainder 
  for (int dev = 0; dev < num_dev; ++dev){
    x_ptr[dev] = omp_target_alloc(c_size, dev);
  }

  printf("Host-to-all\n");
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
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
  }

  // verify(x_ptr);
  
  for (int dev = 0; dev < num_dev; ++dev)
    omp_target_free(x_ptr[dev], dev);

//**************************************************//
//            Host-to-one -> One-to-all             //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+20;
  
  print_hval(x_arr);

  // Allocate device memory -> Allocating different sizes for each device based on the strategy
  // ToDo: Consider arrays remainder 
  x_ptr[0] = omp_target_alloc(a_size, 0);
  
  for (int dev = 1; dev < num_dev; ++dev)
    x_ptr[dev] = omp_target_alloc(c_size, dev);

  printf("Host-to-one -> One-to-all\n");
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dependency;
      #pragma omp task depend(out:dependency)
        omp_target_memcpy(
          x_ptr[0],                             // dst
          x_ptr[omp_get_initial_device()],      // src
          a_size,                               // length 
          0,                                    // dst_offset
          0,                                    // src_offset, 
          0,                                    // dst_device_num
          omp_get_initial_device()              // src_device_num
        );
      for(int i=1; i<num_dev; ++i)
        #pragma omp task depend(in:dependency) firstprivate(i)
          omp_target_memcpy(
            x_ptr[i],                           // dst
            x_ptr[0],                           // src
            c_size,                             // length 
            0,                                  // dst_offset
            c_size*i,                           // src_offset, 
            i,                                  // dst_device_num
            0                                   // src_device_num
          );
    }
  }
  #pragma omp taskwait
  // verify(x_ptr);

  for (int dev = 0; dev < num_dev; ++dev)
    omp_target_free(x_ptr[dev], dev);

//**************************************************//
//            Host-to-one -> Binary tree            //
//**************************************************//

  int n_desc = 0;

  for (int i=0; i<arr_len; ++i){
    x_arr[i]=i+30;
  }
  
  print_hval(x_arr);
  
  // Calculate number of descendants per node 

  // Allocate device memory -> Allocating different sizes for each device based on the strategy
  // ToDo: Consider arrays remainder
  // ToDo: Define size for each device and offset
  // printf("Array size is: %zu \n", a_size);
  for (int dev = 0; dev < num_dev; ++dev){
    n_desc = get_desc(dev, num_dev);
    ct_size[dev] = (n_desc+1) * c_size;
    if(dev ==0 || dev%2 == 1){
      ct_size[dev+num_dev]=0;
    }else{
      ct_size[dev+num_dev]=c_size*(get_desc(dev-1, num_dev)+1);
    }
    printf("Device %d -> has %d Descendants -> size is %zu, offset is %zu \n", dev, n_desc, ct_size[dev], ct_size[dev+num_dev]);
  }

  printf("Host-to-one -> Binary tree\n");
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr, ct_size)
  {
    #pragma omp single
    {
      int dep_arr[num_dev];
      #pragma omp task depend(out:dep_arr[0])
        omp_target_memcpy(
          x_ptr[0],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          a_size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          0,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      for(int i=1; i<num_dev; ++i)
        #pragma omp task depend(in:dep_arr[(i-1)/2]) depend(out:dep_arr[i]) firstprivate(i)
        {
          omp_target_memcpy(
            x_ptr[i],                         // dst
            x_ptr[(i-1)/2],                   // src
            ct_size[i],                       // length 
            0,                                // dst_offset
            ct_size[i+num_dev],               // src_offset, 
            i,                                // dst_device_num
            (i-1)/2                           // src_device_num
          );
        }
    }
  }
  #pragma omp taskwait
  verify(x_ptr);


//**************************************************//
//               Host-to-Binary tree                //
//**************************************************//
 /* printf(">>>>> >>>>> Here <<<<< <<<<<\n");
  for (int i=0; i<arr_len; ++i){
    x_arr[i]=i+40;
  }
  
  printf("Host -> Value of X: ");

  for (int j=0; j<arr_len; ++j){
    printf("%d ", x_arr[j]);
  }
  printf("\n");

  // Allocate device pointer
  // ToDo: Consider arrays remainder 
  // Allocating different sizes for each device based on the strategy?
  for (int dev = 0; dev < num_dev; ++dev){
    x_ptr[dev] = omp_target_alloc(size, dev);
  }

  printf("Host-to-Binary tree\n");
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
      #pragma omp task depend(out:dep_arr[1])
        omp_target_memcpy(
          x_ptr[1],                           // dst
          x_ptr[omp_get_initial_device()],    // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          1,                                  // dst_device_num
          omp_get_initial_device()            // src_device_num
        );
      for(int i=2; i<num_dev; ++i)
        #pragma omp task depend(in:dep_arr[(i-1)/2]) depend(out:dep_arr[i]) firstprivate(i)
        omp_target_memcpy(
          x_ptr[i],                           // dst
          x_ptr[(i-1)/2],                           // src
          size,                               // length 
          0,                                  // dst_offset
          0,                                  // src_offset, 
          i,                                  // dst_device_num
          (i-1)/2                                   // src_device_num
        );
    }
  }
  #pragma omp taskwait
  verify(x_ptr);

*/
//**************************************************//
//            Host-to-one -> Linked List            //
//**************************************************//
 /* printf(">>>>> >>>>> Here <<<<< <<<<<\n"); 
  for (int i=0; i<arr_len; ++i){
    x_arr[i]=i+50;
  }
  
  printf("Host -> Value of X: ");

  for (int j=0; j<arr_len; ++j){
    printf("%d ", x_arr[j]);
  }
  printf("\n");

  // Allocate device pointer
  // ToDo: Consider arrays remainder 
  // Allocating different sizes for each device based on the strategy?
  for (int dev = 0; dev < num_dev; ++dev){
    x_ptr[dev] = omp_target_alloc(size, dev);
  }

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
  for (int i=0; i<arr_len; ++i){
    x_arr[i]=i+60;
  }
  
  printf("Host -> Value of X: ");

  for (int j=0; j<arr_len; ++j){
    printf("%d ", x_arr[j]);
  }
  printf("\n");

  // Allocate device pointer
  // ToDo: Consider arrays remainder 
  // Allocating different sizes for each device based on the strategy?
  for (int dev = 0; dev < num_dev; ++dev){
    x_ptr[dev] = omp_target_alloc(size, dev);
  }

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
*/
  free(x_ptr);


  return 0;
}
