//**************************************************//
//            Host-to-one -> One-to-all             //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+20;

  print_hval(x_arr);

  printf("Host-to-one -> One-to-all\n");
  #pragma omp parallel num_threads(omp_get_num_devices()) shared(x_ptr)
  {
    #pragma omp single
    {
      int dependency;
      #pragma omp task depend(out:dependency)
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
        #pragma omp task depend(in:dependency) firstprivate(i)
          omp_target_memcpy(
            x_ptr[i],                           // dst
            x_ptr[0],                           // src
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
            i,                                  // dst_device_num
            0                                   // src_device_num
          );
    }
  }
  #pragma omp taskwait
  verify(x_ptr);

//**************************************************//
//            Host-to-one -> Binary tree            //
//**************************************************//
  
  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+30;
  
  print_hval(x_arr);

  printf("Host-to-one -> Binary tree\n");
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
        #pragma omp task depend(in:dep_arr[(i-1)/2]) depend(out:dep_arr[i]) firstprivate(i)
          omp_target_memcpy(
            x_ptr[i],                           // dst
            x_ptr[(i-1)/2],                     // src
            size,                               // length 
            0,                                  // dst_offset
            0,                                  // src_offset, 
            i,                                  // dst_device_num
            (i-1)/2                             // src_device_num
          );
    }
  }
  #pragma omp taskwait
  verify(x_ptr);

//**************************************************//
//               Host-to-Binary tree                //
//**************************************************//

  for (int i=0; i<arr_len; ++i)
    x_arr[i]=i+40;
  
  print_hval(x_arr);

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
