#include <stdio.h>
#include <omp.h>
#include <cstdlib>
#include <cuda_runtime.h>

int main()
{
  int num_dev = omp_get_num_devices();    
  cudaError_t c_error;
  // Check PeerToPeer
  int can_access = 0;
  for(int dev_a = 0; dev_a < num_dev; ++dev_a){
    for(int dev_b = 0; dev_b < num_dev; ++dev_b){
      if (dev_a != dev_b){
        cudaSetDevice(dev_a);
        c_error = cudaDeviceEnablePeerAccess(dev_b, 0);
          if(c_error){
            printf("Dev[%d]->Dev[%d] Fail: Error(%d): %s\n", dev_a, dev_b, c_error, cudaGetErrorString(c_error));
          }else{
            printf("Dev[%d]->Dev[%d] Enabled Err(%d)\n", dev_a, dev_b, c_error);
        }
      }
    }
  }
  return 0;
}

// PROG=test_nvlink; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -L/soft/compilers/cuda/cuda-11.0.2/lib64 -L/soft/compilers/cuda/cuda-11.0.2/targets/x86_64-linux/lib/ -I/soft/compilers/cuda/cuda-11.0.2/include -ldl -lcudart -pthread $PROG.cpp
// ./test_nvlink.x 
// nvprof --print-gpu-trace ./test_nvlink.x 
// PROG=test_nvlink; nvcc -o $PROG.x $PROG.cpp
// nvidia-smi topo -m