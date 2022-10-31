#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <omp.h>
#define N 1000

// Test: set tasks to specific core/socket

int main() {
 
  unsigned cpu, node;
  int *x, *y;
  // omp_memspace_handle_t  xy_memspace = omp_default_mem_space;
  // omp_alloctrait_t       xy_traits[1]={omp_atk_alignment, 64};
  // omp_allocator_handle_t xy_alloc    = omp_init_allocator(xy_memspace,1,xy_traits);
  omp_memspace_handle_t  x_memspace = omp_default_mem_space;
  omp_alloctrait_t       x_traits[1]={omp_atk_alignment, 64};
  omp_allocator_handle_t x_alloc    = omp_init_allocator(x_memspace,1,x_traits);
  
  omp_memspace_handle_t  y_memspace = omp_default_mem_space;
  omp_alloctrait_t       y_traits[1]={omp_atk_alignment, 64};
  omp_allocator_handle_t y_alloc    = omp_init_allocator(y_memspace,1,y_traits);

 
  // Get current CPU core and NUMA node via system call
  // Note this has no glibc wrapper so we must call it directly
  syscall(SYS_getcpu, &cpu, &node, NULL);

  // Display information
  printf("Main: CPU core %u NUMA node %u Thread %d\n\n", cpu, node, omp_get_thread_num());

  #pragma omp parallel num_threads(2) proc_bind(spread)
  {
    // printf("CPU core %.2u NUMA node %u Thread %d Task 10\n", cpu, node, omp_get_thread_num());
      
    if (omp_get_thread_num() == 0){
      syscall(SYS_getcpu, &cpu, &node, NULL);
      printf("Allocating: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
      x=(int *)omp_alloc(N*sizeof(int), x_alloc);
      if(((intptr_t)(x))%64 != 0 ) { printf("ERROR: x not 64-Byte aligned\n"); exit(1); }
    }

    if (omp_get_thread_num() == 1){
      syscall(SYS_getcpu, &cpu, &node, NULL);
      printf("Allocating: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
      y=(int *)omp_alloc(N*sizeof(int), y_alloc);
      if( ((intptr_t)(y))%64 != 0) { printf("ERROR: y not 64-Byte aligned\n"); exit(1); }
    }
  }

  #pragma omp parallel shared(x, y)
  {
    #pragma omp single
    {
      int dep_0, dep_1, max_tasks = 5;
      #pragma omp task depend(out:dep_0) affinity(x) shared(x)
      {
        syscall(SYS_getcpu, &cpu, &node, NULL);
        printf("Main: CPU core %.2u NUMA node %u Thread %d Task 10\n", cpu, node, omp_get_thread_num());
      }
      #pragma omp task depend(out:dep_1) affinity(y) shared(y)
      {
        syscall(SYS_getcpu, &cpu, &node, NULL);
        printf("\tMain: CPU core %.2u NUMA node %u Thread %d Task 20\n", cpu, node, omp_get_thread_num());
      }
      for(int i=0; i<max_tasks; ++i){
        #pragma omp task depend(in:dep_0) affinity(x) shared(x)
        {
          syscall(SYS_getcpu, &cpu, &node, NULL);
          printf("Subt: CPU core %.2u NUMA node %u Thread %d Task %d \n", cpu, node, omp_get_thread_num(), i+11);
        }
      }
      for(int i=0; i<max_tasks; ++i){
        #pragma omp task depend(in:dep_1) affinity(y) shared(y)
        {
          syscall(SYS_getcpu, &cpu, &node, NULL);
          printf("\tSubt: CPU core %.2u NUMA node %u Thread %d Task %d \n", cpu, node, omp_get_thread_num(), i+21);
        }
      }
      #pragma omp taskwait
    }
  }
 
  return 0;
 
}

// OMP_DISPLAY_ENV=VERBOSE OMP_PROC_BIND=spread OPM_PLACES=sockets ./test_13.x