#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <omp.h>
#define N 1000

// Test: set tasks to specific core/socket

int main() {
 
  unsigned cpu, node;

  // Get current CPU core and NUMA node via system call
  // Note this has no glibc wrapper so we must call it directly
  syscall(SYS_getcpu, &cpu, &node, NULL);

  // Display information
  printf("Main: CPU core %u NUMA node %u Thread %d\n\n", cpu, node, omp_get_thread_num());

  #pragma omp parallel num_threads(3) 
  {
    syscall(SYS_getcpu, &cpu, &node, NULL);
    switch(cpu){
      case 0:
        printf("Case 0: CPU core %.2u NUMA node %u Thread %d Task 10\n", cpu, node, omp_get_thread_num());
      break;
      case 32:
        printf("Case 1: CPU core %.2u NUMA node %u Thread %d Task 10\n", cpu, node, omp_get_thread_num());
      break;
      case 33:
        printf("Case 2: CPU core %.2u NUMA node %u Thread %d Task 10\n", cpu, node, omp_get_thread_num());
      break;
      default:
        printf("Case X: CPU core %.2u NUMA node %u Thread %d Task 10\n", cpu, node, omp_get_thread_num());
    }
  }
  return 0;
}

// clang++ -fopenmp test_16.cpp -o test_16.x
// OMP_DISPLAY_ENV=VERBOSE OMP_PROC_BIND=spread OPM_PLACES=sockets ./test_16.x
// OMP_PROC_BIND=true OMP_DYNAMIC=true OMP_MAX_ACTIVE_LEVELS=2 OMP_DISPLAY_ENV=VERBOSE ./test_16.x 

// OMP_PROC_BIND=true OMP_DYNAMIC=true OMP_MAX_ACTIVE_LEVELS=2 ./test_16.x 
// OMP_PROC_BIND=true OMP_DYNAMIC=true OPM_PLACES=sockets ./test_16.x 
// OMP_PROC_BIND=true OPM_PLACES=sockets ./test_16.x 
// OMP_PROC_BIND=true OMP_DYNAMIC=true OPM_PLACES={0,32,33} OMP_DISPLAY_ENV=VERBOSE ./test_16.x 
// taskset -c 0,32,33 ./test_16.x