#include "autoStrategizer.hpp"

int main() {

  // Set Architecture
  // AutoStrategizer::AutoStrategizer my_AutoS("topo_dgx");
  AutoStrategizer::AutoStrategizer my_AutoS("topo_smx");

  // Print topology
  my_AutoS.printTopo(AutoStrategizer::CLI);
   
  void ** mem_ptr;
  mem_ptr = my_AutoS.get_memptr();

  // Define Operation
  AutoStrategizer::CollectiveOperation my_CoP;

  my_CoP.add_origin(0);
  my_CoP.add_destination(3);
  my_CoP.set_size(60);
  my_CoP.set_coop(AutoStrategizer::D2D);
  my_CoP.set_mhtd(AutoStrategizer::P2P); // Peer to Peer
  // my_CoP.set_mhtd(AutoStrategizer::MXF); // Max Flow
  // my_CoP.set_mhtd(AutoStrategizer::DVT); // Distant Vector

  my_AutoS.addCO(&my_CoP);

  // my_AutoS.printTopo_cpy(AutoStrategizer::CLI);

  my_AutoS.printCO(AutoStrategizer::CLI);

  
  // Malloc hosts and targets
  my_AutoS.auto_malloc();

  // Initialize Origin


  // Get dependencies
  for (auto& a_dep : *(my_AutoS.getDeps()))
  {
    printf("[EXEC:] Orig: %d Dest: %d Size: %zu O_Offs: %zu D_Offs: %zu, Path No.: %d\n", a_dep->orig, a_dep->dest, a_dep->size, a_dep->of_s, a_dep->of_d, a_dep->ipth);
  }

  // Executing

  double start, end;
  int n_deps;
  n_deps = my_AutoS.getDeps()->size();

  #pragma omp parallel shared(mem_ptr) num_threads(n_deps) 
  {
    #pragma omp single
    {
      start = omp_get_wtime();
    // }

      for (auto& a_dep : *(my_AutoS.getDeps()))
      {
        if (a_dep->deps == 0){
          #pragma omp task depend(out:mem_ptr[a_dep->dest]) shared(mem_ptr)
          {
            // printf("Thread = %d\n", omp_get_thread_num());
            // printf("host_id: %d\n", omp_get_initial_device());
            printf("[EXEC:] Orig: %d (ID: %d) - Dest: %d (ID: %d) - Size: %zu O_Offs: %zu D_Offs: %zu, Path No.: %d - Thread = %d\n", a_dep->orig, a_dep->o_id, a_dep->dest, a_dep->d_id, a_dep->size, a_dep->of_s, a_dep->of_d, a_dep->ipth, omp_get_thread_num());
            omp_target_memcpy
            (
              mem_ptr[a_dep->dest],                     // dst
              mem_ptr[a_dep->orig],                     // src
              a_dep->size * sizeof(int),                // length 
              a_dep->of_d * sizeof(int),                // dst_offset
              a_dep->of_s * sizeof(int),                // src_offset, 
              a_dep->d_id,                              // dst_device_num
              a_dep->o_id                               // src_device_num
            );
          }
        }else{
          #pragma omp task depend(in:mem_ptr[a_dep->orig]) depend(out:mem_ptr[a_dep->dest]) shared(mem_ptr)
          {
            // printf("Thread = %d\n", omp_get_thread_num());
            printf("[EXEC2:] Orig: %d (ID: %d) - Dest: %d (ID: %d) - Size: %zu O_Offs: %zu D_Offs: %zu, Path No.: %d - Thread = %d\n", a_dep->orig, a_dep->o_id, a_dep->dest, a_dep->d_id, a_dep->size, a_dep->of_s, a_dep->of_d, a_dep->ipth, omp_get_thread_num());
            omp_target_memcpy
            (
              mem_ptr[a_dep->dest],                     // dst
              mem_ptr[a_dep->orig],                     // src
              a_dep->size * sizeof(int),                // length 
              a_dep->of_d * sizeof(int),                // dst_offset
              a_dep->of_s * sizeof(int),                // src_offset, 
              a_dep->d_id,                              // dst_device_num
              a_dep->o_id                               // src_device_num
            );
          }
        }
      }
    #pragma omp taskwait
    // #pragma omp barrier
    //   #pragma omp single
    //   {
    end = omp_get_wtime();
    }
  #pragma omp barrier
  }


  // Free memory hosts and targets
  my_AutoS.auto_mfree();


  return 0;
}


// Compile

// clang++ -o main_auto.o main_auto.cpp autoStrategizer.cpp -fopenmp -lnuma -fopenmp-targets=nvptx64 --cuda-gpu-arch=sm_70

// Run

// ./main_auto.o 