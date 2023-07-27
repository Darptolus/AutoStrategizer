#include <iostream>
#include <fstream>
#include <string>
#include<bits/stdc++.h>
#include <vector>
#include <numa.h>
#include <omp.h>
#include <chrono>

// #include <sstream>

namespace autostrat
{
  using namespace std;

  const bool v_flag = true; // Print verifications True=On False=Off

  struct numa_dom
  {
    int h_i[4];
    int conf = 0;
  };

  class arch
  {
    private:
      int n_dev; // Number of devices
      int n_net; // Number of Networks
      int n_hst; // Number of hosts
      int n_nod; // Total number of devices
      int n_nma; // Number of numa domains
      numa_dom *numa_d;
      // int **hst_mx;
      int **dev_mx; // Interconnectivity Matrix
    public:
      arch(){
        n_nma = numa_num_configured_nodes();
        n_hst = n_nma; // Single Node
        numa_d = (numa_dom*) malloc(n_nma * sizeof(numa_dom));
        // Initialize numa domains
        for (int n_dom = 0; n_dom < n_nma; ++n_dom)
          numa_d[n_dom].conf = 0;
        // Fill Host to Host information
      };
      ~arch(){
        for (int dev = 0; dev < (n_nod); ++dev)
          free(dev_mx[dev]);
        free(dev_mx);
        free(numa_d);
        free(node_id);
      };
      int * node_id;
      int get_nhst(){return n_hst;}
      int get_ndev(){return n_dev;}
      int get_nnod(){return n_nod;}
      int get_nnuma(){return n_nma;}
      int get_nnet(){return n_net;}
      int ** get_devmx(){return dev_mx;}
      numa_dom * get_numa(){return numa_d;}
      void set_ndev(int a_ndev){
        n_dev = a_ndev;
        n_nod = n_dev + n_hst;
        node_id = (int*) malloc(n_nod * sizeof(int));
        // Allocate memory for Interconnectivity Matrix 
        dev_mx = (int**) malloc((n_nod) * sizeof(int*));
        // printf("host_id: %d\n", omp_get_initial_device());
        for (int dev = 0; dev < (n_nod); ++dev)
          {
            dev_mx[dev] = (int*)malloc((n_nod) * sizeof(int));
            // Set device ID
            if (dev < n_hst)
              node_id[dev]=omp_get_initial_device();
            else
              node_id[dev]=dev-n_hst;
            // printf("Dev: %d ID: %d\n", dev, node_id[dev]);
          }
      }
      void set_nhst(int a_nhst){n_hst = a_nhst;}
      void set_nnet(int a_nnet){n_net = a_nnet;}
  };


  // Communication pattern dependencies
  class cops_dep
  {
    private:
      cops_dep * p_dep;
    public:
      // cops_dep(int a_deps, int a_done, int a_orig, int a_dest, int a_size, int a_oof, int a_dof):
      // deps(a_deps), done(a_done), orig(a_orig), dest(a_dest), size(a_size), of_s(a_oof), of_d(a_dof){};
      cops_dep(int a_deps, int a_done, int a_orig, int a_dest, int a_oid, int a_did, int a_size, int a_oof, int a_dof, int a_ipth):
      deps(a_deps), done(a_done), orig(a_orig), dest(a_dest), o_id(a_oid), d_id(a_did), size(a_size), of_s(a_oof), of_d(a_dof), ipth(a_ipth){};
      ~cops_dep(){};
      // Todo: Add set/get methods
      int deps;
      int done;
      int orig;
      int dest;
      int o_id;
      int d_id;
      size_t size;
      size_t of_s;     // Origin Offset
      size_t of_d;     // Destination Offset
      int ipth;
  } *op_deps; // Remove

  // All deps
  typedef std::vector<cops_dep*> ve_deps;

  ve_deps all_deps;
  
  // Collective operation definition
  enum c_ops { D2D, BRC, SCT, GAT, RED };
  enum h_mth { P2P, DVT, MXF }; // Host-to-Device, Distant Vector, Max-Flow

  class cops_def
  {
    private:
      std::vector<int> o_orig; // Origin devices (node_num)
      std::vector<int> o_dest; // Destination devides (node_num)
      int size; // ToDo: Change to size_t
      c_ops cop; // Collective operation
      h_mth mht; // Heuristic Method
      int n_orig; // Number of origin devices
      int n_dest; // Number of destination devices
    public:
      cops_def(){};
      ~cops_def(){};
      void add_orig(int a_orig){
        o_orig.push_back(a_orig);
      };
      void add_dest(int a_dest){
        o_dest.push_back(a_dest);
      };
      void set_size(int a_size){size = a_size;};
      int get_size(){return size;};
      void set_coop(c_ops a_cop){cop = a_cop;};
      void set_mhtd(h_mth a_mht){mht = a_mht;};
      int get_norig(){return o_orig.size();};
      int get_ndest(){return o_dest.size();};
      c_ops get_coop(){return cop;};
      h_mth get_mhtd(){return mht;};
      std::vector<int>* get_orig(){return &o_orig;};
      std::vector<int>* get_dest(){return &o_dest;};
  } *def_ops; // Remove

  // All Ops
  typedef std::vector<cops_def*> ve_ops;

  ve_ops all_ops;

  struct mem_info
  {
    int node_id;
    size_t size;
  };

  typedef std::vector<mem_info*> ve_meminfo;

  ve_meminfo all_meminfo;

  int get_topo(ifstream *arch_f, arch * t_arch);
  void copy_mx(arch * a_arch, int ***dev_mx_cpy_p, int new_cpy);
  void del_mx(arch * a_arch, int ***dev_mx_cpy_p);
  void print_mx(arch * a_arch);
  void print_numa(arch *a_arch);
  int get_ops(ifstream *ops_f, ve_ops *all_ops, arch * a_arch);
  void print_ops(ve_ops *all_ops);
  void ops_deps(arch * a_arch, ve_ops *all_ops, ve_deps *all_deps, int ***dev_mx_cpy_p, ve_meminfo *all_meminfo);
  void print_cpat(ve_deps *all_deps);
  void auto_malloc(arch * a_arch, ve_meminfo *all_meminfo, void ** mem_ptr);
  // void exec_op(ve_deps *all_deps, void *** mem_ptr);
  void auto_mfree(arch * a_arch, ve_meminfo *all_meminfo, void ** mem_ptr);
}

int main()
{
  using namespace autostrat;
  int dev_a = 0, dev_b = 0, dev_i, dev_ii, n_nodes = 1, arr_len;
  int f_sts;
  int **dev_mx_cpy; //**hst_mx, **dev_mx, 
  void **mem_ptr;
  // int *o_arr;

  double start, end;
  // unsigned cpu, node;


  ifstream c_ops ("test_ops");

  // numa_d = (numa_dom*) malloc((n_nma) * sizeof(numa_dom));
  
  arch t_arch;
  // Get Topology

  ifstream t_dgx ("topo_dgx");
  if (v_flag) printf("DGX\n"); // ToDo:  
  if (get_topo(&t_dgx, &t_arch)) printf("Unable to get topology\n");
  
  // ifstream t_smx ("topo_smx");
  // if (v_flag) printf("SMX\n"); // ToDo:
  // if (get_topo(&t_smx, &t_arch)) printf("Unable to get topology\n");
  
  if (v_flag) cout << "no. Devices: " << t_arch.get_ndev() << "\n";
  if (v_flag) cout << "no. Networks: " << t_arch.get_nnet() << "\n";

  // Print Interconnectivity Matrix
  if (v_flag) print_mx(&t_arch);
  // Print Numa Cores
  if (v_flag) print_numa(&t_arch);

  // print_mx(&dev_mx_cpy, n_dev, n_hst); // Print Copy
  
  // Inputs
  // Set up origin, dest, size, operation and method
  if (get_ops(&c_ops, &all_ops, &t_arch)) printf("Unable get ops\n");

  if (v_flag) print_ops(&all_ops);

  const auto start_time = std::chrono::steady_clock::now();
  ops_deps(&t_arch, &all_ops, &all_deps, &dev_mx_cpy, &all_meminfo);
  const auto end_time = std::chrono::steady_clock::now();

  const std::chrono::duration<double> elapsed_seconds = end_time - start_time;

  std::cout << elapsed_seconds.count() << '\n'; // C++20: operator<< chrono::duration

  // Outputs
  if (v_flag) print_cpat(&all_deps);

//**************************************************//
// Memory allocation
//**************************************************//

  mem_ptr = (void **) malloc(sizeof(void**) * (t_arch.get_nhst() + t_arch.get_ndev()));
  auto_malloc(&t_arch, &all_meminfo, mem_ptr);
  
  // mem_ptr[0] = (int*) malloc(240); 
  // mem_ptr[3] = omp_target_alloc(240, 1);


//**************************************************//
// Array initialization (copy to device(s) if necesary)
//**************************************************//

  // ToDo: 
  def_ops = all_ops.front();
  if (def_ops->get_norig() > 1){
    if (v_flag) printf("Multi Origin \n");
  }
  else
  {
    int node_num = (def_ops->get_orig())->front();
    int dev_id = t_arch.node_id[node_num];
    int a_len = all_meminfo[0]->size/sizeof(int);
  //   o_arr = (int*)mem_ptr[all_meminfo[dev_o]->node_id];

  //   for (int i=0; i<all_meminfo[dev_o]->size/sizeof(int); ++i)
  //     o_arr[i]=i; //+v_no;
    if (v_flag) printf("Single Origin Node: %d Dev: %d\n", node_num, dev_id);


    #pragma omp target device(dev_id) map(mem_ptr[node_num], a_len) 
    {
      int* a_val = (int*)mem_ptr[node_num];
      printf("Initializing %s No. %d ArrX", omp_is_initial_device()?"Host":"Device", omp_get_device_num());
      for (int j=0; j<a_len; ++j){
        *(a_val+j) = j;
      }
      printf("\n");
      // printf("N_ID: %d, %d\n", n_id, omp_get_device_num());
    }

  // printf("Host -> Value of X: ");
    // for (int j=0; j<(all_meminfo[dev_o]->size/sizeof(int)); ++j)
    //   printf("%d ", o_arr[j]);
    // printf("\n");

    #pragma omp target device(dev_id) map(mem_ptr[node_num], a_len) 
    {
      int* a_val = (int*)mem_ptr[node_num];
      printf("%s ID: %d ArrX = ", omp_is_initial_device()?"Host":"Device", omp_get_device_num());
      for (int j=0; j<a_len; ++j){
        printf("%d ", *(a_val+j));
      }
      printf("\n");
      // printf("N_ID: %d, %d\n", n_id, omp_get_device_num());
    }

  }





//**************************************************//
// Execute
//************************************************`*//

  // exec_op(&all_deps, &mem_ptr);

  printf("[EXEC:] Execution:\n");
  
  // for (auto& a_dep : all_deps)
  // {
  //   printf("[EXEC:] Orig: %d Dest: %d Size: %zu O_Offs: %zu D_Offs: %zu, Path No.: %d\n", a_dep->orig, a_dep->dest, a_dep->size, a_dep->of_s, a_dep->of_d, a_dep->ipth);
  // }

#pragma omp parallel shared(mem_ptr) num_threads(all_deps.size()) 
  {
    #pragma omp single
    {
      start = omp_get_wtime();
    // }

      for (auto& a_dep : all_deps)
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

//**************************************************//
// Verify
//**************************************************//
// for (auto& a_dep : all_deps)
// {
//   printf("[VAL:]Orig: %d (ID: %d) Dest: %d (ID: %d) - Size: %zu O_Offs: %zu D_Offs: %zu, Path No.: %d\n", a_dep->orig, a_dep->o_id, a_dep->dest, a_dep->d_id, a_dep->size, a_dep->of_s, a_dep->of_d, a_dep->ipth);
//   arr_len = a_dep->size;
//   int dest = a_dep->dest;
//   int d_id = a_dep->d_id;          
//   #pragma omp target device(d_id) map(mem_ptr[dest], arr_len) 
//   {
//     int* x = (int*)mem_ptr[dest];
//     printf("%s No. %d ArrX = ", omp_is_initial_device()?"Host":"Device", omp_get_device_num());
//     for (int j=0; j<arr_len; ++j){
//       printf("%d ", *(x+j));
//     }
//     printf("\n");
//   }
// }


for (auto& a_mem : all_meminfo)
{
  // printf("[VAL:]Orig: %d (ID: %d) Dest: %d (ID: %d) - Size: %zu O_Offs: %zu D_Offs: %zu, Path No.: %d\n", a_dep->orig, a_dep->o_id, a_dep->dest, a_dep->d_id, a_dep->size, a_dep->of_s, a_dep->of_d, a_dep->ipth);
  
  arr_len = a_mem->size / sizeof(int);
  int n_id = t_arch.node_id[a_mem->node_id];
  int d_id = a_mem->node_id;
  printf("[VAL:] Node: %d Dev_ID: %d Size: %zu\n", a_mem->node_id, n_id, a_mem->size);

  #pragma omp target device(n_id) map(mem_ptr[d_id], arr_len) 
  {
    int* x = (int*)mem_ptr[d_id];
    printf("%s ID: %d ArrX = ", omp_is_initial_device()?"Host":"Device", omp_get_device_num());
    for (int j=0; j<arr_len; ++j){
      printf("%d ", *(x+j));
    }
    printf("\n");
    // printf("N_ID: %d, %d\n", n_id, omp_get_device_num());
  }
}

// arr_len = 60;
// #pragma omp target device(1) map(mem_ptr[3], arr_len) 
//   {
//     int* x = (int*)mem_ptr[3];
//     printf("%s No. %d ArrX = ", omp_is_initial_device()?"Host":"Device", omp_get_device_num());
//     for (int j=0; j<arr_len; ++j){
//       printf("%d ", *(x+j));
//     }
//     printf("\n");
//   }
//**************************************************//
// Memory Free
//**************************************************//

  // printf("\n");
  auto_mfree(&t_arch, &all_meminfo, mem_ptr);
  // free(mem_ptr[0]); 
  // omp_target_free(mem_ptr[3], 1);

 
  free(mem_ptr);
  // free(o_arr);

  del_mx(&t_arch, &dev_mx_cpy);

  for (auto& element : all_deps)
    delete element;

  for (auto& element : all_meminfo)
    delete element;

  for (auto& element : all_ops)
    delete element;

  return 0;
} // End of Main

int autostrat::get_topo(ifstream *arch_f, arch * a_arch)
{
  int line_n = 0, word_n = 0;
  int n_dev_v = 0, n_net_v = 0, dev_a = 0, dev_b =0, n_core = 0, nma_x = 0;
  string line, word, word_b;
  stringstream iss_a, iss_b;
  numa_dom numa_t; // Temporary Numa for comparison 

  if (arch_f->is_open())
  {
    while (!arch_f->eof())
    {
      getline (*arch_f, line);
      iss_a.clear();
      iss_a << line;
      word_n = 0;
      
      // Get all words in a line
      while (iss_a >> word)
      {
        if (line_n == 0)
        {
          if (word.compare(0,3,"GPU") == 0) ++n_dev_v;
          else if (word.compare(0,3,"mlx") == 0) ++n_net_v;
        }
        else if (word_n == 0 && word.compare(0,3,"GPU") == 0)
        {
          // Add info to Interconnectivity Matrix
          for (dev_b = a_arch->get_nhst(); dev_b < a_arch->get_ndev() + a_arch->get_nhst(); ++dev_b)
          {
            iss_a >> word;
            if (word.compare("X") == 0) a_arch->get_devmx()[dev_a][dev_b] = 0;
            else if (word.compare("NV1") == 0) a_arch->get_devmx()[dev_a][dev_b] = 25; // Based on experiments 22 GB/s
            else if (word.compare("NV2") == 0) a_arch->get_devmx()[dev_a][dev_b] = 50; // Based on experiments 45 GB/s
            else if (word.compare("SYS") == 0) a_arch->get_devmx()[dev_a][dev_b] = 0; // No P2P - Based on experiments 6 GB/s
            else a_arch->get_devmx()[dev_a][dev_b] = -1;
      
          }
          // Skip network information // ***** Future_Work *****
          for (int i = 0; i <= n_net_v; ++i) iss_a >> word;
          
          // Get CPU Affinity
          // Replace separators with spaces
          for(int i = 0; i < word.length(); ++i) if(word[i]==',' || word[i]=='-') word[i] = *strdup(" ");

          iss_b.clear();
          iss_b << word;
          n_core = 0;
          while (iss_b >> word_b || n_core==3)
          {
            if (stringstream(word_b) >> numa_t.h_i[n_core]) ++n_core;
          }

          // Add numa information
          iss_a >> word;
          stringstream(word) >> nma_x;
          // Check if new information and copy
          // printf("NUMA %d Conf: %d\n", nma_x, a_arch->get_numa()[nma_x].conf);
          if (!a_arch->get_numa()[nma_x].conf){
            a_arch->get_numa()[nma_x] = numa_t;
            a_arch->get_numa()[nma_x].conf = 1;
          }

          // Add Host to Device information // ToDo: Parametrize this
          if(nma_x)
          {
            a_arch->get_devmx()[0][dev_a]=6;
            a_arch->get_devmx()[1][dev_a]=10;
            a_arch->get_devmx()[dev_a][0]=6;
            a_arch->get_devmx()[dev_a][1]=10;
          }
          else
          {
            a_arch->get_devmx()[0][dev_a]=10;
            a_arch->get_devmx()[1][dev_a]=6;
            a_arch->get_devmx()[dev_a][0]=10;
            a_arch->get_devmx()[dev_a][1]=6;
          }

          // cout << word << endl;
          ++dev_a;
        }
      
      // else{
      //   // cout << "other" << '\n';
      //   // cout << word << " " << word.compare(0,3,"GPU") << endl;
      // }

      } // End While: Get all words in a line 
      if (line_n == 0) // End of line 0
      {
        // Allocate interconnectivity matrix
        a_arch->set_ndev(n_dev_v);
        
        // printf("dev_mx_p address: %p\n", (*dev_mx_p));

        
        // hst_mx = (int**) malloc(n_dev * sizeof(int*));
        // for (dev_a = 0; dev_a < n_dev; ++dev_a)
        //   hst_mx[dev_a] = (int*)malloc(n_hst_v * sizeof(int));

        // Initialize variable dev_a
        dev_a = a_arch->get_nhst();
      }
      ++line_n;
    }
    arch_f->close();
    iss_a.clear();
    iss_b.clear();
    
    a_arch->set_nnet(n_net_v);
    a_arch->get_devmx()[0][1]=40; // Theoretically 38.4 GB/s
    a_arch->get_devmx()[1][0]=40; // Theoretically 38.4 GB/s
    a_arch->get_devmx()[0][0]=0; // Theoretically 38.4 GB/s
    a_arch->get_devmx()[1][1]=0; // Theoretically 38.4 GB/s
    return 0;
  }
  else{
    cout << "Unable to open file"; 
    return 1;
  }

}

void autostrat::copy_mx(arch * a_arch, int ***dev_mx_cpy_p, int new_cpy)
{
  int dev_a, dev_b;
  // Check if new copy
  if (new_cpy)
  {
    // printf("Allocating memory for copy\n");
    *dev_mx_cpy_p = (int**) malloc((a_arch->get_ndev() + a_arch->get_nhst()) * sizeof(int*));
    for (dev_a = 0; dev_a < (a_arch->get_ndev() + a_arch->get_nhst()); ++dev_a)
      (*dev_mx_cpy_p)[dev_a] = (int*)malloc((a_arch->get_ndev() + a_arch->get_nhst()) * sizeof(int));
  }
  // Copy device matrix information
  for (dev_a = 0; dev_a < a_arch->get_ndev() + a_arch->get_nhst(); ++dev_a)
    for (dev_b = 0; dev_b < a_arch->get_ndev() + a_arch->get_nhst(); ++dev_b)
      (*dev_mx_cpy_p)[dev_a][dev_b] = a_arch->get_devmx()[dev_a][dev_b];
}

void autostrat::del_mx(arch * a_arch, int ***dev_mx_cpy_p)
{
  int dev_a, dev_b;
    for (dev_a = 0; dev_a < (a_arch->get_ndev() + a_arch->get_nhst()); ++dev_a) free((*dev_mx_cpy_p)[dev_a]);
  free(*dev_mx_cpy_p);
}

// void print_mx(int ***dev_mx_p, int a_arch->get_ndev(), int a_arch->get_nhst())
void autostrat::print_mx(arch * a_arch)
{
  int dev_a, dev_b;
  printf("Interconnectivity Matrix\n");
  for (dev_a = 0; dev_a < a_arch->get_ndev() + a_arch->get_nhst(); ++dev_a) {
    for (dev_b = 0; dev_b < a_arch->get_ndev() + a_arch->get_nhst(); ++dev_b)
        printf("%2d ", a_arch->get_devmx()[dev_a][dev_b]);
    printf("\n");
  }

}

// printf("dev_mx address: %p\n", dev_mx);

void autostrat::print_numa(arch *a_arch)
{
  // Print Numa Cores
  printf("Numa Dom: %d\n", a_arch->get_nnuma());
  for (int n_dom = 0; n_dom < a_arch->get_nnuma(); ++n_dom) {
    printf("Numa: %d Cores: ",  n_dom);
    for (int dev_n = 0; dev_n < 4; ++dev_n)
      printf("%d ",  (a_arch->get_numa())[n_dom].h_i[dev_n]);
    printf("\n");
  }
}

int autostrat::get_ops(ifstream *ops_f, ve_ops *all_ops, arch * a_arch)
{
  int line_n = 0, word_n = 0, op_set = 0, d_id, op_size, i_dev;
  string line, word, word_b;
  stringstream iss_a, iss_b;
  // enum c_ops a_cop;
  // enum h_mth a_hmt;
  cops_def *a_ops;

  if (ops_f->is_open())
  {
    while (!ops_f->eof())
    {
      getline (*ops_f, line);
      // Replace separators with spaces
      for(int i = 0; i < line.length(); ++i) if(line[i]==',' || line[i]=='-') line[i] = *strdup(" ");
      iss_a.clear();
      iss_a << line;
      word_n = 0;
      a_ops = new cops_def();
      // Get all words in a line
      while (iss_a >> word)
      {
        if (word_n == 0){
          // Get operation size
          stringstream(word) >> op_size;
          a_ops->set_size(op_size);
          iss_a >> word; // Get next word

          while (word.length() == 2)
          {
            if (word.compare(0,1,"H") == 0)
            {
              // Host
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              word.erase(0,1);
              stringstream(word) >> d_id;
              a_ops->add_orig(d_id);
            }
            else if (word.compare(0,1,"D") == 0)
            {
              // Device
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              word.erase(0,1);
              stringstream(word) >> d_id;
              a_ops->add_orig(d_id + a_arch->get_nhst()); // Add No. Hosts
            }
            else if (word.compare(0,2,"AH") == 0)
            {
              // All Devices
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              for(i_dev = 0; i_dev<a_arch->get_nhst(); ++i_dev) a_ops->add_orig(i_dev); // Add No. Hosts
            }
            else if (word.compare(0,2,"AD") == 0)
            {
              // All Devices
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              for(i_dev = 0; i_dev<a_arch->get_ndev(); ++i_dev) a_ops->add_orig(i_dev + a_arch->get_nhst()); // Add No. Hosts
            }
            iss_a >> word;
          }
          if (word.length() == 3 && op_set == 0)
          {
            if (word.compare(0,3,"D2D") == 0)
            {
              a_ops->set_coop(D2D);
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              op_set = 1;
            }
            else if (word.compare(0,3,"BRC") == 0)
            {
              a_ops->set_coop(BRC);
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              op_set = 1;
            }
            else if (word.compare(0,3,"SCT") == 0)
            {
              a_ops->set_coop(SCT);
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              op_set = 1;
            }
            else if (word.compare(0,3,"GAT") == 0)
            {
              a_ops->set_coop(GAT);
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              op_set = 1;
            }
            else if (word.compare(0,3,"RED") == 0)
            {
              a_ops->set_coop(RED);
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              op_set = 1;
            }
            else
            {
              printf("Invalid Input Operation\n");
            }
            iss_a >> word;
          }

          while (word.length() == 2)
          {
            if (word.compare(0,1,"H") == 0)
            {
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              word.erase(0,1);
              stringstream(word) >> d_id;
              a_ops->add_dest(d_id);
            }
            else if (word.compare(0,1,"D") == 0 && op_set == 1)
            {
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              word.erase(0,1);
              stringstream(word) >> d_id;
              a_ops->add_dest(d_id + a_arch->get_nhst()); // Add No. Hosts
            }
            else if (word.compare(0,2,"AH") == 0)
            {
              // All Hosts
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              for(i_dev = 0; i_dev<a_arch->get_nhst(); ++i_dev) 
                if((def_ops->get_orig())->front() != i_dev)
                  a_ops->add_dest(i_dev); // Add No. Hosts
            }
            else if (word.compare(0,2,"AD") == 0)
            {
              // All Devices
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
              for(i_dev = a_arch->get_nhst(); i_dev<a_arch->get_nnod(); ++i_dev)
                if ((a_ops->get_orig())->front() != i_dev) 
                  a_ops->add_dest(i_dev); // Add No. Devices
            }
            iss_a >> word;
          }
          if (word.length() == 3 && op_set == 1)
          {
            if (word.compare(0,3,"P2P") == 0)
            {
              a_ops->set_mhtd(P2P);
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
            }
            else if (word.compare(0,3,"DVT") == 0)
            {
              a_ops->set_mhtd(DVT);
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
            }
            else if (word.compare(0,3,"MXF") == 0)
            {
              a_ops->set_mhtd(MXF);
              cout << "[IN:] "<< word << " " << endl; //ToBeDeleted
            }
            else
            {
              printf("Invalid Input Method\n");
            }
            iss_a >> word;
            stringstream(word) >> d_id;
          }
          else
          {
            printf("Invalid Input\n");
          }

        }
        
        ++word_n;
      
      // else{
      //   // cout << "other" << '\n';
      //   // cout << word << " " << word.compare(0,3,"GPU") << endl;
      // }

      } // End While: Get all words in a line 
      

      // Save OP
      all_ops->push_back(a_ops);
      op_set = 0;

      ++line_n;
    }
    ops_f->close();
    
    return 0;
  }
  else
  {
    cout << "Unable to open file"; 
    return 1;
  }
}

void autostrat::print_ops(ve_ops *all_ops)
{
  printf("[PRINT:] Operation(s):\n");
  for (auto& it : *all_ops)
  {
    // Print the values
    printf("[PRINT:] Type: ");
    if (it->get_coop() == D2D) printf("D2D");
    else if (it->get_coop() == BRC) printf("BRC");
    else if (it->get_coop() == SCT) printf("SCT");
    else if (it->get_coop() == GAT) printf("GAT");
    else if (it->get_coop() == RED) printf("RED");

    printf(", Mthd: ");
    if (it->get_mhtd() == P2P) printf("P2P");
    else if (it->get_mhtd() == DVT) printf("DVT");
    else if (it->get_mhtd() == MXF) printf("MXF");

    printf(", Size: %d ", it->get_size());

    printf(", Orig: ");
    for (auto element : *it->get_orig()) {
        cout << element << " ";
    }


    // printf("%d ", it->get_size());
    printf(", Dest: ");
    for (auto element : *it->get_dest()) {
        cout << element << " ";
    }
    printf("\n");

    // printf("%d ", it->get_size());
    // printf("Orig: Dest: Size: %d Offs: \n", it->get_size());
    // cout << it->orig << ' ';
  } 
}

void autostrat::ops_deps(arch * a_arch, ve_ops *all_ops, ve_deps *all_deps, int ***dev_mx_cpy_p, ve_meminfo *all_meminfo)
{
  int m_paths = 4, m_hops = 5, ind_p = 1; // ToDo: define as inputs max_paths, max_hops, use indirect_paths
  int dev_a = 0, dev_b = 0, dev_i, dev_ii, i_hops, n_hops, i_paths, p_done, max_bw, lnk_bw, i_link, n_link, h_aff;
  float min_lat;
  // typedef std::vector<int> a_path;

  // Create a copy Interconnectivity Matrix 
  autostrat::copy_mx(a_arch, dev_mx_cpy_p, 1);

  class op_path
  {
    public:
      op_path(int a_n_id, float a_p_bwth, int a_n_hops): n_id(a_n_id), n_hops(a_n_hops){
        p_op_path = NULL;
        // p_lat = 1/a_p_bwth;
        p_lat = 0;
        // printf("Link lat : %f \n", p_lat);
      };
      op_path(int a_n_id, float a_p_bwth, int a_n_hops, op_path * a_op_path): n_id(a_n_id), n_hops(a_n_hops), p_op_path(a_op_path){
        p_lat = p_op_path->p_lat + 1/a_p_bwth;
        o_id = p_op_path->n_id;
      };
      ~op_path(){};
      // ToDo: define set and get methods
      op_path * p_op_path;
      int n_id;
      int o_id;
      float p_lat; // path latency
      int n_hops;
      // int get_orig_id(){return o_id;};
        // variable = (condition) ? expressionTrue : expressionFalse;
  } * iop_path;

  typedef std::vector<op_path*> op_paths;
  op_paths::iterator it_path;
  ve_deps::iterator i_cop;
  
  op_paths v_paths, vi_paths; // vector of paths

  std::vector<int> node_conect; // Node connectivity
  std::vector<int> op_orig;
  std::vector<int> op_dest;

  for (auto& t_op : *all_ops)
  {
    switch (t_op->get_coop())
    {
      case D2D:
        printf("[AUST:] Setting D2D: "); //ToBeDeleted
        // Unidirectional one-to-one transfer between devices
        dev_a = *(t_op->get_orig())->begin();
        // n_hops = 2;
        switch (t_op->get_mhtd())
        {
          case P2P:
            printf("Using P2P\n"); //ToBeDeleted
            // Allows multiple destinations
            for (auto& dev_b : *t_op->get_dest()) {
              // Check direct path D2D
              if ((*dev_mx_cpy_p)[dev_a][dev_b]>0) {
                // ToDo: Remove available link?
                printf("[AUST:] Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, (*dev_mx_cpy_p)[dev_a][dev_b]); //ToBeDeleted
                all_deps->push_back(new cops_dep(0, 0, dev_a, dev_b, a_arch->node_id[dev_a], a_arch->node_id[dev_b], t_op->get_size(), 0, 0, 0));
                all_meminfo->push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
                all_meminfo->push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
              } else if(ind_p) {
                // ToDo: Validate indirect path (-1)?
                printf("[AUST:] Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, (*dev_mx_cpy_p)[dev_a][dev_b]); //ToBeDeleted
                all_deps->push_back(new cops_dep(0, 0, dev_a, dev_b, a_arch->node_id[dev_a], a_arch->node_id[dev_b], t_op->get_size(), 0, 0, 0));
                all_meminfo->push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
                all_meminfo->push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
              } else {
                printf("[AUST_ERR:] Invalid H/D combination\n"); //ToBeDeleted
              }
            }
            // return 0;
            break; // P2P

          // Max-Flow
          case MXF:
            printf("Using MXF\n"); //ToBeDeleted
            // Find paths with bigher bandwitdh
            // 2-Hop - Single source - Single sink
            dev_b = *(t_op->get_dest())->begin();

            i_paths = 0;
            p_done = 0;
            // n_hops = a_arch->get_nnod() - 2; // max number of hops
            all_meminfo->push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
            all_meminfo->push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
            // ToDo: check numa affinty 
            // Check direct connection (1 hop)
            if ((*dev_mx_cpy_p)[dev_a][dev_b]>0){
              all_deps->push_back(new cops_dep(0, 0, dev_a, dev_b, a_arch->node_id[dev_a], a_arch->node_id[dev_b], 0, 0, 0, i_paths));
              i_paths++;
              // Remove link
              printf("[AUST:] Valid H/D Orig: %d Dest: %d Value: %d i_paths: %d\n", dev_a, dev_b, (*dev_mx_cpy_p)[dev_a][dev_b], i_paths); //ToBeDeleted
              (*dev_mx_cpy_p)[dev_a][dev_b] = 0;
              (*dev_mx_cpy_p)[dev_b][dev_a] = 0;

            }

            // Build Paths
            while(!p_done){
              // Find connecting paths
              if(i_paths>=m_paths) {
                p_done = 1;
              } else {
                max_bw = 0;
                for (dev_i = 0; dev_i < a_arch->get_nnod(); ++dev_i){
                  // Check if link exists 
                  if (dev_i != dev_b && (*dev_mx_cpy_p)[dev_a][dev_i] > 0 && (*dev_mx_cpy_p)[dev_i][dev_b] > 0 ) {
                    // Get the minimum value of the link 
                    lnk_bw = ((*dev_mx_cpy_p)[dev_a][dev_i] > (*dev_mx_cpy_p)[dev_i][dev_b]) ? (*dev_mx_cpy_p)[dev_i][dev_b] : (*dev_mx_cpy_p)[dev_a][dev_i];
                    // Set max bandwidth
                    if (lnk_bw > max_bw) {
                      max_bw = lnk_bw;
                      dev_ii = dev_i;
                    }
                  }
                }

                if (max_bw > 0) {
                  // Generate dependencies
                  op_deps = new cops_dep(0, 0, dev_a, dev_ii, a_arch->node_id[dev_a], a_arch->node_id[dev_ii], 0, i_paths, 0, i_paths);
                  all_deps->push_back(op_deps);
                  op_deps = new cops_dep(1, 0, dev_ii, dev_b, a_arch->node_id[dev_ii], a_arch->node_id[dev_b], 0, 0, i_paths, i_paths);
                  all_deps->push_back(op_deps);
                  // Define memory information
                  all_meminfo->push_back(new mem_info{dev_ii, 0});
                  // Remove links
                  printf("[AUST:] Valid H/D Orig: %d Dest: %d MaxBW: %d i_paths: %d\n", dev_ii, dev_b, max_bw, i_paths);
                  (*dev_mx_cpy_p)[dev_a][dev_ii] = 0;
                  (*dev_mx_cpy_p)[dev_ii][dev_b] = 0;
                  i_paths++;
                } else {
                  p_done = 1;
                  printf("i_paths: %d\n", i_paths);
                }
              }
            }

            // Calculate sizes / offsets
            // Evenly 
            dev_i = 1;
            for (auto& a_dep : *all_deps) {
              a_dep->size = t_op->get_size()/i_paths;
              a_dep->of_s = a_dep->of_s*a_dep->size;
              a_dep->of_d = a_dep->of_d*a_dep->size;
              // printf("Orig: %d Dest: %d Size: %d Offs: %d\n", c_dep->orig, c_dep->dest, c_dep->size, c_dep->offs);
              dev_i++;
            }

            for (auto& a_mem : *all_meminfo) {
              if (a_mem->node_id != dev_a && a_mem->node_id != dev_b) {
                
                a_mem->size = t_op->get_size()/i_paths * sizeof(int); // ToDo: MEM integers only
              }
            }
            
            // Size based on bandwidth
            // ******************** HERE ******************** //


          break; // MXF

          // Distant Vector
          case DVT:
            printf("Using DVT\n"); //ToBeDeleted
            // Find multiple routes that increase throughput
            // Single sink
            // dev_a = *(t_op->get_orig())->begin();
            dev_b = *(t_op->get_dest())->begin();
            i_paths = 0;

            // ToDo: check numa affinty 

            // Find direct path between origin / destination
            if ((*dev_mx_cpy_p)[dev_a][dev_b]>0) {
              printf("Valid O/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, (*dev_mx_cpy_p)[dev_a][dev_b]); //ToBeDeleted
              all_meminfo->push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
              all_meminfo->push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
              // op_deps = new cops_dep(0, 0, dev_a, dev_b, 0, 0, i_paths);
              // all_deps->push_back(op_deps);
              all_deps->push_back(new cops_dep(0, 0, dev_a, dev_b, a_arch->node_id[dev_a], a_arch->node_id[dev_b], 0, 0, 0, i_paths));
              i_paths++;
              // Remove link
              (*dev_mx_cpy_p)[dev_a][dev_b] = 0;
              (*dev_mx_cpy_p)[dev_b][dev_a] = 0;
            }

            if (i_paths < m_paths) {
              dev_ii = dev_b;
              p_done = 0;
              i_link = 0;
              n_link = 0;
              i_hops = 0;

              printf("Creating dest %d \n", dev_ii);
              v_paths.push_back(new op_path(dev_ii, 0, i_hops));
              i_hops = 1;
              i_link = 1;

              while (!p_done) {
                // Calculate latency 
                // Store paths increasing number of links
                for (dev_i = 0; dev_i < a_arch->get_nnod(); ++dev_i) {
                  // printf("Testing Link %d to %d \n", dev_i, dev_ii);
                  if (dev_i != dev_ii && (*dev_mx_cpy_p)[dev_i][dev_ii] > 0) {
                    // Add link to list
                    printf("Creating II Link %d to %d bw: %d, parent %d\n", dev_i, dev_ii, (*dev_mx_cpy_p)[dev_i][dev_ii], (v_paths.at(i_link-1))->n_id);
                    iop_path = new op_path(dev_i, static_cast< float >((*dev_mx_cpy_p)[dev_i][dev_ii]), i_hops, v_paths.at(i_link-1));
                    v_paths.push_back(iop_path);
                    (*dev_mx_cpy_p)[dev_i][dev_ii] = 0;
                    (*dev_mx_cpy_p)[dev_ii][dev_i] = 0;
                    ++n_link;
                    if (dev_i == dev_a) {
                      // Valid link to origin
                      printf("Valid Link %d to %d latency: %f\n", dev_i, dev_ii, v_paths.at(i_link)->p_lat);
                      vi_paths.push_back(iop_path);
                    }
                  }
                  // printf("Testing H/D Orig: %d Dest: %d Value: %d\n", dev_i, dev_b, (*dev_mx_cpy_p)[dev_i][dev_b]); //ToBeDeleted
                }
                if (n_link > i_link) {
                  // Assign origin from list of valid links
                  dev_ii = v_paths.at(i_link)->n_id;
                  // printf("New Node %d \n", dev_ii);
                  i_hops = v_paths.at(i_link)->n_hops + 1;
                  ++i_link;
                } else if (i_hops >= m_hops) {
                  p_done = 1;
                  printf(">>>>> Max Hops <<<<<\n");
                } else {
                  // No more available links 
                  printf("No More Valid Links\n");
                  p_done = 1;
                }
                // Get other node with same numer of hops or increase 
              } 
            }
            
            // Find lowest latency path(s) among valid paths
            p_done = 0;
            while(!p_done) {
              min_lat = 1000; //ToDo: Figure this out
              // Evaluate all valid paths to find lowest latency 
              for (auto i_path = vi_paths.begin(); i_path != vi_paths.end(); i_path++) {
                if (min_lat > (*i_path)->p_lat) {
                  printf("Path to %d latency: %f\n", (*i_path)->n_id, (*i_path)->p_lat);
                  min_lat = (*i_path)->p_lat;
                  it_path = i_path;
                  iop_path = (*i_path);
                }
              }
              if (min_lat != 1000) {
                // Min latency path found
                while(iop_path->n_id != dev_b) {
                  // Generate link for communication pattern based on path
                  all_deps->push_back(new cops_dep(iop_path->n_hops-1, 0, iop_path->n_id, iop_path->o_id, a_arch->node_id[iop_path->n_id], a_arch->node_id[iop_path->o_id], 0, 0, 0, i_paths));
                  // Define memory information
                  printf("[min] Path from %d to %d latency: %f\n", iop_path->n_id, iop_path->o_id, iop_path->p_lat);
                  if (iop_path->n_id != dev_a) all_meminfo->push_back(new mem_info{iop_path->n_id, 0});
                  iop_path = iop_path->p_op_path;
                }
                i_paths++;
                vi_paths.erase(it_path--);
              }
              if(i_paths>=m_paths){
                p_done = 1;
              }
            }

            // ******************** HERE ******************** //
            // Calculate sizes / offsets
            // Evenly 
            for (auto& a_dep : *all_deps) {
              a_dep->size = t_op->get_size()/i_paths;
              if (a_dep->orig == dev_a) {
                // Origin
                a_dep->of_s = a_dep->ipth*a_dep->size;
                a_dep->of_d = 0; 
              } else if (a_dep->dest == dev_b) {
                // Destination
                a_dep->of_s = 0;
                a_dep->of_d = a_dep->ipth*a_dep->size; 
              } else { 
                // Intermediate
                a_dep->of_s = 0;
                a_dep->of_d = 0; 
              }
              // printf("Orig: %d Dest: %d Size: %d Offs: %d\n", c_dep->orig, c_dep->dest, c_dep->size, c_dep->offs);
              dev_i++;
            }
            
            for (auto& a_mem : *all_meminfo){
              if (a_mem->node_id != dev_a && a_mem->node_id != dev_b) {
                a_mem->size = t_op->get_size()/i_paths * sizeof(int); // ToDo: MEM integers only
              }
            }
            // ToDo: Size based on bandwidth
            // ******************** HERE ******************** //

          break; // DVT


          default:
            printf("Invalid Method\n");
            // return 1;
        }
      break; // D2D

      case BRC:
        printf("[AUST:] Setting BRC: ");
        // Unidirectional one-to-all transfer between devices
        // Copy origin vector
        op_orig = *t_op->get_orig();
        // Copy destination vector
        op_dest = *t_op->get_dest();
        
        // Define memory information
        // Setup memory - Origin
        for (auto& dev_a : op_orig)
          all_meminfo->push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
        // Setup memory - Dest
        for (auto& dev_b : op_dest)
          all_meminfo->push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
        
        switch (t_op->get_mhtd())
        {
          case P2P:
          printf("Using P2P\n");
          dev_a = *(t_op->get_orig())->begin();
          // all_meminfo->push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
          for (auto dev_b : *t_op->get_dest()) {
            // Check direct path P2P
            if ((*dev_mx_cpy_p)[dev_a][dev_b]>0) {
              // ToDo: Remove available link?
              printf("Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, (*dev_mx_cpy_p)[dev_a][dev_b]);
              all_deps->push_back(new cops_dep(0, 0, dev_a, dev_b, a_arch->node_id[dev_a], a_arch->node_id[dev_b], t_op->get_size(), 0, 0, 0));
              // all_meminfo->push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
            } else if (ind_p){
              // ToDo: Validate indirect path (-1)?
              printf("Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, (*dev_mx_cpy_p)[dev_a][dev_b]);
              all_deps->push_back(new cops_dep(0, 0, dev_a, dev_b, a_arch->node_id[dev_a], a_arch->node_id[dev_b], t_op->get_size(), 0, 0, 0)); 
            } else {
              printf("Invalid H/D combination\n");
            }
          }
          // return 0;
          break; // P2P

          // Max-Flow
          case MXF:
            printf("Using MXF\n");
            // Find paths with bigher bandwith
            // 2-Hop - Single source - Multi-sink

            i_paths = 0;
            p_done = 0;
            // n_hops = a_arch->get_nnod() - 2; // max number of hops
            // ToDo: check numa affinty 
            
            while (!p_done) {
              max_bw = 0;
              // Find nodes with highest connectivity
              for (auto& dev_a : op_orig) {
                // Find max bandwidth
                for (auto& dev_b : op_dest) {
                  if ((*dev_mx_cpy_p)[dev_a][dev_b]>max_bw){
                    // Set max bandwidth
                    max_bw = (*dev_mx_cpy_p)[dev_a][dev_b];
                    dev_i = dev_a;
                    dev_ii = dev_b;
                  }
                }
              }

              if (max_bw > 0) {
                // Generate dependencies
                if (dev_i == *(t_op->get_orig())->begin()){
                  printf("A %d\n", *(t_op->get_orig())->begin());
                  all_deps->push_back(new cops_dep(0, 0, dev_i, dev_ii, a_arch->node_id[dev_i], a_arch->node_id[dev_ii], t_op->get_size(), 0, 0, i_paths));
                } else {
                  printf("B\n");
                  all_deps->push_back(new cops_dep(1, 0, dev_i, dev_ii, a_arch->node_id[dev_i], a_arch->node_id[dev_ii], t_op->get_size(), 0, 0, i_paths));
                }
                // Remove links
                printf("[AUST:] Valid H/D Orig: %d Dest: %d MaxBW: %d i_paths: %d\n", dev_i, dev_ii, max_bw, i_paths);
                (*dev_mx_cpy_p)[dev_i][dev_ii] = 0;
                (*dev_mx_cpy_p)[dev_ii][dev_i] = 0;
                op_dest.erase(std::remove(op_dest.begin(), op_dest.end(), dev_ii), op_dest.end());
                op_orig.push_back(dev_ii);
                i_paths++;
              } else {
                printf("no more valid paths\n");
                p_done=1;
              }
              // Check if all destinations have been reached
              if(op_dest.empty()){
                p_done = 1;
                printf("all done \n");
              }
            }


          break; // MXF

          // Distant Vector
          case DVT:
          printf("Using DVT\n");
            p_done = 0;
            dev_a = *(t_op->get_orig())->begin();
            
            while (!p_done) {
              // Find connecting paths
              for (auto& dev_b : *t_op->get_dest()) {
                printf("Creating dest %d \n", dev_b);
                i_hops = 1;

                v_paths.push_back(new op_path(dev_ii, 0, i_hops));
                for (dev_i = 0; dev_i < a_arch->get_nnod(); ++dev_i) {

                }

              }
            }
          break; // DVT


          default:
            printf("Invalid Method\n");
            // return 1;
        }
      break; // BRC

      case SCT:
        printf("Unsupported H/D combination\n");
        printf("\n");
        switch (t_op->get_mhtd())
        {
          case P2P:
          // return 0;
          break; // P2P

          // Distant Vector
          // Max-Flow

          default:
            printf("Invalid Method\n");
            // return 1;
        }
      break; // 
      
      case GAT:
        printf("Unsupported H/D combination\n");
        printf("\n");
        switch (t_op->get_mhtd())
        {
          case P2P:
          // return 0;
          break; // P2P

          // Distant Vector
          // Max-Flow

          default:
            printf("Invalid Method\n");
            // return 1;
        }
      break; // 

      case RED:
        printf("Unsupported H/D combination\n");
        printf("\n");
        switch (t_op->get_mhtd())
        {
          case P2P:
          // return 0;
          break; // P2P

          // Distant Vector
          // Max-Flow

          default:
            printf("Invalid Method\n");
            // return 1;
        }

      break; // RED

      default:
        printf("Invalid Operation\n");
        // return 1;
        
        

      // case DVT:
      //   printf("P2P\n");

      // break; // DVT

      // case MXF:
      //   printf("\n");

      // break; // MXF

      // default:
      //   printf("Invalid Method\n");
      //   return 1;
    } // End Switch
  } // End for (all_ops)
}

void autostrat::print_cpat(ve_deps *all_deps)
{
  printf("[OUT:] Communication Pattern: \n");
  for (auto& a_dep : *all_deps)
  {
    // Print Output values
    printf("[OUT:] Orig: %d Dest: %d Size: %zu O_Offs: %zu D_Offs: %zu, Path No.: %d\n", a_dep->orig, a_dep->dest, a_dep->size, a_dep->of_s, a_dep->of_d, a_dep->ipth);
  }
}

void autostrat::auto_malloc(arch * a_arch, ve_meminfo *all_meminfo, void ** mem_ptr)
{
  size_t chunk_s;
  int dev_id;
  // Memory allocation
  printf("[MALLOC:] Memory  Allocation:\n");
  for (auto& a_mem : *all_meminfo)
  {
    // chunk_s = sizeof(int) * a_mem->size;
    if (a_mem->node_id<a_arch->get_nhst())
    {
      printf("[MALLOC:] Dev: %d Size: %zu\n", a_mem->node_id, a_mem->size);
      mem_ptr[a_mem->node_id] = numa_alloc_onnode(a_mem->size, a_mem->node_id);
      // mem_ptr[a_mem->node_id] = (int*) malloc(a_mem->size); 
    }
    else
    {
      dev_id = a_arch->node_id[a_mem->node_id];
      printf("[MALLOC2:] Node: %d Dev: %d Size: %zu\n", a_mem->node_id, dev_id, a_mem->size);
      mem_ptr[a_mem->node_id] = omp_target_alloc(a_mem->size, dev_id);
      
    } 
  }
}


// void exec_op(ve_deps *all_deps, void *** mem_ptr)
// {
//   printf("[EXEC:] Execution:\n");
//   for (auto& a_dep : *all_deps)
//   {
//     // Print the values
//     printf("[EXEC:] Orig: %d Dest: %d Size: %d O_Offs: %d D_Offs: %d, Path No.: %d\n", a_dep->orig, a_dep->dest, a_dep->size, a_dep->of_s, a_dep->of_d, a_dep->ipth);
//   }
  

// }

void autostrat::auto_mfree(arch * a_arch, ve_meminfo *all_meminfo, void ** mem_ptr)
{
  size_t chunk_s;
  int dev_id;
  // Memory Free
  printf("[MFREE:] Memory Free:\n");
  for (auto& a_mem : *all_meminfo)
  {
    // chunk_s = sizeof(int) * a_mem->size;
    if (a_mem->node_id<a_arch->get_nhst())
    {
      printf("[MFREE:] Dev: %d Size: %zu\n", a_mem->node_id, a_mem->size);
      numa_free(mem_ptr[a_mem->node_id], chunk_s);
    }
    else
    {
      dev_id = a_mem->node_id-a_arch->get_nhst();
      printf("[MFREE2:] Node: %d Dev: %d Size: %zu\n", a_mem->node_id, dev_id, a_mem->size);
      omp_target_free(mem_ptr[a_mem->node_id], dev_id);
    }
  }
}



  // printf("All Operations\n");
  // for (auto& it : all_ops)
  // {
  //   // Print the values
  //   printf("Orig: Dest: Size: %d Offs: \n", it->get_size());
  //   // cout << it->orig << ' ';
  // } 


// Get Array length from command line inputs
  // // if(const char* arr_size = std::getenv("ARR_SZ"))
  // //   arr_len = atoi(arr_size);

  // arr_len = 60;



//**************************************************//
// qsub -I -n 1 -t 6:00 -q gpu_v100_smx2
// cd ~/collective_ops/tests/D2D; source ~/source/LLVM/exports.sh
// module load llvm
//**************************************************//

//**************************************************//
//                      COMPILE                     //
//**************************************************//
// PROG=auto_s_v0; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 $PROG.cpp -lnuma
// PROG=d2d_test_7; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -L/soft/compilers/cuda/cuda-11.0.2/lib64 -L/soft/compilers/cuda/cuda-11.0.2/targets/x86_64-linux/lib/ -I/soft/compilers/cuda/cuda-11.0.2/include -ldl -lcudart -pthread $PROG.cpp
// PROG=auto_s_v0; clang++ -g -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -ldl -lcudart -lnuma -pthread $PROG.cpp -ferror-limit=9

//**************************************************//
//                        RUN                       //
//**************************************************//

// OMP_PROC_BIND=spread taskset -c 0,1,22,23 ./d2d_test_7.x

// nsys profile -o auto_s_v0_0 ./run_one.sh

