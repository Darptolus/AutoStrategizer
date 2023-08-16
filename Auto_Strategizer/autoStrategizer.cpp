#include "autoStrategizer.hpp"

namespace AutoStrategizer
{
  int get_topo(std::ifstream *arch_f, Architecture * a_arch)
  {
    int line_n = 0, word_n = 0, n_dev_v = 0, n_net_v = 0, dev_a = 0, dev_b = 0, n_core = 0, nma_x = 0;
    std::string line, word, word_b;
    std::stringstream iss_a, iss_b;

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
              if (std::stringstream(word_b) >> numa_t.h_i[n_core]) ++n_core;
            }

            // Add numa information
            iss_a >> word;
            std::stringstream(word) >> nma_x;
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

        } // End While: Get all words in a line 
        if (line_n == 0) // End of line 0
        {
          // Allocate interconnectivity matrix
          a_arch->set_ndev(n_dev_v);

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
      std::cout << "Unable to open file"; 
      return 1;
    }
  }

  Architecture::Architecture() : n_dev(0),
                                 n_net(0),
                                 n_hst(0),
                                 n_nma(0),
                                 n_nod{0},
                                 dev_mx(nullptr),
                                 dev_mx_cpy(nullptr)
  {
    n_nma = numa_num_configured_nodes();
    n_hst = n_nma; // Single Node
    numa_d = (numa_dom*) malloc(n_nma * sizeof(numa_dom));
    // Initialize numa domains
    for (int n_dom = 0; n_dom < n_nma; ++n_dom)
      numa_d[n_dom].conf = 0;
    // Fill Host to Host information
  }

  Architecture::~Architecture()
  {
    for (int dev = 0; dev < (n_nod); ++dev)
      free(dev_mx[dev]);
    free(dev_mx);
    free(dev_mx_cpy);
    free(numa_d);
    free(node_id);
  }

  AutoStrategizer::AutoStrategizer(std::string architectureFile) : dev_a(0),
                                                                   dev_b(0),
                                                                   dev_i(0),
                                                                   dev_ii(0),
                                                                   n_nodes(0),
                                                                   arr_len(0),
                                                                   n_org(0),
                                                                   n_dst(0),
                                                                   f_sts(0)
  { 
    std::ifstream arch_f(architectureFile);
    // All Operations
    
  
    // Get Topology
    if (get_topo(&arch_f, &t_arch))
      std::cout << "Unable to get topology" << std::endl;
    
    // Allocate memory pointers
    mem_ptr = (void **) malloc(sizeof(void**) * (t_arch.get_nnod()));
  }

  void AutoStrategizer::printTopo(printMode mode)
  {
    if(mode == CLI)
    {
      printf("Interconnectivity Matrix\n");
      for (dev_a = 0; dev_a < t_arch.get_ndev() + t_arch.get_nhst(); ++dev_a)
      {
        for (dev_b = 0; dev_b < t_arch.get_ndev() + t_arch.get_nhst(); ++dev_b)
          printf("%2d ", t_arch.get_devmx()[dev_a][dev_b]);
        printf("\n");
      }
    }
  }

  void AutoStrategizer::printTopo_cpy(printMode mode)
  {
    if(mode == CLI)
    {
      printf("Interconnectivity Matrix\n");
      for (dev_a = 0; dev_a < t_arch.get_nnod(); ++dev_a)
      {
        for (dev_b = 0; dev_b < t_arch.get_nnod(); ++dev_b)
          printf("%2d ", t_arch.get_devmx_cpy()[dev_a][dev_b]);
        printf("\n");
      }
    }
  }

  void AutoStrategizer::copy_mx()
  {
    // Copy device matrix information
    for (dev_a = 0; dev_a < t_arch.get_nnod(); ++dev_a)
      for (dev_b = 0; dev_b < t_arch.get_nnod(); ++dev_b)
        t_arch.get_devmx_cpy()[dev_a][dev_b] = t_arch.get_devmx()[dev_a][dev_b];
  }

  void AutoStrategizer::printCO(printMode mode)
  {
    if(mode == CLI)
    {
      printf("[PRINT:] Operation(s):\n");
      for (auto& it : all_ops)
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
            std::cout << element << " ";
        }


        // printf("%d ", it->get_size());
        printf(", Dest: ");
        for (auto element : *it->get_dest()) {
            std::cout << element << " ";
        }
        printf("\n");

        // printf("%d ", it->get_size());
        // printf("Orig: Dest: Size: %d Offs: \n", it->get_size());
        // cout << it->orig << ' ';
      } 
    }
  }

  void AutoStrategizer::addCO(CollectiveOperation * a_CO)
  {
    all_ops.push_back(a_CO);
    // Create a copy Interconnectivity Matrix 
    this->copy_mx();

    // this->ops_deps(&t_arch, &all_ops, &all_deps, t_arch.get_devmxC(), &all_meminfo);
    this->ops_deps();
  }

  
  OD_vector * AutoStrategizer::getDeps(){
    return &all_deps;
  }

  CollectiveOperation * AutoStrategizer::getOP(){
    CollectiveOperation * t_op;
    t_op = all_ops.front();
    return t_op;
  }

  MI_vector * AutoStrategizer::getMI(){
    return &all_meminfo;
  }


  int AutoStrategizer::get_node_id(int node_num){
    return t_arch.node_id[node_num];
  }

  void ** AutoStrategizer::get_memptr(){
    return mem_ptr;
  }

  int AutoStrategizer::get_nnod(){
    return t_arch.get_nnod();
  }

  void AutoStrategizer::auto_malloc()
  {
    size_t chunk_s;
    int dev_id;
    // Memory allocation
    printf("[MALLOC:] Memory  Allocation:\n");
    for (auto& a_mem : this->all_meminfo)
    {
      // chunk_s = sizeof(int) * a_mem->size;
      if (a_mem->node_id<t_arch.get_nhst())
      {
        printf("[MALLOC:] Dev: %d Size: %zu\n", a_mem->node_id, a_mem->size);
        mem_ptr[a_mem->node_id] = numa_alloc_onnode(a_mem->size, a_mem->node_id);
        // mem_ptr[a_mem->node_id] = (int*) malloc(a_mem->size); 
      }
      else
      {
        dev_id = t_arch.node_id[a_mem->node_id];
        printf("[MALLOC2:] Node: %d Dev: %d Size: %zu\n", a_mem->node_id, dev_id, a_mem->size);
        mem_ptr[a_mem->node_id] = omp_target_alloc(a_mem->size, dev_id);
        
      } 
    }
  }

  void AutoStrategizer::auto_mfree()
  {
    size_t chunk_s;
    int dev_id;
    // Memory Free
    printf("[MFREE:] Memory Free:\n");
    for (auto& a_mem : this->all_meminfo)
    {
      // chunk_s = sizeof(int) * a_mem->size;
      if (a_mem->node_id<t_arch.get_nhst())
      {
        printf("[MFREE:] Dev: %d Size: %zu\n", a_mem->node_id, a_mem->size);
        numa_free(mem_ptr[a_mem->node_id], chunk_s);
      }
      else
      {
        dev_id = t_arch.node_id[a_mem->node_id];
        printf("[MFREE2:] Node: %d Dev: %d Size: %zu\n", a_mem->node_id, dev_id, a_mem->size);
        omp_target_free(mem_ptr[a_mem->node_id], dev_id);
      }
    }
  }

  // void ops_deps(Architecture * a_arch, CO_vector *all_ops, OD_vector *all_deps, int ***dev_mx_cpy_p, MI_vector *all_meminfo)
  void AutoStrategizer::ops_deps()
  {
    int m_paths = 4, m_hops = 5, ind_p = 1; // ToDo: define as inputs max_paths, max_hops, use indirect_paths
    int dev_a = 0, dev_b = 0, dev_i, dev_ii, i_hops, n_hops, i_paths, p_done, max_bw, lnk_bw, i_link, n_link, h_aff, f_new;
    float min_lat;
    OperationDependence * op_deps;
    // typedef std::vector<int> a_path;

    class op_path
    {
      public:
        // Constructor for origin 
        op_path(int a_n_id, float a_p_bwth, int a_n_hops): n_id(a_n_id), n_hops(a_n_hops){
          p_op_path = NULL;
          // p_lat = 1/a_p_bwth;
          p_lat = 0;
        };
        // Constructor for link
        op_path(int a_n_id, float a_p_bwth, int a_n_hops, op_path * a_op_path): n_id(a_n_id), n_hops(a_n_hops), p_op_path(a_op_path){
          p_lat = p_op_path->p_lat + 1/a_p_bwth;
          o_id = p_op_path->n_id;
        };
        ~op_path(){};
        // ToDo: define set and get methods
        op_path * p_op_path;
        int n_id; // node id
        int o_id; // origin_id
        float p_lat; // path latency
        int n_hops;
        // int get_orig_id(){return o_id;};
          // variable = (condition) ? expressionTrue : expressionFalse;
    } * iop_path;

    typedef std::vector<op_path*> op_paths;
    op_paths::iterator it_path;
    MI_vector::iterator i_cop;
    
    op_paths v_paths, vi_paths; // vector of paths

    std::vector<int> node_conect; // Node connectivity
    std::vector<int> op_orig;
    std::vector<int> op_dest;

    for (auto& t_op : this->all_ops)
    {
      switch (t_op->get_coop())
      {
        case D2D: // D2D
          printf("[AUST:] Setting D2D: "); //ToBeDeleted
          // Unidirectional one-to-one transfer between devices
          dev_a = *(t_op->get_orig())->begin();
          // n_hops = 2;
          switch (t_op->get_mhtd())
          {
            // Peer-to-Peer - D2D
            case P2P:
              printf("Using P2P\n"); //ToBeDeleted
              // Allows multiple destinations
              for (auto& dev_b : *t_op->get_dest()) {
                // Check direct path D2D
                if (t_arch.get_devmx_cpy()[dev_a][dev_b]>0) {
                  // ToDo: Remove available link?
                  printf("[AUST:] Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, t_arch.get_devmx_cpy()[dev_a][dev_b]); //ToBeDeleted
                  all_deps.push_back(new OperationDependence(0, 0, dev_a, dev_b, t_arch.node_id[dev_a], t_arch.node_id[dev_b], t_op->get_size(), 0, 0, 0));
                  all_meminfo.push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
                  all_meminfo.push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
                } else if(ind_p) {
                  // ToDo: Validate indirect path (-1)?
                  printf("[AUST:] Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, t_arch.get_devmx_cpy()[dev_a][dev_b]); //ToBeDeleted
                  all_deps.push_back(new OperationDependence(0, 0, dev_a, dev_b, t_arch.node_id[dev_a], t_arch.node_id[dev_b], t_op->get_size(), 0, 0, 0));
                  all_meminfo.push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
                  all_meminfo.push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
                } else {
                  printf("[AUST_ERR:] Invalid H/D combination\n"); //ToBeDeleted
                }
              }
              // return 0;
              break; // P2P

            // Max-Flow - D2D
            case MXF:
              printf("Using MXF\n"); //ToBeDeleted
              // Find paths with bigher bandwitdh
              // 2-Hop - Single source - Single sink
              dev_b = *(t_op->get_dest())->begin();

              i_paths = 0;
              p_done = 0;
              // n_hops = t_arch.get_nnod() - 2; // max number of hops
              all_meminfo.push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
              all_meminfo.push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
              // ToDo: check numa affinty 
              // Check direct connection (1 hop)
              if (t_arch.get_devmx_cpy()[dev_a][dev_b]>0){
                all_deps.push_back(new OperationDependence(0, 0, dev_a, dev_b, t_arch.node_id[dev_a], t_arch.node_id[dev_b], 0, 0, 0, i_paths));
                i_paths++;
                // Remove link
                printf("[AUST:] Valid H/D Orig: %d Dest: %d Value: %d i_paths: %d\n", dev_a, dev_b, t_arch.get_devmx_cpy()[dev_a][dev_b], i_paths); //ToBeDeleted
                t_arch.get_devmx_cpy()[dev_a][dev_b] = 0;
                t_arch.get_devmx_cpy()[dev_b][dev_a] = 0;

              }

              // Build Paths
              while(!p_done){
                // Find connecting paths
                if(i_paths>=m_paths) {
                  p_done = 1;
                } else {
                  max_bw = 0;
                  for (dev_i = 0; dev_i < t_arch.get_nnod(); ++dev_i){
                    // Check if link exists 
                    if (dev_i != dev_b && t_arch.get_devmx_cpy()[dev_a][dev_i] > 0 && t_arch.get_devmx_cpy()[dev_i][dev_b] > 0 ) {
                      // Get the minimum value of the link 
                      lnk_bw = (t_arch.get_devmx_cpy()[dev_a][dev_i] > t_arch.get_devmx_cpy()[dev_i][dev_b]) ? t_arch.get_devmx_cpy()[dev_i][dev_b] : t_arch.get_devmx_cpy()[dev_a][dev_i];
                      // Set max bandwidth
                      if (lnk_bw > max_bw) {
                        max_bw = lnk_bw;
                        dev_ii = dev_i;
                      }
                    }
                  }

                  if (max_bw > 0) {
                    // Generate dependencies
                    op_deps = new OperationDependence(0, 0, dev_a, dev_ii, t_arch.node_id[dev_a], t_arch.node_id[dev_ii], 0, i_paths, 0, i_paths);
                    all_deps.push_back(op_deps);
                    op_deps = new OperationDependence(1, 0, dev_ii, dev_b, t_arch.node_id[dev_ii], t_arch.node_id[dev_b], 0, 0, i_paths, i_paths);
                    all_deps.push_back(op_deps);
                    // Define memory information
                    all_meminfo.push_back(new mem_info{dev_ii, 0});
                    // Remove links
                    printf("[AUST:] Valid H/D Orig: %d Dest: %d MaxBW: %d i_paths: %d\n", dev_ii, dev_b, max_bw, i_paths);
                    t_arch.get_devmx_cpy()[dev_a][dev_ii] = 0;
                    t_arch.get_devmx_cpy()[dev_ii][dev_b] = 0;
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
              for (auto& a_dep : this->all_deps) {
                a_dep->size = t_op->get_size()/i_paths;
                a_dep->of_s = a_dep->of_s*a_dep->size;
                a_dep->of_d = a_dep->of_d*a_dep->size;
                // printf("Orig: %d Dest: %d Size: %d Offs: %d\n", c_dep->orig, c_dep->dest, c_dep->size, c_dep->offs);
                dev_i++;
              }

              for (auto& a_mem : this->all_meminfo) {
                if (a_mem->node_id != dev_a && a_mem->node_id != dev_b) {
                  
                  a_mem->size = t_op->get_size()/i_paths * sizeof(int); // ToDo: MEM integers only
                }
              }
              
              // Size based on bandwidth
              // ******************** HERE ******************** //


            break; // MXF

            // Distant Vector - D2D
            case DVT:
              printf("Using DVT\n"); //ToBeDeleted
              // Find multiple routes that increase throughput
              // Single sink
              // dev_a = *(t_op->get_orig())->begin();
              dev_b = *(t_op->get_dest())->begin();
              i_paths = 0;

              // ToDo: check numa affinty 

              // Find direct path between origin / destination
              if (t_arch.get_devmx_cpy()[dev_a][dev_b]>0) {
                printf("Valid O/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, t_arch.get_devmx_cpy()[dev_a][dev_b]); //ToBeDeleted
                all_meminfo.push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
                all_meminfo.push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
                // op_deps = new OperationDependence(0, 0, dev_a, dev_b, 0, 0, i_paths);
                // all_deps.push_back(op_deps);
                all_deps.push_back(new OperationDependence(0, 0, dev_a, dev_b, t_arch.node_id[dev_a], t_arch.node_id[dev_b], 0, 0, 0, i_paths));
                i_paths++;
                // Remove link
                t_arch.get_devmx_cpy()[dev_a][dev_b] = 0;
                t_arch.get_devmx_cpy()[dev_b][dev_a] = 0;
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
                  for (dev_i = 0; dev_i < t_arch.get_nnod(); ++dev_i) {
                    // printf("Testing Link %d to %d \n", dev_i, dev_ii);
                    if (dev_i != dev_ii && t_arch.get_devmx_cpy()[dev_i][dev_ii] > 0) {
                      // Add link to list
                      printf("Creating II Link %d to %d bw: %d, parent %d\n", dev_i, dev_ii, t_arch.get_devmx_cpy()[dev_i][dev_ii], (v_paths.at(i_link-1))->n_id);
                      iop_path = new op_path(dev_i, static_cast< float >(t_arch.get_devmx_cpy()[dev_i][dev_ii]), i_hops, v_paths.at(i_link-1));
                      v_paths.push_back(iop_path);
                      t_arch.get_devmx_cpy()[dev_i][dev_ii] = 0;
                      t_arch.get_devmx_cpy()[dev_ii][dev_i] = 0;
                      ++n_link;
                      if (dev_i == dev_a) {
                        // Valid link to origin
                        printf("Valid Link %d to %d latency: %f\n", dev_i, dev_ii, v_paths.at(i_link)->p_lat);
                        vi_paths.push_back(iop_path);
                      }
                    }
                    // printf("Testing H/D Orig: %d Dest: %d Value: %d\n", dev_i, dev_b, t_arch.get_devmx_cpy()[dev_i][dev_b]); //ToBeDeleted
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
                    all_deps.push_back(new OperationDependence(iop_path->n_hops-1, 0, iop_path->n_id, iop_path->o_id, t_arch.node_id[iop_path->n_id], t_arch.node_id[iop_path->o_id], 0, 0, 0, i_paths));
                    // Define memory information
                    printf("[min] Path from %d to %d latency: %f\n", iop_path->n_id, iop_path->o_id, iop_path->p_lat);
                    if (iop_path->n_id != dev_a) all_meminfo.push_back(new mem_info{iop_path->n_id, 0});
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
              for (auto& a_dep : this->all_deps) {
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
              
              for (auto& a_mem : this->all_meminfo){
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
          // op_orig = *t_op->get_orig();
          // // Copy destination vector
          // op_dest = *t_op->get_dest();
          
          // // Define memory information
          // // Setup memory - Origin
          // for (auto& dev_a : op_orig)
          //   all_meminfo.push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
          // // Setup memory - Dest
          // for (auto& dev_b : op_dest)
          //   all_meminfo.push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
          
          switch (t_op->get_mhtd())
          {
            // Peer to Peer - Broadcast
            case P2P:
              printf("Using P2P\n");
          //     dev_a = *(t_op->get_orig())->begin();
          //     // all_meminfo.push_back(new mem_info{dev_a, sizeof(int) * t_op->get_size()});
          //     for (auto dev_b : *t_op->get_dest()) {
          //       // Check direct path P2P
          //       if (t_arch.get_devmx_cpy()[dev_a][dev_b]>0) {
          //         // ToDo: Remove available link?
          //         printf("Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, t_arch.get_devmx_cpy()[dev_a][dev_b]);
          //         all_deps.push_back(new OperationDependence(0, 0, dev_a, dev_b, t_arch.node_id[dev_a], t_arch.node_id[dev_b], t_op->get_size(), 0, 0, 0));
          //         // all_meminfo.push_back(new mem_info{dev_b, sizeof(int) * t_op->get_size()});
          //       } else if (ind_p){
          //         // ToDo: Validate indirect path (-1)?
          //         printf("Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, t_arch.get_devmx_cpy()[dev_a][dev_b]);
          //         all_deps.push_back(new OperationDependence(0, 0, dev_a, dev_b, t_arch.node_id[dev_a], t_arch.node_id[dev_b], t_op->get_size(), 0, 0, 0)); 
          //       } else {
          //         printf("Invalid H/D combination\n");
          //       }
          //     }
              // return 0;
            break; // P2P

            // Max-Flow - Broadcast
            case MXF:
              printf("Using MXF\n");
              // Find paths with bigher bandwith
              // 2-Hop - Single source - Multi-sink

              // i_paths = 0;
              // p_done = 0;
              // // n_hops = t_arch.get_nnod() - 2; // max number of hops
              // // ToDo: check numa affinty 
              
              // while (!p_done) {
              //   max_bw = 0;
              //   // Find nodes with highest connectivity
              //   for (auto& dev_a : op_orig) {
              //     // Find max bandwidth
              //     for (auto& dev_b : op_dest) {
              //       if (t_arch.get_devmx_cpy()[dev_a][dev_b]>max_bw){
              //         // Set max bandwidth
              //         max_bw = t_arch.get_devmx_cpy()[dev_a][dev_b];
              //         dev_i = dev_a;
              //         dev_ii = dev_b;
              //       }
              //     }
              //   }

              //   if (max_bw > 0) {
              //     // Generate dependencies
              //     if (dev_i == *(t_op->get_orig())->begin()){
              //       all_deps.push_back(new OperationDependence(0, 0, dev_i, dev_ii, t_arch.node_id[dev_i], t_arch.node_id[dev_ii], t_op->get_size(), 0, 0, i_paths));
              //     } else {
              //       all_deps.push_back(new OperationDependence(1, 0, dev_i, dev_ii, t_arch.node_id[dev_i], t_arch.node_id[dev_ii], t_op->get_size(), 0, 0, i_paths));
              //     }
              //     // Remove links
              //     printf("[AUST:] Valid H/D Orig: %d Dest: %d MaxBW: %d i_paths: %d\n", dev_i, dev_ii, max_bw, i_paths);
              //     t_arch.get_devmx_cpy()[dev_i][dev_ii] = 0;
              //     t_arch.get_devmx_cpy()[dev_ii][dev_i] = 0;
              //     op_dest.erase(std::remove(op_dest.begin(), op_dest.end(), dev_ii), op_dest.end());
              //     op_orig.push_back(dev_ii);
              //     i_paths++;
              //   } else {
              //     printf("no more valid paths\n");
              //     p_done=1;
              //   }
              //   // Check if all destinations have been reached
              //   if(op_dest.empty()){
              //     p_done = 1;
              //     printf("all done \n");
              //   }
              // }


            break; // MXF

            // Distant Vector - Broadcast
            case DVT:
              printf("Using DVT\n");
              // p_done = 0;
              // dev_i = *(t_op->get_orig())->begin();
              // i_hops = 0;
              // printf("Creating orig %d \n", dev_i);
              // // Add origin
              // v_paths.push_back(new op_path(dev_i, 0, i_hops));
              // i_hops = 1;
              // i_link = 1;
              // while (!p_done) {
              //   // Find connecting paths
              //   for (dev_ii = 0; dev_ii < t_arch.get_nnod(); ++dev_ii) {
              //     // Check if dev is destination
              //     if (t_arch.get_devmx_cpy()[dev_i][dev_ii] > 0)  {
              //       printf("Creating II Link %d to %d bw: %d, parent %d\n", dev_i, dev_i, t_arch.get_devmx_cpy()[dev_i][dev_ii], (v_paths.at(i_link-1))->n_id);
              //       iop_path = new op_path(dev_i, static_cast<float>(t_arch.get_devmx_cpy()[dev_i][dev_ii]), i_hops, v_paths.at(i_link-1));
              //       v_paths.push_back(iop_path);
              //       t_arch.get_devmx_cpy()[dev_i][dev_ii] = 0;
              //       t_arch.get_devmx_cpy()[dev_ii][dev_i] = 0;
              //       ++n_link;
              //       for (auto& dev_b : op_dest){
              //         if (dev_ii == dev_b){
              //           // Destination reached 
              //           // Check for existing path
              //           f_new = 1;
              //           auto a_path = vi_paths.begin(); 
              //           while (a_path != vi_paths.end()){
              //             if ((*a_path)->n_id == dev_ii && (*a_path)->p_lat > iop_path->p_lat) {
              //               printf("Lower latency - Existing destination %d\n", dev_ii);
              //               printf("Valid Link %d to %d latency: %f\n", dev_i, dev_b, v_paths.at(i_link)->p_lat);
              //               a_path = vi_paths.erase(a_path);
              //               vi_paths.push_back(iop_path);
              //               f_new = 0;
              //             } else {
              //               a_path++;
              //             }
              //           }
              //           if (f_new == 1) {
              //             printf("Valid Link %d to %d latency: %f\n", dev_i, dev_b, v_paths.at(i_link)->p_lat);
              //             vi_paths.push_back(iop_path);
              //           }
              //         }
              //       }
              //     }
              //   }
              // // Check ranges here <<<<<< 
              //       printf("n_link=  %d i_link= %d\n", n_link, i_link);
              //   if (n_link > i_link) {
              //     // Assign origin from list of valid links
              //     dev_i = v_paths.at(i_link)->n_id;
              //     // printf("New Node %d \n", dev_ii);
              //     i_hops = v_paths.at(i_link)->n_hops + 1;
              //     ++i_link;
              //   } else if (i_hops >= m_hops) {
              //     printf(">>>>> Max Hops <<<<<\n");
              //     p_done = 1;
              //   } else if (vi_paths.size() == op_dest.size()) {
              //     // All paths
              //     printf("All paths \n");
              //     p_done = 1;
              //   }else {
              //     // No more available links 
              //     printf("No More Valid Links\n");
              //     p_done = 1;
              //   }

              //       // Check if device is origin dev
              //         //  
              //         // Add destination to origins and remove destination
              //         // op_dest.erase(std::remove(op_dest.begin(), op_dest.end(), dev_ii), op_dest.end());
              //         // op_orig.push_back(dev_ii);

              //   // for (auto& dev_i : op_orig) {
              //   // }
              // }
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

}