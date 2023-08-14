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

    // numa_d = (numa_dom *)malloc((n_nma) * sizeof(numa_dom));
  
    // Get Topology
    if (get_topo(&arch_f, &t_arch))
        std::cout << "Unable to get topology" << std::endl;

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
}