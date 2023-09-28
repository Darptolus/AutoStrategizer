/**
 * @brief Library to create smart memory movements
 *
 * Based on the topology of the system, this library allows to
 * define a dependence graph of data movements that maximizes network
 * utilization, minimizing memory movement type
 *
 * @copyright Diego Andres Roa Perdomo
 */

#include <iostream>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
#include <vector>
#include <omp.h>
#include <numa.h>
#include "autoStrategizer_tools.hpp"

namespace AutoStrategizer
{
  using device_id = uint64_t;

  /**
   * @brief Options for the print Topo
   *
   * CLI is command line interface
   * and DOT is .DOT file to use graphviz to get an image
   *
   */
  enum printMode
  {
      CLI,
      DOT
  };

  /**
   * @brief Define the operation to perform
   *
   * D2D: Device 2 Device
   * BRC: Broadcast
   * SCT: Scatter
   * GAT: Gather
   * RED: Reduction
   *
   */
  enum operation
  {
    D2D,
    BRC,
    SCT,
    GAT,
    RED
  };

  /**
   * @brief Specifies algorithm to use to create strategy
   *
   * H2D: Host to Device
   * DVT: Distance Vector
   * MXF: Max Flow
   */
  enum algorithm
  {
    P2P,
    MXF,
    DVT
  };

  class dataMovement
  {
  private:
      /// @brief ID of the origin node where data lives
      device_id origin;
      /// @brief ID of the destination node where data lives
      device_id dest;
      /// @brief Offset from local buffer at the origin
      uint64_t offset;
      /// @brief Size of data movement
      uint64_t size;

      /// @brief Contains all the output dependencies that form the graph
      std::vector<dataMovement *> outDeps;

  public:
  };

  /**
   * @brief
   *
   */
  class depGraph
  {
    private:
      std::vector<dataMovement *> headNodes;
  };

  struct numa_dom
  {
    int h_i[4];
    int conf = 0;
  };

  class Architecture
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
    int **dev_mx_cpy; // Interconnectivity Matrix copy
  public:
    Architecture();
    ~Architecture();
    int * node_id;
    int get_nhst(){return n_hst;}
    int get_ndev(){return n_dev;}
    int get_nnod(){return n_nod;}
    int get_nnuma(){return n_nma;}
    int get_nnet(){return n_net;}
    int ** get_devmx(){return dev_mx;}
    int ** get_devmx_cpy(){return dev_mx_cpy;}
    numa_dom * get_numa(){return numa_d;}
    void set_ndev(int a_ndev){
      n_dev = a_ndev;
      n_nod = n_dev + n_hst;
      node_id = (int*) malloc(n_nod * sizeof(int));
      // Allocate memory for Interconnectivity Matrix 
      dev_mx = (int**) malloc((n_nod) * sizeof(int*));
      dev_mx_cpy = (int**) malloc((n_nod) * sizeof(int*));
      // printf("host_id: %d\n", omp_get_initial_device());
      for (int dev = 0; dev < (n_nod); ++dev)
        {
          dev_mx[dev] = (int*)malloc((n_nod) * sizeof(int));
          dev_mx_cpy[dev] = (int*)malloc((n_nod) * sizeof(int));
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

  class CollectiveOperation
  {
  private:
    std::vector<int> o_orig; // Origin devices (node_num)
    std::vector<int> o_dest; // Destination devides (node_num)
    int size; // ToDo: Change to size_t
    operation cop; // Collective operation
    algorithm mht; // Heuristic Method
    int n_orig; // Number of origin devices
    int n_dest; // Number of destination devices
  public:
    CollectiveOperation(){};
    ~CollectiveOperation(){};
    void add_origin(int a_orig){
      o_orig.push_back(a_orig);
    };
    void add_destination(int a_dest){
      o_dest.push_back(a_dest);
    };
    void set_size(int a_size){size = a_size;};
    int get_size(){return size;};
    void set_coop(operation a_cop){cop = a_cop;};
    void set_mhtd(algorithm a_mht){mht = a_mht;};
    int get_norig(){return o_orig.size();};
    int get_ndest(){return o_dest.size();};
    operation get_coop(){return cop;};
    algorithm get_mhtd(){return mht;};
    std::vector<int>* get_orig(){return &o_orig;};
    std::vector<int>* get_dest(){return &o_dest;};
  };

  typedef std::vector<CollectiveOperation*> CO_vector;

  // Communication pattern dependencies
  class OperationDependence
  {
  private:
    OperationDependence * p_dep;
  public:
    // cops_dep(int a_deps, int a_done, int a_orig, int a_dest, int a_size, int a_oof, int a_dof):
    // deps(a_deps), done(a_done), orig(a_orig), dest(a_dest), size(a_size), of_s(a_oof), of_d(a_dof){};
    OperationDependence(int a_deps, int a_done, int a_orig, int a_dest, int a_oid, int a_did, int a_size, int a_oof, int a_dof, int a_ipth):
    deps(a_deps), done(a_done), orig(a_orig), dest(a_dest), o_id(a_oid), d_id(a_did), size(a_size), of_s(a_oof), of_d(a_dof), ipth(a_ipth){};
    ~OperationDependence(){};
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
  };

  // All deps
  typedef std::vector<OperationDependence*> OD_vector;

  struct mem_info
  {
    int node_id;
    size_t size;
  };

  typedef std::vector<mem_info*> MI_vector;

  class AutoStrategizer
  {
  private:
    int dev_a = 0, dev_b = 0, dev_i, dev_ii, n_nodes = 1, arr_len;
    int n_org, n_dst, f_sts;
    Architecture t_arch;
    CO_vector all_ops;  // Operation Definition
    OD_vector all_deps; // Operation Depencences
    MI_vector all_meminfo; // Memory information

  public:
    AutoStrategizer(std::string architectureFile);
    /**
     * @brief Create an strategy for a move operation
     *
     * This function returns a dependence graph of data movements.
     * Each data movement represents a single D2D move from an origin to a destination
     * of a given size, with a given offset.
     *
     * @param source List of input devices where data lives
     * @param destination List of output devices where data must go
     * @param operation Gather, Scatter, reduction, broadcast, D2D
     * @param alg Algorithm
     * @param size Size of data per operation
     */
    // depGraph getStrategy(device_id *source, device_id *destination, operation operation, algorithm alg, uint64_t size);

    /**
     * @brief Prints the current topology
     *
     * @param mode The printing mode
     *
     */
    void **mem_ptr; // Memory pointers
    
    void printTopo(printMode mode);
    void printTopo_cpy(printMode mode);
    void printCO(printMode mode);

    void addCO(CollectiveOperation * a_CO);
    void copy_mx();

    void ops_deps();

    void auto_malloc(int skip_od);
    void auto_mfree(int skip_od);

    OD_vector * getDeps();
    CollectiveOperation * getOP();
    MI_vector * getMI();
    int get_node_id(int node_num);

    void ** get_memptr();
    int get_nnod();
  };

  

}
