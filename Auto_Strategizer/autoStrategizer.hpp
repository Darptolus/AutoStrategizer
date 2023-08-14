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
      H2D,
      DVT,
      MXF
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

  

  class AutoStrategizer
  {
  private:
    int dev_a = 0, dev_b = 0, dev_i, dev_ii, n_nodes = 1, arr_len;
    int n_org, n_dst, f_sts;
    Architecture t_arch;

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
    depGraph getStrategy(device_id *source, device_id *destination, operation operation, algorithm alg, uint64_t size);

    /**
     * @brief Prints the current topology
     *
     * @param mode The printing mode
     *
     */
    void printTopo(printMode mode);

  };

  

}
