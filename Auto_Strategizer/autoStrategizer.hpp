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
    }

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

    class AutoStrategizer
    {
    private:
        int n_dev = 0, n_net = 0, dev_a = 0, dev_b = 0, dev_i, dev_ii, n_nodes = 1, n_hst = 2, n_nma = 2, arr_len;
        int n_org, n_dst, f_sts;
        int **hst_mx, **dev_mx, **hst_mx_cpy, **dev_mx_cpy;

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
