#include "autoStrategizer.hpp"

namespace AutoStrategizer
{
    int get_topo(std::ifstream *arch_f, int n_hst_v, int ***dev_mx_p, int *n_dev, int *n_net)
    {
        int line_n = 0, word_n = 0, n_dev_v = 0, n_net_v = 0, dev_a = 0, dev_b = 0, n_core = 0, nma_x = 0;
        std::string line, word, word_b;
        std::stringstream iss_a, iss_b;

        if (arch_f->is_open())
        {
            while (!arch_f->eof())
            {
                getline(*arch_f, line);
                iss_a.clear();
                iss_a << line;
                word_n = 0;

                // Get all words in a line
                while (iss_a >> word)
                {
                    if (line_n == 0)
                    {
                        if (word.compare(0, 3, "GPU") == 0)
                            ++n_dev_v;
                        else if (word.compare(0, 3, "mlx") == 0)
                            ++n_net_v;
                    }
                    else if (word_n == 0 && word.compare(0, 3, "GPU") == 0)
                    {
                        // Add info to Interconnectivity Matrix
                        for (dev_b = 0; dev_b < n_dev_v; ++dev_b)
                        {
                            iss_a >> word;
                            if (word.compare("X") == 0)
                                (*dev_mx_p)[dev_a + n_hst_v][dev_b + n_hst_v] = 0;
                            else if (word.compare("NV1") == 0)
                                (*dev_mx_p)[dev_a + n_hst_v][dev_b + n_hst_v] = 25; // Based on experiments 22 GB/s
                            else if (word.compare("NV2") == 0)
                                (*dev_mx_p)[dev_a + n_hst_v][dev_b + n_hst_v] = 50; // Based on experiments 45 GB/s
                            else if (word.compare("SYS") == 0)
                                (*dev_mx_p)[dev_a + n_hst_v][dev_b + n_hst_v] = 0; // No P2P - Based on experiments 6 GB/s
                            else
                                (*dev_mx_p)[dev_a + n_hst_v][dev_b + n_hst_v] = -1;
                        }
                        // Skip network information // ***** Future_Work *****
                        for (int i = 0; i <= n_net_v; ++i)
                            iss_a >> word;

                        // Get CPU Affinity
                        // Replace separators with spaces
                        for (int i = 0; i < word.length(); ++i)
                            if (word[i] == ',' || word[i] == '-')
                                word[i] = *strdup(" ");

                        iss_b.clear();
                        iss_b << word;
                        n_core = 0;
                        while (iss_b >> word_b || n_core == 3)
                        {
                            if (stringstream(word_b) >> numa_t.h_i[n_core])
                                ++n_core;
                        }

                        // Add numa information
                        iss_a >> word;
                        std::stringstream(word) >> nma_x;
                        // Check if new information and copy
                        if (!numa_d[nma_x].conf)
                        {
                            numa_d[nma_x] = numa_t;
                            numa_d[nma_x].conf = 1;
                        }

                        // Add Host to Device information // ToDo: Parametrize this
                        if (nma_x)
                        {
                            (*dev_mx_p)[0][dev_a + n_hst_v] = 6;
                            (*dev_mx_p)[1][dev_a + n_hst_v] = 10;
                            (*dev_mx_p)[dev_a + n_hst_v][0] = 6;
                            (*dev_mx_p)[dev_a + n_hst_v][1] = 10;
                        }
                        else
                        {
                            (*dev_mx_p)[0][dev_a + n_hst_v] = 10;
                            (*dev_mx_p)[1][dev_a + n_hst_v] = 6;
                            (*dev_mx_p)[dev_a + n_hst_v][0] = 10;
                            (*dev_mx_p)[dev_a + n_hst_v][1] = 6;
                        }

                        // cout << word << endl;
                        ++dev_a;
                    }

                    // else{
                    //   // cout << "other" << '\n';
                    //   // cout << word << " " << word.compare(0,3,"GPU") << endl;
                    // }

                } // End While: Get all words in a line
                if (line_n == 0)
                {
                    // Allocate interconnectivity matrix
                    *dev_mx_p = (int **)malloc((n_dev_v + n_hst_v) * sizeof(int *));

                    // printf("dev_mx_p address: %p\n", (*dev_mx_p));

                    for (dev_a = 0; dev_a < (n_dev_v + n_hst_v); ++dev_a)
                        (*dev_mx_p)[dev_a] = (int *)malloc((n_dev_v + n_hst_v) * sizeof(int));

                    // hst_mx = (int**) malloc(n_dev * sizeof(int*));
                    // for (dev_a = 0; dev_a < n_dev; ++dev_a)
                    //   hst_mx[dev_a] = (int*)malloc(n_hst_v * sizeof(int));

                    // Initialize variable dev_a
                    dev_a = 0;
                }
                ++line_n;
            }
            arch_f->close();

            *n_net = n_net_v;
            *n_dev = n_dev_v;
            return 0;
        }
        else
        {
            cout << "Unable to open file";
            return 1;
        }
    }

    AutoStrategizer::AutoStrategizer(std::string architectureFile) : n_dev(0),
                                                                     n_net(0),
                                                                     dev_a(0),
                                                                     dev_b(0),
                                                                     dev_i(0),
                                                                     dev_ii(0),
                                                                     n_nodes(0),
                                                                     n_hst(0),
                                                                     n_nma(0),
                                                                     arr_len(0),
                                                                     n_org(0),
                                                                     n_dst(0),
                                                                     f_sts(0),
                                                                     hst_mx(nullptr),
                                                                     dev_mx(nullptr),
                                                                     hst_mx_cpy(nullptr),
                                                                     dev_mx_cpy(nullptr);
    {
        std::ifstream t_dgx(architectureFile);

        // numa_d = (numa_dom *)malloc((n_nma) * sizeof(numa_dom));

        // Get Topology
        if (get_topo(&t_dgx, n_hst, &dev_mx, &n_dev, &n_net))
            printf("Unable to get topology\n");

        // Fill Host to Host information
        dev_mx[0][1] = 40; // Theoretically 38.4 GB/s
        dev_mx[1][0] = 40; // Theoretically 38.4 GB/s
    }
}