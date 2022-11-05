#include <iostream>
#include <fstream>
#include <string>
#include<bits/stdc++.h>
// #include <sstream>

using namespace std;

void get_topo(ifstream arch_f);

int main()
{
  int n_dev = 0, n_net = 0, line_n = 0, word_n = 0, dev_a = 0, dev_b =0, n_nodes = 1, n_hst = 2, n_nma = 2, n_core = 0, nma_x;
  int n_org, n_dst;
  int **hst_mx, **dev_mx, **hst_mx_cpy, **dev_mx_cpy, *d_org, *d_dst; 

  enum c_ops { D2D, BRC, SCT, GAT, RED } c_op;
  enum h_mth { H2D, DVT, MXF } h_mt; // Host-to-Device, Distant Vector, Max-Flow

  string line, word, word_b;
  
  ifstream t_dgx ("topo_dgx");
  ifstream t_smx ("topo_smx");
  stringstream iss_a, iss_b;

  struct numa_dom
  {
    int h_i[4];
    int conf = 0;
  } *numa_d, numa_t;

  struct cops_dep
  {
    int orig;
    int dest;
    int size;
    int offs;
    int done;
    cops_dep * p_dep;
  } *op_deps;

  numa_d = (numa_dom*) malloc((n_nma) * sizeof(numa_dom));
  
  if (t_dgx.is_open())
  {
    while (!t_dgx.eof())
    {
      getline (t_dgx, line);
      iss_a.clear();
      iss_a << line;
      word_n = 0;
      
      // Get all words in a line
      while (iss_a >> word)
      {
        if (line_n == 0)
        {
          if (word.compare(0,3,"GPU") == 0) ++n_dev;
          else if (word.compare(0,3,"mlx") == 0) ++n_net;
        }
        else if (word_n == 0 && word.compare(0,3,"GPU") == 0)
        {
          // Add info to Interconnectivity Matrix

          for (dev_b = 0; dev_b < n_dev; ++dev_b)
          {
            iss_a >> word;
            // cout << "here" << dev_b << n_dev << endl;
            if (word.compare("X") == 0) dev_mx[dev_a + n_hst][dev_b + n_hst] = 0;
            else if (word.compare("NV1") == 0) dev_mx[dev_a + n_hst][dev_b + n_hst] = 25; // Based on experiments 22 GB/s
            else if (word.compare("NV2") == 0) dev_mx[dev_a + n_hst][dev_b + n_hst] = 50; // Based on experiments 45 GB/s
            else if (word.compare("SYS") == 0) dev_mx[dev_a + n_hst][dev_b + n_hst] = 6; // Based on experiments 6 GB/s
            else dev_mx[dev_a + n_hst][dev_b + n_hst] = -1;
          }
          // Skip network information // ***** Future_Work *****
          for (int i = 0; i <= n_net; ++i) iss_a >> word;
          
          // Get CPU Affinity
          // Replace separators with spaces
          for(int i = 0; i < word.length(); ++i) if(word[i]==',' || word[i]=='-') word[i] = *strdup(" ");

          iss_b.clear();
          iss_b << word;
          n_core = 0;
          while (iss_b >> word_b || n_core==3) {
            if (stringstream(word_b) >> numa_t.h_i[n_core]) ++n_core;
            //   cout << n_core << " ";
          }

          // Add numa information
          iss_a >> word;
          stringstream(word) >> nma_x;
          // Check if new information and copy
          if (!numa_d[nma_x].conf){
            numa_d[nma_x] = numa_t;
            numa_d[nma_x].conf = 1;
          }

          // Add Host to Device information 
          // cout << "here " << dev_a << " " << nma_x << " " << n_dev << endl;
          if(nma_x)
          {
            dev_mx[0][dev_a + n_hst]=6;
            dev_mx[1][dev_a + n_hst]=10;
            dev_mx[dev_a + n_hst][0]=6;
            dev_mx[dev_a + n_hst][1]=10;
          }
          else
          {
            dev_mx[0][dev_a + n_hst]=10;
            dev_mx[1][dev_a + n_hst]=6;
            dev_mx[dev_a + n_hst][0]=10;
            dev_mx[dev_a + n_hst][1]=6;
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
        dev_mx = (int**) malloc((n_dev + n_hst) * sizeof(int*));
        for (dev_a = 0; dev_a < (n_dev + n_hst); ++dev_a)
          dev_mx[dev_a] = (int*)malloc((n_dev + n_hst) * sizeof(int));
        
        // hst_mx = (int**) malloc(n_dev * sizeof(int*));
        // for (dev_a = 0; dev_a < n_dev; ++dev_a)
        //   hst_mx[dev_a] = (int*)malloc(n_hst * sizeof(int));

        // Initialize variable dev_a
        dev_a = 0;
      }
      ++line_n;
    }
    t_dgx.close();
  }
  else cout << "Unable to open file"; 


  // Fill Host to Host information
  dev_mx[0][1]=40; // Theoretically 38.4 GB/s
  dev_mx[1][0]=40; // Theoretically 38.4 GB/s


  printf("DGX\n");
  cout << "no. Devices: " << n_dev << "\n";
  cout << "no. Ext Networks: " << n_net << "\n";

  // Print Interconnectivity Matrix
  // printf("H2D Interconnectivity Matrix\n");
  // for (dev_a = 0; dev_a < n_hst; ++dev_a) {
  //   for (dev_b = 0; dev_b < n_dev; ++dev_b)
  //     // printf("%d ", dev_mx[dev_a * n_dev + dev_b]);
  //     printf("%2d ", hst_mx[dev_b][dev_a]);
  //   printf("\n");
  // }
  printf("D2D Interconnectivity Matrix\n");
  for (dev_a = 0; dev_a < n_dev + n_hst; ++dev_a) {
    for (dev_b = 0; dev_b < n_dev + n_hst; ++dev_b)
      printf("%2d ", dev_mx[dev_a][dev_b]);
    printf("\n");
  }

  // Print Numa Cores
  printf("Numa Cores\n");
  for (dev_a = 0; dev_a < n_nma; ++dev_a) {
    printf("Numa: %d Cores: ",  dev_a);
    for (dev_b = 0; dev_b < 4; ++dev_b)
      printf("%d ",  numa_d[dev_a].h_i[dev_b]);
    printf("\n");
  }
  
  // Inputs
  // Set up origin, dest, operation and method
  n_org = 1;
  d_org = (int*) malloc((n_org) * sizeof(int*));

  n_dst = 1;
  d_dst = (int*) malloc((n_dst) * sizeof(int*));

  d_org[0]=0;
  d_dst[0]=2;

  c_op = D2D;
  h_mt = H2D;
           
  // Strategy
  switch (c_op)
  {
    case D2D:
      printf("D2D \n");
      switch (h_mt)
      {
        case H2D:
        // Unidirectional one-to-one transfer between devices
        printf("H2D\n");
        for (dev_a = 0; dev_a < n_org; ++dev_a)
        {
          for (dev_b = 0; dev_b < n_dst; ++dev_b)
          {
            // Check direct path D2D
            if (dev_mx[d_org[dev_a]][d_dst[dev_b]]>0) 
            {
              //Remove available link
              printf("Valid H/D combination\n");
            }
            else
            {
              printf("Invalid H/D combination\n");
            }
          }
        }

        break; // H2D

        default:
          printf("Invalid Method\n");
          return 1;
      }
    break; // D2D

    case BRC:
      printf("Unsupported H/D combination\n");
      printf("\n");
      switch (h_mt)
      {
        case H2D:
        break; // H2D

        default:
          printf("Invalid Method\n");
          return 1;
      }
    break; // BRC

    case SCT:
      printf("Unsupported H/D combination\n");
      printf("\n");
    break; // 
    
    case GAT:
      printf("Unsupported H/D combination\n");
      printf("\n");
    break; // 

    case RED:
      printf("Unsupported H/D combination\n");
      printf("\n");

    break; // RED

    default:
      printf("Invalid Operation\n");
      return 1;
      
      

    // case DVT:
    //   printf("H2D\n");

    // break; // DVT

    // case MXF:
    //   printf("\n");

    // break; // MXF

    // default:
    //   printf("Invalid Method\n");
    //   return 1;
  }
  



  // Outputs
  // Dependencies
  // Sizes
  // Offsets

  // 
  free(dev_mx);
  // free(hst_mx);
  free(d_org);
  free(d_dst);

  return 0;
}


  // int n_dev = 0, n_net = 0;
  // string line, word;
  
  // ifstream t_dgx ("topo_dgx");
  // ifstream t_smx ("topo_smx");
  
  // if (t_dgx.is_open())
  // {
  //   while ( getline (t_dgx,line,'\t') )
  //   {
  //     if (line.compare(0,3,"GPU") == 0){
  //       // cout << "GPU" << '\n';
  //       // cout << line << '\n';
  //       ++n_dev;
  //     }else if (line.compare(0,3,"mlx") == 0){
  //       ++n_net;
  //     }else{
  //       // cout << "other" << '\n';
  //       cout << line << line.compare(0,3,"GPU") <<'\n';
  //     }
  //   }
  //   t_dgx.close();
  // }else cout << "Unable to open file"; 

  // cout << "no. Devices: " << n_dev << "\n";
  // cout << "no. Networks: " << n_net << "\n";

  // return 0;