#include <iostream>
#include <fstream>
#include <string>
#include<bits/stdc++.h>
#include <vector>
// #include <sstream>

using namespace std;

struct numa_dom
{
  int h_i[4];
  int conf = 0;
} *numa_d, numa_t;

// Communications dependences
class cops_dep
{
  private:
    cops_dep * p_dep;
  public:
    cops_dep(int a_deps, int a_done, int a_orig, int a_dest, int a_size, int a_offs):
    deps(a_deps), done(a_done), orig(a_orig), dest(a_dest), size(a_size), offs(a_offs){};
    cops_dep(int a_deps, int a_done, int a_orig, int a_dest, int a_size, int a_offs, int a_ipth):
    deps(a_deps), done(a_done), orig(a_orig), dest(a_dest), size(a_size), offs(a_offs), ipth(a_ipth){};
    ~cops_dep(){};
    // Todo: Add set/get methods
    int deps;
    int done;
    int orig;
    int dest;
    int size;
    int offs;
    int ipth;
} *op_deps; // Remove

// All deps
typedef std::vector<cops_dep*> ve_deps;

ve_deps all_deps;

enum c_ops { D2D, BRC, SCT, GAT, RED };
enum h_mth { H2D, DVT, MXF }; // Host-to-Device, Distant Vector, Max-Flow

class cops_def
{
  private:
    std::vector<int> o_orig;
    std::vector<int> o_dest;
    int size;
    c_ops cop; // Collective operation
    h_mth mht; // Heuristic Method
  public:
    cops_def(){};
    // cops_def(int a_orig, int a_dest, int a_size, c_ops a_cop, h_mth a_mht):
    // orig(a_orig), dest(a_dest), size(a_size), cop(a_cop), mht(a_mht){};
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
    c_ops get_coop(){return cop;};
    h_mth get_mhtd(){return mht;};
    std::vector<int>* get_orig(){return &o_orig;};
    std::vector<int>* get_dest(){return &o_dest;};
} *def_ops; // Remove

// All Ops
typedef std::vector<cops_def*> ve_ops;

ve_ops all_ops;

int get_topo(ifstream *arch_f, int n_hst_v, int ***dev_mx_p, int *n_dev, int *n_net);
void copy_mx(int ***dev_mx_p, int ***dev_mx_cpy_p, int n_dev_v, int n_hst_v, int new_cpy);
void print_mx(int ***dev_mx_p, int n_dev_v, int n_hst_v);
void print_numa(numa_dom *numa_d, int n_nma);
int get_ops(ifstream *ops_f, ve_ops *all_ops, int n_hst, int n_dev);
void print_ops(ve_ops *all_ops);
void ops_deps(ve_ops *all_ops, ve_deps *all_deps, int ***dev_mx_cpy_p, int n_hst, int n_dev);

int main()
{
  int n_dev = 0, n_net = 0, dev_a = 0, dev_b = 0, dev_i, dev_ii, n_nodes = 1, n_hst = 2, n_nma = 2, arr_len;
  int n_org, n_dst, f_sts;
  int **hst_mx, **dev_mx, **hst_mx_cpy, **dev_mx_cpy; 

  ifstream t_dgx ("topo_dgx");
  ifstream t_smx ("topo_smx");

  ifstream c_ops ("test_ops");

  numa_d = (numa_dom*) malloc((n_nma) * sizeof(numa_dom));

  // Get Topology 
  if (get_topo(&t_dgx, n_hst, &dev_mx, &n_dev, &n_net)) printf("Unable to get topology\n");

  // Fill Host to Host information
  dev_mx[0][1]=40; // Theoretically 38.4 GB/s
  dev_mx[1][0]=40; // Theoretically 38.4 GB/s

  printf("DGX\n");
  cout << "no. Devices: " << n_dev << "\n";
  cout << "no. Ext Networks: " << n_net << "\n";

  // Print Interconnectivity Matrix
  print_mx(&dev_mx, n_dev, n_hst);
  // Print Numa Cores
  print_numa(numa_d, n_nma);

  // Create a copy Interconnectivity Matrix 
  copy_mx(&dev_mx, &dev_mx_cpy, n_dev, n_hst, 1);
  // print_mx(&dev_mx_cpy, n_dev, n_hst); // Print Copy
  
  // Inputs
  // Set up origin, dest, size, operation and method
  if (get_ops(&c_ops, &all_ops, n_hst, n_dev)) printf("Unable get ops\n");

  print_ops(&all_ops);

  ops_deps(&all_ops, &all_deps, &dev_mx_cpy, n_hst, n_dev);

  // ******************** //

  // Outputs
  // Dependencies, Sprintf("Valid H/D combination\n");izes, Offsets
  printf("Outputs\n");
  for (auto& a_dep : all_deps)
  {
    // Print the values
    cops_dep * c_dep = static_cast <cops_dep *> (a_dep);
    printf("Orig: %d Dest: %d Size: %d Offs: %d\n", c_dep->orig, c_dep->dest, c_dep->size, c_dep->offs);
    // cout << it->orig << ' ';
  }

  // printf("\n");


  // 
  free(dev_mx);
  // free(hst_mx);

  for (auto& element : all_deps) {
    delete element;
  }

  return 0;
}


int get_topo(ifstream *arch_f, int n_hst_v, int ***dev_mx_p, int *n_dev, int *n_net)
{
  int line_n = 0, word_n = 0, n_dev_v = 0, n_net_v = 0, dev_a = 0, dev_b =0, n_core = 0, nma_x = 0;
  string line, word, word_b;
  stringstream iss_a, iss_b;

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
          for (dev_b = 0; dev_b < n_dev_v; ++dev_b)
          {
            iss_a >> word;
            if (word.compare("X") == 0) (*dev_mx_p)[dev_a + n_hst_v][dev_b + n_hst_v] = 0;
            else if (word.compare("NV1") == 0) (*dev_mx_p)[dev_a + n_hst_v][dev_b + n_hst_v] = 25; // Based on experiments 22 GB/s
            else if (word.compare("NV2") == 0) (*dev_mx_p)[dev_a + n_hst_v][dev_b + n_hst_v] = 50; // Based on experiments 45 GB/s
            else if (word.compare("SYS") == 0) (*dev_mx_p)[dev_a + n_hst_v][dev_b + n_hst_v] = 0; // No P2P - Based on experiments 6 GB/s
            else (*dev_mx_p)[dev_a + n_hst_v][dev_b + n_hst_v] = -1;
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
          if (!numa_d[nma_x].conf){
            numa_d[nma_x] = numa_t;
            numa_d[nma_x].conf = 1;
          }

          // Add Host to Device information // ToDo: Parametrize this
          if(nma_x)
          {
            (*dev_mx_p)[0][dev_a + n_hst_v]=6;
            (*dev_mx_p)[1][dev_a + n_hst_v]=10;
            (*dev_mx_p)[dev_a + n_hst_v][0]=6;
            (*dev_mx_p)[dev_a + n_hst_v][1]=10;
          }
          else
          {
            (*dev_mx_p)[0][dev_a + n_hst_v]=10;
            (*dev_mx_p)[1][dev_a + n_hst_v]=6;
            (*dev_mx_p)[dev_a + n_hst_v][0]=10;
            (*dev_mx_p)[dev_a + n_hst_v][1]=6;
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
        *dev_mx_p = (int**) malloc((n_dev_v + n_hst_v) * sizeof(int*));
        
        // printf("dev_mx_p address: %p\n", (*dev_mx_p));

        for (dev_a = 0; dev_a < (n_dev_v + n_hst_v); ++dev_a)
          (*dev_mx_p)[dev_a] = (int*)malloc((n_dev_v + n_hst_v) * sizeof(int));
        
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
  else{
    cout << "Unable to open file"; 
    return 1;
  }

}


void copy_mx(int ***dev_mx_p, int ***dev_mx_cpy_p, int n_dev_v, int n_hst_v, int new_cpy)
{
  int dev_a, dev_b;
  // Check if new copy
  if (new_cpy)
  {
    // printf("Allocating memory for copy\n");
    *dev_mx_cpy_p = (int**) malloc((n_dev_v + n_hst_v) * sizeof(int*));
    for (dev_a = 0; dev_a < (n_dev_v + n_hst_v); ++dev_a)
      (*dev_mx_cpy_p)[dev_a] = (int*)malloc((n_dev_v + n_hst_v) * sizeof(int));
  }
  // Copy device matrix information
  for (dev_a = 0; dev_a < n_dev_v + n_hst_v; ++dev_a)
    for (dev_b = 0; dev_b < n_dev_v + n_hst_v; ++dev_b)
      (*dev_mx_cpy_p)[dev_a][dev_b] = (*dev_mx_p)[dev_a][dev_b];
}

void print_mx(int ***dev_mx_p, int n_dev_v, int n_hst_v)
{
  int dev_a, dev_b;
  printf("Interconnectivity Matrix\n");
  for (dev_a = 0; dev_a < n_dev_v + n_hst_v; ++dev_a) {
    for (dev_b = 0; dev_b < n_dev_v + n_hst_v; ++dev_b)
      printf("%2d ", (*dev_mx_p)[dev_a][dev_b]);
    printf("\n");
  }
}

  // printf("dev_mx address: %p\n", dev_mx);

void print_numa(numa_dom *numa_d, int n_nma)
{
  int dev_a, dev_b;
  // Print Numa Cores
  printf("Numa Cores\n");
  for (dev_a = 0; dev_a < n_nma; ++dev_a) {
    printf("Numa: %d Cores: ",  dev_a);
    for (dev_b = 0; dev_b < 4; ++dev_b)
      printf("%d ",  numa_d[dev_a].h_i[dev_b]);
    printf("\n");
  }
}

int get_ops(ifstream *ops_f, ve_ops *all_ops, int n_hst, int n_dev)
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
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              word.erase(0,1);
              stringstream(word) >> d_id;
              a_ops->add_orig(d_id);
            }
            else if (word.compare(0,1,"D") == 0)
            {
              // Device
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              word.erase(0,1);
              stringstream(word) >> d_id;
              a_ops->add_orig(d_id + n_hst); // Add No. Hosts
            }
            else if (word.compare(0,2,"AH") == 0)
            {
              // All Devices
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              for(i_dev = 0; i_dev<n_hst; ++i_dev) a_ops->add_orig(i_dev); // Add No. Hosts
            }
            else if (word.compare(0,2,"AD") == 0)
            {
              // All Devices
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              for(i_dev = 0; i_dev<n_dev; ++i_dev) a_ops->add_orig(i_dev + n_hst); // Add No. Hosts
            }
            iss_a >> word;
          }
          if (word.length() == 3 && op_set == 0)
          {
            if (word.compare(0,3,"D2D") == 0)
            {
              a_ops->set_coop(D2D);
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              op_set = 1;
            }
            else if (word.compare(0,3,"BRC") == 0)
            {
              a_ops->set_coop(BRC);
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              op_set = 1;
            }
            else if (word.compare(0,3,"SCT") == 0)
            {
              a_ops->set_coop(SCT);
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              op_set = 1;
            }
            else if (word.compare(0,3,"GAT") == 0)
            {
              a_ops->set_coop(GAT);
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              op_set = 1;
            }
            else if (word.compare(0,3,"RED") == 0)
            {
              a_ops->set_coop(RED);
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
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
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              word.erase(0,1);
              stringstream(word) >> d_id;
              a_ops->add_dest(d_id);
            }
            else if (word.compare(0,1,"D") == 0 && op_set == 1)
            {
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              word.erase(0,1);
              stringstream(word) >> d_id;
              a_ops->add_dest(d_id + n_hst); // Add No. Hosts
            }
            else if (word.compare(0,2,"AH") == 0)
            {
              // All Devices
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              for(i_dev = 0; i_dev<n_hst; ++i_dev) a_ops->add_dest(i_dev); // Add No. Hosts
            }
            else if (word.compare(0,2,"AD") == 0)
            {
              // All Devices
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
              for(i_dev = 0; i_dev<n_dev; ++i_dev) a_ops->add_dest(i_dev + n_hst); // Add No. Hosts
            }
            iss_a >> word;
          }
          if (word.length() == 3 && op_set == 1)
          {
            if (word.compare(0,3,"H2D") == 0)
            {
              a_ops->set_mhtd(H2D);
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
            }
            else if (word.compare(0,3,"DVT") == 0)
            {
              a_ops->set_mhtd(DVT);
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
            }
            else if (word.compare(0,3,"MXF") == 0)
            {
              a_ops->set_mhtd(MXF);
              cout << "[TMP:] "<< word << " " << endl; //ToBeDeleted
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

void print_ops(ve_ops *all_ops)
{
  printf("All Operations\n");
  for (auto& it : *all_ops)
  {
    // Print the values
    printf("Type: ");
    if (it->get_coop() == D2D) printf("D2D ");
    else if (it->get_coop() == BRC) printf("BRC ");
    else if (it->get_coop() == SCT) printf("SCT ");
    else if (it->get_coop() == GAT) printf("GAT ");
    else if (it->get_coop() == RED) printf("RED ");

    printf("Mthd: ");
    if (it->get_mhtd() == H2D) printf("H2D ");
    else if (it->get_mhtd() == DVT) printf("DVT ");
    else if (it->get_mhtd() == MXF) printf("MXF ");

    printf("Size: %d ", it->get_size());

    printf("Orig: ");
    for (auto element : *it->get_orig()) {
        cout << element << " ";
    }


    // printf("%d ", it->get_size());
    printf("Dest: ");
    for (auto element : *it->get_dest()) {
        cout << element << " ";
    }
    printf("\n");

    // printf("%d ", it->get_size());
    // printf("Orig: Dest: Size: %d Offs: \n", it->get_size());
    // cout << it->orig << ' ';
  } 
}

void ops_deps(ve_ops *all_ops, ve_deps *all_deps, int ***dev_mx_cpy_p,int n_hst, int n_dev)
{
  int m_paths = 3, m_hops = 5; // inputs
  int dev_a = 0, dev_b = 0, dev_i, dev_ii, i_hops, n_hops, i_paths, p_done, max_bw, min_lat, lnk_bw, i_link, n_link, h_aff;
  
  // typedef std::vector<int> a_path;

  class op_path
  {
    public:
      op_path(int a_n_id, float a_p_bwth, int a_n_hops): n_id(a_n_id), n_hops(a_n_hops){
        p_op_path = NULL;
        p_lat = 1/a_p_bwth;
        // printf("Link lat : %f \n", p_lat);
      };
      op_path(int a_n_id, int a_p_bwth, int a_n_hops, op_path * a_op_path): n_id(a_n_id), n_hops(a_n_hops), p_op_path(a_op_path){
        p_lat = p_op_path->p_lat + 1/a_p_bwth;
      };
      ~op_path(){};
      // ToDo: define set and get methods
      op_path * p_op_path;
      int n_id;
      float p_lat; // path latency
      int n_hops;
  } * iop_path;

  typedef std::vector<op_path*> op_paths;

  op_paths v_paths, vi_paths; // vector of paths

  for (auto& t_op : *all_ops)
  {
    switch (t_op->get_coop())
    {
      case D2D:
        printf("Setting D2D \n"); //ToBeDeleted
        // Unidirectional one-to-one transfer between devices
        // ToDo: Validate size orig/dest
        dev_a = *(t_op->get_orig())->begin();
        // n_hops = 2;
        switch (t_op->get_mhtd())
        {
          case H2D:
            printf("Using H2D\n"); //ToBeDeleted
            // Allow multiple destinations
            for (auto& dev_b : *t_op->get_dest())
            {
              // Check direct path D2D
              if ((*dev_mx_cpy_p)[dev_a][dev_b]>0) 
              {
                // ToDo: Remove available link?
                printf("Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, (*dev_mx_cpy_p)[dev_a][dev_b]); //ToBeDeleted
                all_deps->push_back(new cops_dep(0, 0, dev_a, dev_b, t_op->get_size(), 0));
              }
              else
              {
                printf("Invalid H/D combination\n"); //ToBeDeleted
              }
            }
            // return 0;
            break; // H2D

          // Distant Vector
          case DVT:
          printf("Using DVT\n"); //ToBeDeleted
            // ******************** HERE ******************** //
            // Find multiple routes that reduce latency 
            // Single sink
            dev_b = *(t_op->get_dest())->begin();

            i_paths = 0;
            p_done = 0;
            i_hops = 0;
            n_hops = n_dev + n_hst - 2; // max number of hops

            // ToDo: check numa affinty 
            // Check direct coonection (1 hop)
            if ((*dev_mx_cpy_p)[dev_a][dev_b]>0) 
            {
              // ToDo: Remove available link?
              printf("Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, (*dev_mx_cpy_p)[dev_a][dev_b]); //ToBeDeleted
              op_deps = new cops_dep(0, 0, dev_a, dev_b, 0, 0, i_paths);
              all_deps->push_back(op_deps);
              i_paths++;
              // Remove link
              (*dev_mx_cpy_p)[dev_a][dev_b] = 0;
            }
            // Build Paths
            // for // all devices
            // for // max hops
            while(!p_done){
              // Find connecting paths
              if(i_paths>=m_paths){
                p_done = 1;
              }
              else
              {
                max_bw = 0;
                for (dev_i = 0; dev_i < n_dev+n_hst; ++dev_i)
                {
                  if (dev_i != dev_b && (*dev_mx_cpy_p)[dev_i][dev_b] > max_bw) 
                  {
                    max_bw = (*dev_mx_cpy_p)[dev_i][dev_b];
                    dev_ii = dev_i;
                  }
                }

                for (dev_i = n_hst; dev_i < n_dev+n_hst; ++dev_i)
                {
                  // ToDo: Check links more than 2 hops?
                  lnk_bw = (*dev_mx_cpy_p)[dev_a][dev_i] + (*dev_mx_cpy_p)[dev_i][dev_b];
                  if (dev_i != dev_b && (*dev_mx_cpy_p)[dev_i][dev_b] > max_bw) 
                  {
                    max_bw = (*dev_mx_cpy_p)[dev_i][dev_b];
                    dev_ii = dev_i;
                  }
                  // printf("Testing H/D Orig: %d Dest: %d Value: %d\n", dev_i, dev_b, (*dev_mx_cpy_p)[dev_i][dev_b]); //ToBeDeleted
                }

                if (max_bw > 0) 
                {
                  printf("Valid H/D Orig: %d Dest: %d Value: %d\n", dev_ii, dev_b, (*dev_mx_cpy_p)[dev_ii][dev_b]);
                  // Generate dependencies
                  op_deps = new cops_dep(0, 0, dev_a, dev_ii, 0, i_paths, i_paths);
                  all_deps->push_back(op_deps);
                  op_deps = new cops_dep(0, 0, dev_ii, dev_b, 0, 0, i_paths);
                  all_deps->push_back(op_deps);
                  // Remove links
                  (*dev_mx_cpy_p)[dev_a][dev_ii] = 0;
                  (*dev_mx_cpy_p)[dev_ii][dev_b] = 0;
                  i_paths++;
                  printf("i_paths: %d\n", i_paths);
                }
                else
                {
                  p_done = 1;
                  printf("i_paths: %d\n", i_paths);
                }
              }
            }

            // Calculate sizes / offsets
            // Evenly 
            dev_i = 1;
            for (auto& a_dep : *all_deps)
            {
              cops_dep * c_dep = static_cast <cops_dep *> (a_dep);
              c_dep->size = t_op->get_size()/i_paths;
              c_dep->offs = c_dep->offs*c_dep->size;
              // printf("Orig: %d Dest: %d Size: %d Offs: %d\n", c_dep->orig, c_dep->dest, c_dep->size, c_dep->offs);
              dev_i++;
            }
            
            // Size based on bandwidth
            // ******************** HERE ******************** //


          break; // DVT

          // Max-Flow
          case MXF:
            printf("Using MXF\n"); //ToBeDeleted
            // Find multiple routes that increase throughput
            // Single sink
            dev_b = *(t_op->get_dest())->begin();
            i_paths = 0;
            

            // ToDo: check numa affinty 

            // i_paths<m_paths || 

            if ((*dev_mx_cpy_p)[dev_a][dev_b]>0) 
            {
              // ToDo: Remove available link?
              printf("Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, (*dev_mx_cpy_p)[dev_a][dev_b]); //ToBeDeleted
              op_deps = new cops_dep(0, 0, dev_a, dev_b, 0, 0, i_paths);
              all_deps->push_back(op_deps);
              i_paths++;
              // Remove link
              (*dev_mx_cpy_p)[dev_a][dev_b] = 0;
              (*dev_mx_cpy_p)[dev_b][dev_a] = 0;
            }

            if (i_paths < m_paths)
            {
              dev_ii = dev_b;
              i_link = 0;
              n_link = 0;
              i_hops = 1;
              p_done = 0;

              while (!p_done)
              {
                // Calculate latency 
                // Store paths increasing number of links
                for (dev_i = 0; dev_i < n_dev+n_hst; ++dev_i)
                {
                  // printf("Testing Link %d to %d \n", dev_i, dev_ii);
                  if (dev_i != dev_ii && (*dev_mx_cpy_p)[dev_i][dev_ii] > 0) 
                  {
                    // Add link to list
                    if (i_hops == 1)
                    {
                      // printf("Creating I Link %d to %d \n", dev_i, dev_ii);
                      iop_path = new op_path(dev_i, static_cast< float >((*dev_mx_cpy_p)[dev_i][dev_ii]), i_hops);
                    }
                    else
                    {
                      // printf("Creating II Link %d to %d \n", dev_i, dev_ii);
                      iop_path = new op_path(dev_i, static_cast< float >((*dev_mx_cpy_p)[dev_i][dev_ii]), i_hops, v_paths.at(i_link-1));
                    }
                    v_paths.push_back(iop_path);
                    // printf("Link %d to %d bw: %d \n", dev_i, dev_ii, (*dev_mx_cpy_p)[dev_i][dev_ii]);
                    (*dev_mx_cpy_p)[dev_i][dev_ii] = 0;
                    (*dev_mx_cpy_p)[dev_ii][dev_i] = 0;
                    ++n_link;
                    if (dev_i == dev_a)
                    {
                      // Valid link to origin
                      printf("Valid Link %d to %d latency: %f\n", dev_i, dev_ii, v_paths.at(i_link)->p_lat);
                      vi_paths.push_back(iop_path);
                    }
                  }
                  // printf("Testing H/D Orig: %d Dest: %d Value: %d\n", dev_i, dev_b, (*dev_mx_cpy_p)[dev_i][dev_b]); //ToBeDeleted
                }
                if (n_link > i_link)
                {
                  // Assign origin from list of valid links
                  dev_ii = v_paths.at(i_link)->n_id;
                  // printf("New Node %d \n", dev_ii);
                  i_hops = v_paths.at(i_link)->n_hops + 1;
                  ++i_link;
                }
                else if (i_hops >= m_hops)
                {
                  p_done = 1;
                  printf("Max Hops\n");
                }
                else
                {
                  // No more available links 
                  printf("No More Valid Links\n");
                  p_done = 1;
                }
                // Get other node with same numer of hops or increase 
              } 
            }

            // ******************** HERE ******************** //
            p_done = 0;
            while(!p_done)
            {
              min_lat = 1000;
              // Evaluate all valid paths to find lowest latency 
              for (auto& i_path : vi_paths)
              {
                if (min_lat > i_path->p_lat)
                {
                  printf("Path to %d latency: %f\n", iop_path->n_id, iop_path->p_lat);
                  min_lat = i_path->p_lat;
                  iop_path = i_path;
                  // i_path = vi_paths.erase(i_path);
                }
              }
              if (min_lat != 1000)
              {
                while (iop_path->p_op_path != NULL)
                {
                  // Get all links of current path
                  // all_deps->push_back(new cops_dep(0, 0, dev_a, dev_b, t_op->get_size(), 0));
                  printf("[min] Path to %d latency: %f\n", iop_path->n_id, iop_path->p_lat);
                  iop_path = iop_path->p_op_path;
                }
                i_paths++;
              }
              if(i_paths>=m_paths){
                p_done = 1;
              }
            }
            // {
            //   cops_dep * c_dep = static_cast <cops_dep *> (a_dep);
            //   c_dep->size = t_op->get_size()/i_paths;
            //   c_dep->offs = c_dep->offs*c_dep->size;
            //   // printf("Orig: %d Dest: %d Size: %d Offs: %d\n", c_dep->orig, c_dep->dest, c_dep->size, c_dep->offs);
            //   dev_i++;
            // }
            // Get connecting paths
            //   if(i_paths>=m_paths){
            //     p_done = 1;
            //   }
            //   else
            //   {
              
                // if (max_bw > 0) 
                // {
                //   printf("Valid H/D Orig: %d Dest: %d Value: %d\n", dev_ii, dev_b, (*dev_mx_cpy_p)[dev_ii][dev_b]);
                //   // Generate dependencies
                //   op_deps = new cops_dep(0, 0, dev_a, dev_ii, 0, i_paths, i_paths);
                //   all_deps->push_back(op_deps);
                //   op_deps = new cops_dep(0, 0, dev_ii, dev_b, 0, 0, i_paths);
                //   all_deps->push_back(op_deps);
                //   // Remove links
                //   (*dev_mx_cpy_p)[dev_a][dev_ii] = 0;
                //   (*dev_mx_cpy_p)[dev_ii][dev_b] = 0;
                //   i_paths++;
                //   printf("i_paths: %d\n", i_paths);
                // }
                // else
                // {
                //   p_done = 1;
                //   printf("i_paths: %d\n", i_paths);
                // }
            //   }
            // }

            // Calculate sizes / offsets
            // Evenly 
            // dev_i = 1;
            // for (auto& a_dep : *all_deps)
            // {
            //   cops_dep * c_dep = static_cast <cops_dep *> (a_dep);
            //   c_dep->size = t_op->get_size()/i_paths;
            //   c_dep->offs = c_dep->offs*c_dep->size;
            //   // printf("Orig: %d Dest: %d Size: %d Offs: %d\n", c_dep->orig, c_dep->dest, c_dep->size, c_dep->offs);
            //   dev_i++;
            // }
            
            // Size based on bandwidth
            // ******************** HERE ******************** //

          break; // MXF


          default:
            printf("Invalid Method\n");
            // return 1;
        }
      break; // D2D

      case BRC:
        printf("Setting BRC\n");
        // Unidirectional one-to-all transfer between devices
        switch (t_op->get_mhtd())
        {
          case H2D:
          printf("Using H2D\n");
          if (t_op->get_orig()->size() > 1)
          {
            // Numa affinity
              for (auto dev_b : *t_op->get_dest()) 
              {
                h_aff = *(t_op->get_orig())->begin();
                for (auto dev_a : *t_op->get_orig()) if ((*dev_mx_cpy_p)[dev_a][dev_b] > (*dev_mx_cpy_p)[h_aff][dev_b]) h_aff = dev_a; // Compare hosts and find affinity
                
                if ((*dev_mx_cpy_p)[h_aff][dev_b]>0)
                {
                  // ToDo: Remove available link?
                  printf("Valid H/D Orig: %d Dest: %d Value: %d\n", h_aff, dev_b, (*dev_mx_cpy_p)[h_aff][dev_b]);
                  op_deps = new cops_dep(0, 0, h_aff, dev_b, t_op->get_size(), 0);
                  all_deps->push_back(op_deps);
                }
                // ToDo: check for multi-link connection
                else
                {
                  printf("Invalid H/D combination\n");
                }
            }
            
          }
          else
          {
            dev_a = *(t_op->get_orig())->begin();
            for (auto dev_b : *t_op->get_dest()) {
              // Check direct path D2D
              if ((*dev_mx_cpy_p)[dev_a][dev_b]>0) 
              {
                // ToDo: Remove available link?
                printf("Valid H/D Orig: %d Dest: %d Value: %d\n", dev_a, dev_b, (*dev_mx_cpy_p)[dev_a][dev_b]);
                op_deps = new cops_dep(0, 0, dev_a, dev_b, t_op->get_size(), 0);
                all_deps->push_back(op_deps);

              }
              else
              {
                printf("Invalid H/D combination\n");
              }
            }
          }
          // return 0;
          break; // H2D

          // Distant Vector
          case DVT:
          printf("Using DVT\n");
          break; // DVT

          // Max-Flow
          case MXF:
          printf("Using MXF\n");

          break; // MXF

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
          case H2D:
          // return 0;
          break; // H2D

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
          case H2D:
          // return 0;
          break; // H2D

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
          case H2D:
          // return 0;
          break; // H2D

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
      //   printf("H2D\n");

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



  // Memory allocation
  // dest_no = 2;
  // // Allocate memory
  // cnk_len = arr_len/dest_no;
  // size_c = sizeof(int) * cnk_len;
  
  // #pragma omp parallel num_threads(num_thr) private(cpu, node, status, ret_code, ptr_to_check)
  // {
  //   syscall(SYS_getcpu, &cpu, &node, NULL);
  //   switch(cpu){
  //     case 0:
  //       if (v_flag) printf("Alloc Host_0: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
  //       x_arr_0 = (int *)omp_alloc(size_c, x_alloc);
  //       ptr_to_check = (void**) &x_arr_0;
  //       ret_code=move_pages(0 /*self memory */, 1, ptr_to_check, NULL, status, 0);
  //       if (v_flag) printf("Memory at %p is at %d node (retcode %d)\n", ptr_to_check, status[0], ret_code);
  //     break;
  //     case 22:
  //       if (v_flag) printf("Alloc Host_1: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
  //       x_arr_1 = (int *)omp_alloc(size_c, y_alloc);
  //       ptr_to_check = (void**) &x_arr_1;
  //       ret_code=move_pages(0 /*self memory */, 1, ptr_to_check, NULL, status, 0);
  //       if (v_flag) printf("Memory at %p is at %d node (retcode %d)\n", ptr_to_check, status[0], ret_code);
  //     break;
  //     // default:
  //       // if (v_flag) printf("Default: CPU core %.2u NUMA node %u Thread %.2d\n", cpu, node, omp_get_thread_num());
  //   }
  //   #pragma omp barrier
  // }


  // #pragma omp parallel num_threads(num_thr) private(dest_id, cpu, node)
  // {
  //   #pragma omp single
  //   {
  //     start = omp_get_wtime();
  //   }
  //   syscall(SYS_getcpu, &cpu, &node, NULL);
  //   switch(cpu){
  //     case 0:
  //       if (v_flag) printf("MemCpy Dev_0: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
  //         omp_target_memcpy
  //         (
  //           x_ptr[0],                                 // dst
  //           x_arr_0,                                  // src
  //           size_c,                                   // length 
  //           0,                                        // dst_offset
  //           0,                                        // src_offset, 
  //           0,                                        // dst_device_num
  //           omp_get_initial_device()                  // src_device_num
  //         );
  //     break;
  //     case 22:
  //       dest_id = 2;
  //       if (v_flag) printf("MemCpy Dev_1: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
  //         omp_target_memcpy
  //         (
  //           x_ptr[2],                     // dst
  //           x_arr_1,                            // src
  //           size_c,                             // length 
  //           0,                                  // dst_offset
  //           0,                                  // src_offset, 
  //           2,                            // dst_device_num
  //           omp_get_initial_device()            // src_device_num
  //         );
  //         omp_target_memcpy
  //         (
  //           x_ptr[0],                           // dst
  //           x_ptr[2],                     // src
  //           size_c,                             // length 
  //           size_c,                             // dst_offset
  //           0,                                  // src_offset, 
  //           0,                                  // dst_device_num
  //           2                             // src_device_num
  //         );
  //     break;
  //     // default:
  //     //   if (v_flag) printf("Default: CPU core %.2u NUMA node %u Thread %d\n", cpu, node, omp_get_thread_num());
  //   }
  //   #pragma omp barrier
  //   #pragma omp single
  //   {
  //     end = omp_get_wtime();
  //   }
  //   #pragma omp barrier
  // }








  // printf("All Operations\n");
  // for (auto& it : all_ops)
  // {
  //   // Print the values
  //   printf("Orig: Dest: Size: %d Offs: \n", it->get_size());
  //   // cout << it->orig << ' ';
  // } 


// for (dev_a = 0; dev_a < n_org; ++dev_a)
// {
//   for (dev_b = 0; dev_b < n_dst; ++dev_b)
//   {
//     // Check direct path D2D
//     if (dev_mx_cpy[d_org[dev_a]][d_dst[dev_b]]>0) 
//     {
//       //Remove available link
//       printf("Valid H/D combination\n");
//       printf("Value %d\n", dev_mx_cpy[d_org[dev_a]][d_dst[dev_b]]);
//       op_deps = new cops_dep(d_org[dev_a], d_dst[dev_b], arr_len, 0);
//       all_deps.push_back(op_deps);
//     }
//     else
//     {
//       printf("Invalid H/D combination\n");
//     }
//   }
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

// PROG=d2d_test_7; clang++ -fopenmp -fopenmp-targets=nvptx64 -o $PROG.x --cuda-gpu-arch=sm_70 -L/soft/compilers/cuda/cuda-11.0.2/lib64 -L/soft/compilers/cuda/cuda-11.0.2/targets/x86_64-linux/lib/ -I/soft/compilers/cuda/cuda-11.0.2/include -ldl -lcudart -pthread $PROG.cpp

//**************************************************//
//                        RUN                       //
//**************************************************//

// OMP_PROC_BIND=spread taskset -c 0,1,22,23 ./d2d_test_7.x

// nsys profile -o d2d_test_6_6G_v2 ./run_one.sh

