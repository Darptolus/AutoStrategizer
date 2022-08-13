#include <stdio.h>
#include <cstdlib>

int get_desc(int id, int n_nodes){
  int n_desc = 0;
  if(id*2+1 > n_nodes || id*2+2 > n_nodes){
    return n_desc;
  }else{
    if(id*2+1 < n_nodes){
      n_desc++;
      n_desc = n_desc + get_desc(id*2+1, n_nodes);
    }  
    if(id*2+2 < n_nodes){
      n_desc++;
      n_desc = n_desc + get_desc(id*2+2, n_nodes);
    }
    return n_desc;
  }
}

int main(int argc, char** argv){
  int n = atoi(argv[1]);
  for (int i=0; i<n; ++i)
    printf("ID= %d No Descendants= %d\n", i, get_desc(i, n));
  return 0;
}

/*
  uint32_t t_lvl;

  t_lvl = log2 (dev+1); // Device 0 is lvl 1 
    // printf("level of dev %d is %d\n", dev, t_lvl);
    ct_size[dev] = a_size/pow(2,t_lvl); // Chunk size
    if (dev == 0)
      ct_size[dev+num_dev] = 0; // Offset
    else  
      ct_size[dev+num_dev] = ct_size[(dev-1)/2+num_dev]; // Offset
    x_ptr[dev] = omp_target_alloc(ct_size[dev], dev);
*/