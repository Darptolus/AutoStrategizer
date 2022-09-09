#include<stdio.h>

void set_an_element(int *p, int val) {
    *p = val;
}

void print_all_elements(int *v, int n) {
  int i;
  for (i = 0; i < n; ++i) {
    printf("%d, ", v[i]);
  }
  printf("\n");
}

// Dependencies
class Dep{
  private:
    int orig;
    int dest;
    int orig_id;
  public:
    Dep (int a_o, int a_d):orig {a_o}, dest {a_d}{};
    Dep (int a_o, int a_d, int o_id):orig {a_o}, dest {a_d}, orig_id {o_id}{};
    ~Dep(){};
    int get_orig(){return this->orig;};
    int get_dest(){return this->dest;};
    void set_orig_id(int o_id){this->orig_id = o_id;};
};

// typedef std::vector<Dep*> deps;

int main()
{
  int n =10;
  int v[n];

  Dep * main_dep_arr, sec_dep_arr;

  #pragma omp parallel
  #pragma omp single
  {
      int i;
      for (i = 0; i < n; ++i)
          #pragma omp task depend(out: v[i])
          set_an_element(&v[i], i);

      #pragma omp task depend(iterator(it = 0:n), in: v[it])
    // #pragma omp task depend(in: v[0:n]) Violates Array section restriction.
      print_all_elements(v, n);

  }

  return 0;
}


// typedef std::vector<Dep*>::iterator deps_itr;

// class Depends{
//   private:
//     Dep * a_dep;
//     std::vector<Dep*> all_deps;
//   public:
//     Depends(){};
//     ~Depends(){};
//     void add_dep(int main, int sec)
//     {
//       Dep * a_dep = new Dep(main,sec);
//       all_deps.push_back(a_dep);
//     };
//     void clear_dep(){all_deps.clear();};
//     int get_main(){return a_dep->main;};
//     int get_sec(){return a_dep->sec;};
//     deps_itr get_beg(){return all_deps.begin();};
//     deps_itr get_end(){return all_deps.end();};
// };

