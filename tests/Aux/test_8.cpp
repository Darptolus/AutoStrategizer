#include <stdio.h>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv){
  if(const char* arr_size = std::getenv("ARR_SZ"))
  {
    int n = atoi(arr_size);
    printf("size %d \n", n);
  }
  else if(argc>=2)
  {
    int n = atoi(argv[1]);
    printf("size %d \n", n);
  }
  else 
  {
    int n = 1;
    printf("size %d \n", n);
  }
  return 0;
}
