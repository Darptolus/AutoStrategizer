#include <omp.h>
#include <stdio.h>

int main(){
  printf("Here\n");
  #pragma omp target                   
  { 
    printf("Here..\n");
  }  
return 0;
}