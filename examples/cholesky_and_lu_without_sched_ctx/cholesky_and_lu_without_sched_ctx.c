#include "cholesky/cholesky.h"
#include "lu/lu_example_float.c"

int main(int argc, char **argv)
{
  struct timeval start;
  struct timeval end;

  gettimeofday(&start, NULL);

  run_cholesky_tile_tag(argc, argv);

  run_lu(argc, argv);

  finish_cholesky_tile_tag();
  finish_lu();
  
  gettimeofday(&end, NULL);

  double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
  //      fprintf(stderr, "Computation took (in ms)\n");                                    
  printf("%2.2f\n", timing/1000);

  double flop = (1.0f*size*size*size)/3.0f;
  //      fprintf(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));     

  return 0;
}
