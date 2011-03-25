#include "cholesky/cholesky.h"
#include "lu/lu_example_float.c"

int main(int argc, char **argv)
{
  struct timeval start;
  struct timeval end;

  run_cholesky_tile_tag(argc, argv, &start);

  run_lu(argc, argv, NULL);

  finish_cholesky_tile_tag(NULL);
  finish_lu(&end);

  double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
  printf("%2.2f\n", timing/1000);

  return 0;
}
