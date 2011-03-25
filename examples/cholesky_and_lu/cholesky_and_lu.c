#include "cholesky/cholesky.h"
#include "lu/lu_example_float.c"

int main(int argc, char **argv)
{
  struct timeval start;
  struct timeval end;

  starpu_init(NULL);

  struct starpu_sched_ctx sched_ctx;
  int procs[] = {1, 2, 3};
  starpu_create_sched_ctx(&sched_ctx, "heft", procs, 3);

  run_cholesky_tile_tag(&sched_ctx, argc, argv, &start);

  struct starpu_sched_ctx sched_ctx2;
  int procs2[] = {0, 4, 5, 6, 7, 8, 9, 10, 11};
  starpu_create_sched_ctx(&sched_ctx2, "heft", procs2, 5);

  run_lu(&sched_ctx2, argc, argv, NULL);

  finish_cholesky_tile_tag(NULL);
  finish_lu(&end);
  //starpu_task_wait_for_all();

  starpu_shutdown();

  double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
  printf("%2.2f\n", timing/1000);

  return 0;
}
