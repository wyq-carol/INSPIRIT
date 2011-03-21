#include "cholesky/cholesky.h"
#include "lu/lu_example_float.c"

int main(int argc, char **argv)
{
  starpu_init(NULL);

  struct starpu_sched_ctx sched_ctx;
  int procs[] = {1, 2, 3};
  starpu_create_sched_ctx(&sched_ctx, "random", procs, 3);

  run_cholesky_tile_tag(&sched_ctx, argc, argv);

  struct starpu_sched_ctx sched_ctx2;
  int procs2[] = {0, 4, 5, 6, 7};
  starpu_create_sched_ctx(&sched_ctx2, "random", procs2, 5);

  run_lu(&sched_ctx2, argc, argv);

  finish_cholesky_tile_tag();
  finish_lu();
  //starpu_task_wait_for_all();

  starpu_shutdown();
  
  return 0;
}
