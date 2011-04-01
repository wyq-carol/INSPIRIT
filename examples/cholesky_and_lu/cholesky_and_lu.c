#include "cholesky/cholesky.h"
#include "lu/lu_example_float.c"
#include <pthread.h>

typedef struct {
  int argc;
  char **argv;
} params;

#define NSAMPLES 10
struct starpu_sched_ctx sched_ctx;
struct starpu_sched_ctx sched_ctx2;

void* func_cholesky(void *val){
  params *p = (params*)val;

  int procs[] = {1, 2, 3};
  starpu_create_sched_ctx(&sched_ctx, "heft", procs, 3, "cholesky");

  int i;
  double *flops = (double*)malloc(sizeof(double));
  (*flops) = 0;
  for(i = 0; i < NSAMPLES; i++)
    {
      (*flops) += run_cholesky_implicit(&sched_ctx, p->argc, p->argv);
    }

  (*flops) /= NSAMPLES;
  return (void*)flops;
}

void* func_lu(void *val){
  params *p = (params*)val;

  int procs2[] = {0, 4, 5, 6, 7, 8, 9, 10, 11};
  starpu_create_sched_ctx(&sched_ctx2, "heft", procs2, 9, "lu");

  int i;
  double *flops = (double*)malloc(sizeof(double));
  (*flops) = 0;
  for(i = 0; i < NSAMPLES; i++)
    {
      printf("%d ", i);
      (*flops) += run_lu(&sched_ctx2, p->argc, p->argv);
    }

  (*flops) /= NSAMPLES;
  return (void*)flops;
}

int main(int argc, char **argv)
{
  starpu_init(NULL);
  starpu_helper_cublas_init();

  pthread_t tid[2];

  params p;
  p.argc = argc;
  p.argv = argv;

  pthread_create(&tid[0], NULL, (void*)func_cholesky, (void*)&p);
  pthread_create(&tid[1], NULL, (void*)func_cholesky, (void*)&p);

  void *gflops_cholesky1;
  void *gflops_cholesky2;
  //  void *gflops_lu = func_lu(&p);
 
  pthread_join(tid[0], &gflops_cholesky1);
  pthread_join(tid[1], &gflops_cholesky2);

  void *gflops_cholesky3 = func_cholesky(&p);

  printf("%2.2f %2.2f %2.2f\n", *((double*)gflops_cholesky1), *((double*)gflops_cholesky2), *((double*)gflops_cholesky3));

  starpu_helper_cublas_shutdown();
  starpu_shutdown();

  return 0;
}
