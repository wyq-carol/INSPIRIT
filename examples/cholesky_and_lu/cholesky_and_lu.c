#include "cholesky/cholesky.h"
#include "lu/lu_example_float.c"
#include <pthread.h>

typedef struct {
  int argc;
  char **argv;
} params;

#define NSAMPLES 1
struct starpu_sched_ctx sched_ctx;
struct starpu_sched_ctx sched_ctx2;

void* func_cholesky(void *val){
  params *p = (params*)val;

  int procs[] = {1, 2, 3, 4, 5, 6};
  starpu_create_sched_ctx(&sched_ctx, "heft", procs, 6, "cholesky");

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

void* func_cholesky2(void *val){
  params *p = (params*)val;

  int procs[] = {0, 7, 8, 9, 10, 11};
  starpu_create_sched_ctx(&sched_ctx2, "heft", procs, 6, "cholesky");

  int i;
  double *flops = (double*)malloc(sizeof(double));
  (*flops) = 0;
  for(i = 0; i < NSAMPLES; i++)
    {
      (*flops) += run_cholesky_implicit(&sched_ctx2, p->argc, p->argv);
    }

  (*flops) /= NSAMPLES;
  return (void*)flops;
}

void* func_cholesky3(void *val){
  params *p = (params*)val;

  int i;
  double *flops = (double*)malloc(sizeof(double));
  (*flops) = 0;
  for(i = 0; i < NSAMPLES; i++)
    {
      (*flops) += run_cholesky_implicit_all_machine(p->argc, p->argv);
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
  params p;
  p.argc = argc;
  p.argv = argv;

  starpu_init(NULL);
  starpu_helper_cublas_init();

  pthread_t tid[2];

  pthread_create(&tid[0], NULL, (void*)func_cholesky, (void*)&p);
  pthread_create(&tid[1], NULL, (void*)func_cholesky2, (void*)&p);

  void *gflops_cholesky1;
  void *gflops_cholesky2;
  //  void *gflops_lu = func_lu(&p);
 
  pthread_join(tid[0], &gflops_cholesky1);
  pthread_join(tid[1], &gflops_cholesky2);

  starpu_helper_cublas_shutdown();
  starpu_shutdown();

  /* starpu_init(NULL); */
  /* starpu_helper_cublas_init(); */

  /* void *gflops_cholesky3 = func_cholesky3(&p); */

  /* starpu_helper_cublas_shutdown(); */
  /* starpu_shutdown(); */

  /* printf("%2.2f %2.2f %2.2f\n", *((double*)gflops_cholesky1), *((double*)gflops_cholesky2), *((double*)gflops_cholesky3)); */

  return 0;
}
