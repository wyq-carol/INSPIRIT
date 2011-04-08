#include "cholesky/cholesky.h"
#include "lu/lu_example_float.c"
#include <pthread.h>

typedef struct {
  int argc;
  char **argv;
} params;

#define NSAMPLES 20

struct starpu_sched_ctx sched_ctx;
struct starpu_sched_ctx sched_ctx2;
struct starpu_sched_ctx sched_ctx3;
struct starpu_sched_ctx sched_ctx4;

void* func_cholesky(void *val){
  params *p = (params*)val;

  int procs[] = {1, 2, 3, 4, 5, 6};
  starpu_create_sched_ctx(&sched_ctx, "heft", procs, 6, "cholesky1");

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
  starpu_create_sched_ctx(&sched_ctx2, "heft", procs, 6, "cholesky2");

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

  int procs2[] = {0, 7, 8, 9, 10, 11};
  starpu_create_sched_ctx(&sched_ctx3, "heft", procs2, 6, "lu");

  int i;
  double *flops = (double*)malloc(sizeof(double));
  (*flops) = 0;
  for(i = 0; i < NSAMPLES; i++)
    {
      (*flops) += run_lu(&sched_ctx3, p->argc, p->argv);
    }

  (*flops) /= NSAMPLES;
  return (void*)flops;
}

void* func_lu2(void *val){
  params *p = (params*)val;

  int procs2[] = {1, 2, 3, 4, 5, 6};
  starpu_create_sched_ctx(&sched_ctx4, "heft", procs2, 6, "lu2");

  int i;
  double *flops = (double*)malloc(sizeof(double));
  (*flops) = 0;
  for(i = 0; i < NSAMPLES; i++)
    {
      (*flops) += run_lu(&sched_ctx4, p->argc, p->argv);
    }

  (*flops) /= NSAMPLES;
  return (void*)flops;
}

void* func_lu3(void *val){
  params *p = (params*)val;

  int i;
  double *flops = (double*)malloc(sizeof(double));
  (*flops) = 0;
  for(i = 0; i < NSAMPLES; i++)
    {
      (*flops) += run_lu(NULL, p->argc, p->argv);
    }

  (*flops) /= NSAMPLES;
  return (void*)flops;
}

void cholesky_vs_cholesky(params *p){
  /* 2 cholesky in different ctxs */
  starpu_init(NULL);
  starpu_helper_cublas_init();

  pthread_t tid[2];

  pthread_create(&tid[0], NULL, (void*)func_cholesky, (void*)p);
  pthread_create(&tid[1], NULL, (void*)func_cholesky2, (void*)p);

  void *gflops_cholesky1;
  void *gflops_cholesky2;
 
  pthread_join(tid[0], &gflops_cholesky1);
  pthread_join(tid[1], &gflops_cholesky2);

  starpu_helper_cublas_shutdown();
  starpu_shutdown();

  /* 1 cholesky all alone on the whole machine */
  starpu_init(NULL);
  starpu_helper_cublas_init();

  void *gflops_cholesky3 = func_cholesky3(p);

  starpu_helper_cublas_shutdown();
  starpu_shutdown();


  /* 2 cholesky in a single ctx */
  starpu_init(NULL);
  starpu_helper_cublas_init();

  pthread_t tid2[2];

  pthread_create(&tid2[0], NULL, (void*)func_cholesky3, (void*)p);
  pthread_create(&tid2[1], NULL, (void*)func_cholesky3, (void*)p);

  void *gflops_cholesky4;
  void *gflops_cholesky5;
 
  pthread_join(tid2[0], &gflops_cholesky4);
  pthread_join(tid2[1], &gflops_cholesky5);

  starpu_helper_cublas_shutdown();
  starpu_shutdown();

  printf("%2.2f %2.2f %2.2f %2.2f %2.2f\n", *((double*)gflops_cholesky1), *((double*)gflops_cholesky2), *((double*)gflops_cholesky3), *((double*)gflops_cholesky4), *((double*)gflops_cholesky5));

  free(gflops_cholesky1);
  free(gflops_cholesky2);
  free(gflops_cholesky3);
  free(gflops_cholesky4);
  free(gflops_cholesky5);
}

void cholesky_vs_lu(params *p){
  /* one cholesky and one lu each one in its own context */
  starpu_init(NULL);
  starpu_helper_cublas_init();

  pthread_t tid[2];

  pthread_create(&tid[0], NULL, (void*)func_cholesky, (void*)p);
  pthread_create(&tid[1], NULL, (void*)func_lu, (void*)p);

  void *gflops_cholesky;
  void *gflops_lu;
 
  pthread_join(tid[0], &gflops_cholesky);
  pthread_join(tid[1], &gflops_lu);

  starpu_helper_cublas_shutdown();
  starpu_shutdown();

  /*one cholesky and one lu mixed in a single context*/
  starpu_init(NULL);
  starpu_helper_cublas_init();

  pthread_t tid2[2];

  pthread_create(&tid2[0], NULL, (void*)func_cholesky3, (void*)p);
  pthread_create(&tid2[1], NULL, (void*)func_lu3, (void*)p);

  void *gflops_cholesky2;
  void *gflops_lu2;
 
  pthread_join(tid2[0], &gflops_cholesky2);
  pthread_join(tid2[1], &gflops_lu2);

  starpu_helper_cublas_shutdown();
  starpu_shutdown();


  /* 1 lu all alone on the whole machine */
  starpu_init(NULL);
  starpu_helper_cublas_init();

  void *gflops_lu3 = func_lu3(p);

  starpu_helper_cublas_shutdown();
  starpu_shutdown();

  /* 1 cholesky all alone on the whole machine */
  starpu_init(NULL);
  starpu_helper_cublas_init();

  void *gflops_cholesky3 = func_cholesky3(p);

  starpu_helper_cublas_shutdown();
  starpu_shutdown();

  printf("%2.2f %2.2f %2.2f %2.2f %2.2f %2.2f \n", *((double*)gflops_cholesky), *((double*)gflops_lu), *((double*)gflops_cholesky2), *((double*)gflops_lu2), *((double*)gflops_cholesky3), *((double*)gflops_lu3));

  free(gflops_cholesky);
  free(gflops_cholesky2);
  free(gflops_cholesky3);
  free(gflops_lu);
  free(gflops_lu2);
  free(gflops_lu3);
}

void lu_vs_lu(params *p){
  /* 2 lu in different ctxs */
  starpu_init(NULL);
  starpu_helper_cublas_init();

  pthread_t tid[2];

  pthread_create(&tid[0], NULL, (void*)func_lu, (void*)p);
  pthread_create(&tid[1], NULL, (void*)func_lu2, (void*)p);

  void *gflops_lu1;
  void *gflops_lu2;
 
  pthread_join(tid[0], &gflops_lu1);
  pthread_join(tid[1], &gflops_lu2);

  starpu_helper_cublas_shutdown();
  starpu_shutdown();

  /* 1 lu all alone on the whole machine */
  starpu_init(NULL);
  starpu_helper_cublas_init();

  void *gflops_lu3 = func_lu3(p);

  starpu_helper_cublas_shutdown();
  starpu_shutdown();

  //   printf("%2.2f\n", *((double*)gflops_lu3));
  printf("%2.2f %2.2f %2.2f\n", *((double*)gflops_lu1), *((double*)gflops_lu2), *((double*)gflops_lu3));

  /* /\* 2 lu in a single ctx *\/ */
  /* starpu_init(NULL); */
  /* starpu_helper_cublas_init(); */

  /* pthread_t tid2[2]; */

  /* pthread_create(&tid2[0], NULL, (void*)func_lu3, (void*)p); */
  /* pthread_create(&tid2[1], NULL, (void*)func_lu3, (void*)p); */

  /* void *gflops_lu4; */
  /* void *gflops_lu5; */
 
  /* pthread_join(tid2[0], &gflops_lu4); */
  /* pthread_join(tid2[1], &gflops_lu5); */

  /* starpu_helper_cublas_shutdown(); */
  /* starpu_shutdown(); */

  /* printf("%2.2f %2.2f %2.2f %2.2f %2.2f\n", *((double*)gflops_lu1), *((double*)gflops_lu2), *((double*)gflops_lu3), *((double*)gflops_lu4), *((double*)gflops_lu5)); */

  /* free(gflops_lu1); */
  /* free(gflops_lu2); */
  /* free(gflops_lu3); */
  /* free(gflops_lu4); */
  /* free(gflops_lu5); */
}

int main(int argc, char **argv)
{
  params p;
  p.argc = argc;
  p.argv = argv;

  lu_vs_lu(&p);

  return 0;
}
