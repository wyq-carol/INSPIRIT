#include "cholesky/cholesky.h"
#include <pthread.h>

typedef struct {
  int start;
  int argc;
  char **argv;
  int ctx;
} params;

typedef struct {
  double flops;
  double avg_timing;
} retvals;

#define NSAMPLES 1

pthread_barrier_t barrier;

void* func_cholesky(void *val){
  params *p = (params*)val;
  int sched_ctx = p->ctx;

  int i;
  retvals *rv  = (retvals*)malloc(sizeof(retvals));
  rv->flops = 0;
  rv->avg_timing = 0;
  double timing = 0;
  for(i = 0; i < NSAMPLES; i++)
    {
      rv->flops += run_cholesky_implicit(sched_ctx, p->start, p->argc, p->argv, &timing, (sched_ctx == -1 ? NULL : &barrier));
      rv->avg_timing += timing;
    }

  rv->flops /= NSAMPLES;
  rv->avg_timing /= NSAMPLES;
  return (void*)rv;
}

void cholesky_vs_cholesky(params *p1, params *p2, params *p3){
  /* 2 cholesky in different ctxs */
  starpu_init(NULL);
  starpu_helper_cublas_init();

  int procs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
		 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
		 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
		 39, 40, 41, 42, 43, 44, 45, 46, 47,
		 48, 49, 50, 51, 52, 53, 54,
		 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
		 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,  78,
		 79, 80, 81, 82, 83, 84, 85};
  int id = starpu_create_sched_ctx("heft", NULL, -1, "cholesky");
  p1->ctx = id;

  int procs2[] =  {86, 87, 88, 89, 90,
		   91, 92, 93, 94, 95};
  p2->ctx = id ;//= starpu_create_sched_ctx("heft", procs2, 10, "cholesky2");

  pthread_t tid[2];
  pthread_barrier_init(&barrier, NULL, 2);

  pthread_create(&tid[0], NULL, (void*)func_cholesky, (void*)p1);
  pthread_create(&tid[1], NULL, (void*)func_cholesky, (void*)p2);

  void *gflops_cholesky1;
  void *gflops_cholesky2;
 
  pthread_join(tid[0], &gflops_cholesky1);
  pthread_join(tid[1], &gflops_cholesky2);

  starpu_helper_cublas_shutdown();
  starpu_shutdown();

  /* /\* 1 cholesky all alone on the whole machine *\/ */
  /* starpu_init(NULL); */
  /* starpu_helper_cublas_init(); */

  /* void *gflops_cholesky3 = func_cholesky((void*)p3); */

  /* starpu_helper_cublas_shutdown(); */
  /* starpu_shutdown(); */


  /* /\* 2 cholesky in a single ctx *\/ */
  /* starpu_init(NULL); */
  /* starpu_helper_cublas_init(); */

  /* pthread_t tid2[2]; */

  /* pthread_create(&tid2[0], NULL, (void*)func_cholesky, (void*)p3); */
  /* pthread_create(&tid2[1], NULL, (void*)func_cholesky, (void*)p3); */

  /* void *gflops_cholesky4; */
  /* void *gflops_cholesky5; */
 
  /* pthread_join(tid2[0], &gflops_cholesky4); */
  /* pthread_join(tid2[1], &gflops_cholesky5); */

  /* starpu_helper_cublas_shutdown(); */
  /* starpu_shutdown(); */

  /* printf("%2.2f %2.2f %2.2f %2.2f %2.2f ", ((retvals*)gflops_cholesky1)->flops, ((retvals*)gflops_cholesky2)->flops, ((retvals*)gflops_cholesky3)->flops, ((retvals*)gflops_cholesky4)->flops, ((retvals*)gflops_cholesky5)->flops); */

  /* printf("%2.2f %2.2f %2.2f %2.2f %2.2f\n", ((retvals*)gflops_cholesky1)->avg_timing, ((retvals*)gflops_cholesky2)->avg_timing, ((retvals*)gflops_cholesky3)->avg_timing, ((retvals*)gflops_cholesky4)->avg_timing, ((retvals*)gflops_cholesky5)->avg_timing); */

  /* free(gflops_cholesky1); */
  /* free(gflops_cholesky2); */
  /* free(gflops_cholesky3); */
  /* free(gflops_cholesky4); */
  /* free(gflops_cholesky5); */
}

int main(int argc, char **argv)
{
  //  printf("argc = %d\n", argc);
  params p1;
  p1.start = 1;
  p1.argc = 5;
  p1.argv = argv;

  params p2;
  p2.start = 5;
  p2.argc = argc;
  p2.argv = argv;

  params p3;
  p3.argc = argc;
  p3.argv = argv;
  p3.ctx = -1;
  cholesky_vs_cholesky(&p1, &p2,&p3);

  return 0;
}
