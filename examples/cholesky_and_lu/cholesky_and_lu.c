#include "cholesky/cholesky.h"
#include <pthread.h>

typedef struct {
  int start;
  int argc;
  char **argv;
  unsigned ctx;
  int the_other_ctx;
  int *procs;
  int ncpus;
} params;

typedef struct {
  double flops;
  double avg_timing;
} retvals;

#define NSAMPLES 3
int first = 1;
pthread_mutex_t mut;

pthread_barrier_t barrier;

void* func_cholesky(void *val){
  params *p = (params*)val;
  unsigned sched_ctx = p->ctx;
  int the_other_ctx = p->the_other_ctx;

  int i;
  retvals *rv  = (retvals*)malloc(sizeof(retvals));
  rv->flops = 0;
  rv->avg_timing = 0;
  double timing = 0;
  for(i = 0; i < NSAMPLES; i++)
    {
      rv->flops += run_cholesky_implicit(sched_ctx, p->start, p->argc, p->argv, &timing, &barrier);
      rv->avg_timing += timing;
    }


  pthread_mutex_lock(&mut);
  if(first)
    {
      starpu_delete_sched_ctx(p->ctx);
      starpu_add_workers_to_sched_ctx(p->procs, p->ncpus, the_other_ctx);
    }
  first = 0;
  pthread_mutex_unlock(&mut);
 

  rv->flops /= NSAMPLES;
  rv->avg_timing /= NSAMPLES;
  return (void*)rv;
}

void cholesky_vs_cholesky(params *p1, params *p2, params *p3, int ncpus1, int ncpus2){
  /* 2 cholesky in different ctxs */
  starpu_init(NULL);
  starpu_helper_cublas_init();

  int procs[ncpus1];
  int i;
  for(i = 0; i < ncpus1; i++)
    procs[i] = i;

  p1->ctx = starpu_create_sched_ctx("heft", procs, ncpus1, "cholesky1");
  p2->the_other_ctx = (int)p1->ctx;
  p1->procs = procs;
  p1->ncpus = ncpus1;
  int procs2[ncpus2];

  for(i = 0; i < ncpus2; i++)
    procs2[i] = ncpus1+i;

  p2->ctx = starpu_create_sched_ctx("heft", procs2, ncpus2, "cholesky2");
  p1->the_other_ctx = (int)p2->ctx;
  p2->procs = procs2;
  p2->ncpus = ncpus2;

  pthread_t tid[2];
  pthread_barrier_init(&barrier, NULL, 2);
  pthread_mutex_init(&mut, NULL);

  struct timeval start;
  struct timeval end;

  gettimeofday(&start, NULL);


  pthread_create(&tid[0], NULL, (void*)func_cholesky, (void*)p1);
  pthread_create(&tid[1], NULL, (void*)func_cholesky, (void*)p2);

  void *gflops_cholesky1;
  void *gflops_cholesky2;
 
  pthread_join(tid[0], &gflops_cholesky1);
  pthread_join(tid[1], &gflops_cholesky2);

  gettimeofday(&end, NULL);

  pthread_mutex_destroy(&mut);
  starpu_helper_cublas_shutdown();
  starpu_shutdown();
  
  double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
  timing /= 1000000;
  timing /= 60;

  printf("%2.2f %2.2f ", ((retvals*)gflops_cholesky1)->flops, ((retvals*)gflops_cholesky2)->flops);
  printf("%2.2f %2.2f %2.2f\n", ((retvals*)gflops_cholesky1)->avg_timing, ((retvals*)gflops_cholesky2)->avg_timing, timing);


}

int main(int argc, char **argv)
{
  //  printf("argc = %d\n", argc);
  int ncpus1=0, ncpus2=0;
  int i;
  
  for (i = 9; i < argc; i++) {
    if (strcmp(argv[i], "-ncpus1") == 0) {
      char *argptr;
      ncpus1 = strtol(argv[++i], &argptr, 10);
    }
    
    if (strcmp(argv[i], "-ncpus2") == 0) {
      char *argptr;
      ncpus2 = strtol(argv[++i], &argptr, 10);
    }    
  }
  //  printf("%d %d\n", ncpus1, ncpus2);
  params p1;
  p1.start = 1;
  p1.argc = 5;
  p1.argv = argv;

  params p2;
  p2.start = 5;
  p2.argc = 9;
  p2.argv = argv;

  params p3;
  p3.argc = argc;
  p3.argv = argv;
  p3.ctx = 0;
  cholesky_vs_cholesky(&p1, &p2,&p3, ncpus1, ncpus2);

  return 0;
}
