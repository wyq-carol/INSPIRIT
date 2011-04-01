#include "cholesky/cholesky.h"
#include "lu/lu_example_float.c"

typedef struct {
  int argc;
  char **argv;
} params;

#define NSAMPLES 10

/* void* func_cholesky(void *val){ */
/*   params *p = (params*)val; */
/*   int i; */
/*   double *flops = (double*)malloc(sizeof(double)); */
/*   for(i = 0; i < NSAMPLES; i++) */
/*     (*flops) += run_cholesky_tile_tag(p->argc, p->argv); */

/*   (*flops) /= NSAMPLES; */

/*   return (void*)flops; */
/* } */

/* void* func_lu(void *val){ */
/*   params *p = (params*)val; */

/*   int i; */
/*   double *flops = (double*)malloc(sizeof(double)); */
/*   for(i = 0; i < NSAMPLES; i++) */
/*     (*flops) += run_lu(p->argc, p->argv); */

/*   (*flops) /= NSAMPLES; */

/*   return (void*)flops; */
/* } */

int main(int argc, char **argv)
{
  pthread_t tid[2];

  params p;
  p.argc = argc;
  p.argv = argv;

  /* pthread_create(&tid[0], NULL, (void*)func_cholesky, (void*)&p); */
  /* pthread_create(&tid[1], NULL, (void*)func_lu, (void*)&p); */

  /* void *gflops_cholesky; */
  /* void *gflops_lu; */

  /* pthread_join(tid[0], &gflops_cholesky); */
  /* pthread_join(tid[1], &gflops_lu); */

  /* printf("%2.2f %2.2f\n", *((double*)gflops_cholesky), *((double*)gflops_lu)); */


  return 0;
}
