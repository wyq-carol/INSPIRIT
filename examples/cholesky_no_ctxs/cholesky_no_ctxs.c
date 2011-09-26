/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  INRIA
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include "cholesky/cholesky.h"
#include <pthread.h>

typedef struct {
  int start;
  int argc;
  char **argv;
} params;

typedef struct {
  double flops;
  double avg_timing;
} retvals;

#define NSAMPLES 3
pthread_barrier_t barrier;

void* func_cholesky3(void *val){
  params *p = (params*)val;

  int i;
  retvals *rv  = (retvals*)malloc(sizeof(retvals));
  rv->flops = 0;
  rv->avg_timing = 0;
  double timing = 0;

  for(i = 0; i < NSAMPLES; i++)
    {
      rv->flops += run_cholesky_implicit(p->start, p->argc, p->argv, &timing, &barrier);
      rv->avg_timing += timing;
    }

  rv->flops /= NSAMPLES;
  rv->avg_timing /= NSAMPLES;

  return (void*)rv;
}

void cholesky_vs_cholesky(params *p1, params *p2, unsigned chole1, unsigned chole2){
  if(!chole1 && !chole2) 
    {
      /* 2 cholesky in a single ctx */
      starpu_init(NULL);
      starpu_helper_cublas_init();
      
      pthread_t tid2[2];
      pthread_barrier_init(&barrier, NULL, 2);
      
      struct timeval start;
      struct timeval end;
      
      gettimeofday(&start, NULL);
  
      pthread_create(&tid2[0], NULL, (void*)func_cholesky3, (void*)p1);
      pthread_create(&tid2[1], NULL, (void*)func_cholesky3, (void*)p2);
      
      void *gflops_cholesky4;
      void *gflops_cholesky5;
      
      pthread_join(tid2[0], &gflops_cholesky4);
      pthread_join(tid2[1], &gflops_cholesky5);
      
      gettimeofday(&end, NULL);
      
      starpu_helper_cublas_shutdown();
      starpu_shutdown();
      double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
      timing /= 1000000;
      //      timing /= 60;
      
      printf("%2.2f %2.2f ", ((retvals*)gflops_cholesky4)->flops, ((retvals*)gflops_cholesky5)->flops);
      printf("%2.2f %2.2f %2.2f\n", ((retvals*)gflops_cholesky4)->avg_timing, ((retvals*)gflops_cholesky5)->avg_timing, timing);
      free(gflops_cholesky4);
      free(gflops_cholesky5);
    }
  else if(chole1 && !chole2)
    {
      starpu_init(NULL);
      starpu_helper_cublas_init();
      
      pthread_t tid2[2];
      pthread_barrier_init(&barrier, NULL, 2);
      
      struct timeval start;
      struct timeval end;
      
      gettimeofday(&start, NULL);
  
      pthread_create(&tid2[0], NULL, (void*)func_cholesky3, (void*)p1);
      
      void *gflops_cholesky4;
      
      pthread_join(tid2[0], &gflops_cholesky4);
      
      gettimeofday(&end, NULL);
      
      starpu_helper_cublas_shutdown();
      starpu_shutdown();
      double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
      timing /= 1000000;
      //      timing /= 60;

      printf("%2.2f %2.2f ", ((retvals*)gflops_cholesky4)->flops, 0.0);
      printf("%2.2f %2.2f %2.2f\n", ((retvals*)gflops_cholesky4)->avg_timing, 0.0, timing);
      free(gflops_cholesky4);
    }
  else if(!chole1 && chole2)
    {
      starpu_init(NULL);
      starpu_helper_cublas_init();
      
      pthread_t tid2[2];
      pthread_barrier_init(&barrier, NULL, 2);
      
      struct timeval start;
      struct timeval end;
      
      gettimeofday(&start, NULL);
  
      pthread_create(&tid2[1], NULL, (void*)func_cholesky3, (void*)p2);
      
      void *gflops_cholesky5;
     
      pthread_join(tid2[1], &gflops_cholesky5);
      
      gettimeofday(&end, NULL);
      
      starpu_helper_cublas_shutdown();
      starpu_shutdown();
      double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
      timing /= 1000000;
      //      timing /= 60;
      
      printf("%2.2f %2.2f ", 0.0, ((retvals*)gflops_cholesky5)->flops);
    
      printf("%2.2f %2.2f %2.2f\n", 0.0, ((retvals*)gflops_cholesky5)->avg_timing, timing);

      free(gflops_cholesky5);
    }
}


int main(int argc, char **argv)
{
  unsigned chole1 = 0, chole2 = 0;
  params p1;
  p1.start = 1;
  p1.argc = 5;
  p1.argv = argv;

  params p2;
  p2.start = 5;
  p2.argc = 9;
  p2.argv = argv;
 
  int i;
  for (i = 9; i < argc; i++) {
    if (strcmp(argv[i], "-chole1") == 0) {
      chole1 = 1;
    }
    
    if (strcmp(argv[i], "-chole2") == 0) {
      chole2 = 1;
    }    
  }

  cholesky_vs_cholesky(&p1, &p2, chole1, chole2);

  return 0;
}
