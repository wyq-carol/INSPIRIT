/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010  Université de Bordeaux 1
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <starpu.h>
#include <starpu_profiling.h>
#include <starpu_bound.h>
#include "xlu.h"
#include "xlu_kernels.h"

static unsigned long lu_size = 10240;//4096;
static unsigned lu_nblocks = 10;
static unsigned lu_check = 0;
static unsigned pivot = 0;
static unsigned no_stride = 0;
static unsigned profile = 0;
static unsigned bound = 0;
static unsigned bounddeps = 0;
static unsigned boundprio = 0;


static void lu_parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size") == 0) {
			char *argptr;
			lu_size = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks") == 0) {
			char *argptr;
			lu_nblocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-check") == 0) {
			lu_check = 1;
		}

		if (strcmp(argv[i], "-piv") == 0) {
			pivot = 1;
		}

		if (strcmp(argv[i], "-no-stride") == 0) {
			no_stride = 1;
		}

		if (strcmp(argv[i], "-profile") == 0) {
			profile = 1;
		}

		if (strcmp(argv[i], "-bound") == 0) {
			bound = 1;
		}
		if (strcmp(argv[i], "-bounddeps") == 0) {
			bound = 1;
			bounddeps = 1;
		}
		if (strcmp(argv[i], "-bounddepsprio") == 0) {
			bound = 1;
			bounddeps = 1;
			boundprio = 1;
		}
	}
}

static void display_matrix(TYPE *m, unsigned n, unsigned ld, char *str)
{
#if 0
	fprintf(stderr, "***********\n");
	fprintf(stderr, "Display matrix %s\n", str);
	unsigned i,j;
	for (j = 0; j < n; j++)
	{
		for (i = 0; i < n; i++)
		{
			fprintf(stderr, "%2.2f\t", m[i+j*ld]);
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "***********\n");
#endif
}

void copy_blocks_into_matrix(TYPE *A, TYPE **A_blocks)
{
	unsigned blocklu_size = (lu_size/lu_nblocks);

	unsigned i, j;
	unsigned bi, bj;
	for (bj = 0; bj < lu_nblocks; bj++)
	for (bi = 0; bi < lu_nblocks; bi++)
	{
		for (j = 0; j < blocklu_size; j++)
		for (i = 0; i < blocklu_size; i++)
		{
			A[(i+bi*blocklu_size) + (j + bj*blocklu_size)*lu_size] =
				A_blocks[bi+lu_nblocks*bj][i + j * blocklu_size];
		}

		//free(A_blocks[bi+lu_nblocks*bj]);
	}
}



void copy_matrix_into_blocks(TYPE *A, TYPE **A_blocks)
{
	unsigned blocklu_size = (lu_size/lu_nblocks);

	unsigned i, j;
	unsigned bi, bj;
	for (bj = 0; bj < lu_nblocks; bj++)
	for (bi = 0; bi < lu_nblocks; bi++)
	{
		starpu_data_malloc_pinned_if_possible((void **)&A_blocks[bi+lu_nblocks*bj], (size_t)blocklu_size*blocklu_size*sizeof(TYPE));

		for (j = 0; j < blocklu_size; j++)
		for (i = 0; i < blocklu_size; i++)
		{
			A_blocks[bi+lu_nblocks*bj][i + j * blocklu_size] =
			A[(i+bi*blocklu_size) + (j + bj*blocklu_size)*lu_size];
		}
	}
}

static void init_matrix(TYPE **A)
{
	/* allocate matrix */
	starpu_data_malloc_pinned_if_possible((void **)A, (size_t)lu_size*lu_size*sizeof(TYPE));
	STARPU_ASSERT(*A);

	starpu_srand48((long int)time(NULL));
	//	starpu_srand48(0);

	/* initialize matrix content */
	unsigned long i,j;
	for (j = 0; j < lu_size; j++)
	{
		for (i = 0; i < lu_size; i++)
		{
		  (*A)[i + j*lu_size] = (TYPE)starpu_drand48();
		}
	}

}

static void save_matrix(TYPE *A, TYPE *A_saved)
{
	A_saved = malloc((size_t)lu_size*lu_size*sizeof(TYPE));
	STARPU_ASSERT(A_saved);

	memcpy(A_saved, A, (size_t)lu_size*lu_size*sizeof(TYPE));
}

static double frobenius_norm(TYPE *v, unsigned n)
{
	double sum2 = 0.0;

	/* compute sqrt(Sum(|x|^2)) */

	unsigned i,j;
	for (j = 0; j < n; j++)
	for (i = 0; i < n; i++)
	{
		double a = fabsl((double)v[i+n*j]);
		sum2 += a*a;
	}

	return sqrt(sum2);
}

static void pivot_saved_matrix(unsigned *ipiv, TYPE *A_saved)
{
	unsigned k;
	for (k = 0; k < lu_size; k++)
	{
		if (k != ipiv[k])
		{
	//		fprintf(stderr, "SWAP %d and %d\n", k, ipiv[k]);
			CPU_SWAP(lu_size, &A_saved[k*lu_size], 1, &A_saved[ipiv[k]*lu_size], 1);
		}
	}
}

static void lu_check_result(TYPE *A, TYPE *A_saved)
{
	unsigned i,j;
	TYPE *L, *U;

	L = malloc((size_t)lu_size*lu_size*sizeof(TYPE));
	U = malloc((size_t)lu_size*lu_size*sizeof(TYPE));

	memset(L, 0, lu_size*lu_size*sizeof(TYPE));
	memset(U, 0, lu_size*lu_size*sizeof(TYPE));

	/* only keep the lower part */
	for (j = 0; j < lu_size; j++)
	{
		for (i = 0; i < j; i++)
		{
			L[j+i*lu_size] = A[j+i*lu_size];
		}

		/* diag i = j */
		L[j+j*lu_size] = A[j+j*lu_size];
		U[j+j*lu_size] = 1.0;

		for (i = j+1; i < lu_size; i++)
		{
			U[j+i*lu_size] = A[j+i*lu_size];
		}
	}

	display_matrix(L, lu_size, lu_size, "L");
	display_matrix(U, lu_size, lu_size, "U");

	/* now A_err = L, compute L*U */
	CPU_TRMM("R", "U", "N", "U", lu_size, lu_size, 1.0f, U, lu_size, L, lu_size);

	display_matrix(A_saved, lu_size, lu_size, "P A_saved");
	display_matrix(L, lu_size, lu_size, "LU");

	/* compute "LU - A" in L*/
	CPU_AXPY(lu_size*lu_size, -1.0, A_saved, 1, L, 1);
	display_matrix(L, lu_size, lu_size, "Residuals");
	
	TYPE err = CPU_ASUM(lu_size*lu_size, L, 1);
	int max = CPU_IAMAX(lu_size*lu_size, L, 1);

	fprintf(stderr, "Avg error : %e\n", err/(lu_size*lu_size));
	fprintf(stderr, "Max error : %e\n", L[max]);

	double residual = frobenius_norm(L, lu_size);
	double matnorm = frobenius_norm(A_saved, lu_size);

	fprintf(stderr, "||%sA-LU|| / (||A||*N) : %e\n", pivot?"P":"", residual/(matnorm*lu_size));

	if (residual/(matnorm*lu_size) > 1e-5)
		exit(-1);
}

double run_lu(struct starpu_sched_ctx *sched_ctx, int argc, char **argv)
{
	TYPE *A;
	TYPE *A_saved;

	/* in case we use non-strided blocks */
	TYPE **A_blocks;

	lu_parse_args(argc, argv);

	//	starpu_init(NULL);

	//	starpu_helper_cublas_init();

	init_matrix(&A);

	unsigned *ipiv;
	if (lu_check)
	  save_matrix(A, A_saved);

	display_matrix(A, lu_size, lu_size, "A");

	if (bound)
		starpu_bound_start(bounddeps, boundprio);

	if (profile)
		starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

	double gflops = -1;
	/* Factorize the matrix (in place) */
	if (pivot)
	{
 		ipiv = malloc(lu_size*sizeof(unsigned));
		if (no_stride)
		{
			/* in case the LU decomposition uses non-strided blocks, we _copy_ the matrix into smaller blocks */
			A_blocks = malloc(lu_nblocks*lu_nblocks*sizeof(TYPE **));
			copy_matrix_into_blocks(A, A_blocks);

			gflops = STARPU_LU(lu_decomposition_pivot_no_stride)(A_blocks, ipiv, lu_size, lu_size, lu_nblocks, sched_ctx);

			copy_blocks_into_matrix(A, A_blocks);
			free(A_blocks);
		}
		else 
		{
			struct timeval start;
			struct timeval end;

			gettimeofday(&start, NULL);

			gflops = STARPU_LU(lu_decomposition_pivot)(A, ipiv, lu_size, lu_size, lu_nblocks, sched_ctx);
	
			gettimeofday(&end, NULL);

			double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
			
			unsigned n = lu_size;
			double flop = (2.0f*n*n*n)/3.0f;
			gflops = flop/timing/1000.0f;
		}
	}
	else
	{
		gflops = STARPU_LU(lu_decomposition)(A, lu_size, lu_size, lu_nblocks, sched_ctx);
	}

	if (profile)
	{
		starpu_profiling_status_set(STARPU_PROFILING_DISABLE);
		starpu_bus_profiling_helper_display_summary();
	}

	if (bound) {
		double min;
		starpu_bound_stop();
		if (bounddeps) {
			FILE *f = fopen("lu.pl", "w");
			starpu_bound_print_lp(f);
			fprintf(stderr,"system printed to lu.pl\n");
		} else {
			starpu_bound_compute(&min, NULL, 0);
			if (min != 0.)
				fprintf(stderr, "theoretical min: %lf ms\n", min);
		}
	}

	if (lu_check)
	{
		if (pivot)
		  pivot_saved_matrix(ipiv, A_saved);

		lu_check_result(A, A_saved);
	}

	//	starpu_helper_cublas_shutdown();

	//	starpu_shutdown();

	return gflops;
}


