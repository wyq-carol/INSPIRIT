/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2011  Université de Bordeaux 1
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

#include "cholesky.h"

/* A [ y ] [ x ] */
float *A[NMAXBLOCKS][NMAXBLOCKS];
starpu_data_handle A_state[NMAXBLOCKS][NMAXBLOCKS];
/* struct timeval start; */
/* struct timeval end; */


/*
 *	Some useful functions
 */

static struct starpu_task *create_task(starpu_tag_t id)
{
	struct starpu_task *task = starpu_task_create();
		task->cl_arg = NULL;
		task->use_tag = 1;
		task->tag_id = id;

	return task;
}

/*
 *	Create the codelets
 */

static starpu_codelet cl11 =
{
	.where = STARPU_CPU|STARPU_CUDA|STARPU_GORDON,
	.cpu_func = chol_cpu_codelet_update_u11,
#ifdef STARPU_USE_CUDA
	.cuda_func = chol_cublas_codelet_update_u11,
#endif
#ifdef STARPU_USE_GORDON
#ifdef SPU_FUNC_POTRF
	.gordon_func = SPU_FUNC_POTRF,
#else
#warning SPU_FUNC_POTRF is not available
#endif
#endif
	.nbuffers = 1,
	.model = &chol_model_11
};

static struct starpu_task * create_task_11(unsigned k, unsigned nblocks)
{
//	printf("task 11 k = %d TAG = %llx\n", k, (TAG11(k)));

	struct starpu_task *task = create_task(TAG11(k));
	
	task->cl = &cl11;

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = A_state[k][k];
	task->buffers[0].mode = STARPU_RW;

	/* this is an important task */
	task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0) {
		starpu_tag_declare_deps(TAG11(k), 1, TAG22(k-1, k, k));
	}

	return task;
}

static starpu_codelet cl21 =
{
	.where = STARPU_CPU|STARPU_CUDA|STARPU_GORDON,
	.cpu_func = chol_cpu_codelet_update_u21,
#ifdef STARPU_USE_CUDA
	.cuda_func = chol_cublas_codelet_update_u21,
#endif
#ifdef STARPU_USE_GORDON
#ifdef SPU_FUNC_STRSM
	.gordon_func = SPU_FUNC_STRSM,
#else
#warning SPU_FUNC_STRSM is not available
#endif
#endif
	.nbuffers = 2,
	.model = &chol_model_21
};

static void create_task_21(unsigned k, unsigned j)
{
	struct starpu_task *task = create_task(TAG21(k, j));

	task->cl = &cl21;	

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = A_state[k][k]; 
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = A_state[j][k]; 
	task->buffers[1].mode = STARPU_RW;

	if (j == k+1) {
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0) {
		starpu_tag_declare_deps(TAG21(k, j), 2, TAG11(k), TAG22(k-1, k, j));
	}
	else {
		starpu_tag_declare_deps(TAG21(k, j), 1, TAG11(k));
	}

	starpu_task_submit(task);
}

static starpu_codelet cl22 =
{
	.where = STARPU_CPU|STARPU_CUDA|STARPU_GORDON,
	.cpu_func = chol_cpu_codelet_update_u22,
#ifdef STARPU_USE_CUDA
	.cuda_func = chol_cublas_codelet_update_u22,
#endif
#ifdef STARPU_USE_GORDON
#ifdef SPU_FUNC_SGEMM
	.gordon_func = SPU_FUNC_SGEMM,
#else
#warning SPU_FUNC_SGEMM is not available
#endif
#endif
	.nbuffers = 3,
	.model = &chol_model_22
};

static void create_task_22(unsigned k, unsigned i, unsigned j)
{
//	printf("task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG22(k,i,j));

	struct starpu_task *task = create_task(TAG22(k, i, j));

	task->cl = &cl22;

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = A_state[i][k]; 
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = A_state[j][k]; 
	task->buffers[1].mode = STARPU_R;
	task->buffers[2].handle = A_state[j][i]; 
	task->buffers[2].mode = STARPU_RW;

	if ( (i == k + 1) && (j == k +1) ) {
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0) {
		starpu_tag_declare_deps(TAG22(k, i, j), 3, TAG22(k-1, i, j), TAG21(k, i), TAG21(k, j));
	}
	else {
		starpu_tag_declare_deps(TAG22(k, i, j), 2, TAG21(k, i), TAG21(k, j));
	}

	starpu_task_submit(task);
}



/*
 *	code to bootstrap the factorization 
 *	and construct the DAG
 */

static double cholesky_no_stride()
{
	struct timeval start;
	struct timeval end;

	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	for (k = 0; k < nblocks; k++)
	{
		struct starpu_task *task = create_task_11(k, nblocks);
		/* we defer the launch of the first task */
		if (k == 0) {
			entry_task = task;
		}
		else {
		  starpu_task_submit(task);
		}
		
		for (j = k+1; j<nblocks; j++)
		{
		  create_task_21(k, j);

			for (i = k+1; i<nblocks; i++)
			{
				if (i <= j)
				  create_task_22(k, i, j);
			}
		}
	}

	/* schedule the codelet */
	gettimeofday(&start, NULL);
	starpu_task_submit(entry_task);

	/* stall the application until the end of computations */
	starpu_tag_wait(TAG11(nblocks-1));

	gettimeofday(&end, NULL);

        double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
        double flop = (1.0f*size*size*size)/3.0f;
        return flop/timing/1000.0f;
}

double run_cholesky_tile_tag(int argc, char **argv)
{
	unsigned x, y;
	unsigned i, j;

	parse_args(argc, argv);
	assert(nblocks <= NMAXBLOCKS);

	//fprintf(stderr, "BLOCK SIZE = %d\n", size / nblocks);

	starpu_init(NULL);

	/* Disable sequential consistency */
	starpu_data_set_default_sequential_consistency_flag(0);

	starpu_helper_cublas_init();

	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	{
		if (x <= y) {
			A[y][x] = malloc(BLOCKSIZE*BLOCKSIZE*sizeof(float));
			assert(A[y][x]);
		}
	}


	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	{
		if (x <= y) {
#ifdef STARPU_HAVE_POSIX_MEMALIGN
			posix_memalign((void **)&A[y][x], 128, BLOCKSIZE*BLOCKSIZE*sizeof(float));
#else
			A[y][x] = malloc(BLOCKSIZE*BLOCKSIZE*sizeof(float));
#endif
			assert(A[y][x]);
		}
	}

	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1) ( + n In to make is stable ) 
	 * */
	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	if (x <= y) {
		for (i = 0; i < BLOCKSIZE; i++)
		for (j = 0; j < BLOCKSIZE; j++)
		{
			A[y][x][i*BLOCKSIZE + j] =
				(float)(1.0f/((float) (1.0+(x*BLOCKSIZE+i)+(y*BLOCKSIZE+j))));

			/* make it a little more numerically stable ... ;) */
			if ((x == y) && (i == j))
				A[y][x][i*BLOCKSIZE + j] += (float)(2*size);
		}
	}



	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	{
		if (x <= y) {
			starpu_matrix_data_register(&A_state[y][x], 0, (uintptr_t)A[y][x], 
				BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, sizeof(float));
		}
	}

	double gflops = cholesky_no_stride();

	starpu_shutdown();
	return gflops;
}
