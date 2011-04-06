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

#include "xlu.h"
#include "xlu_kernels.h"

#define TAG11(k)	((starpu_tag_t)( (1ULL<<60) | (unsigned long long)(k)))
#define TAG12(k,i)	((starpu_tag_t)(((2ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))
#define TAG21(k,j)	((starpu_tag_t)(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG22(k,i,j)	((starpu_tag_t)(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))
#define PIVOT(k,i)	((starpu_tag_t)(((5ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))

static unsigned no_prio = 0;

/*
 *	Construct the DAG
 */

static struct starpu_task *create_task(starpu_tag_t id)
{
	struct starpu_task *task = starpu_task_create();
		task->cl_arg = NULL;

	task->use_tag = 1;
	task->tag_id = id;

	return task;
}

static void create_task_pivot(starpu_data_handle *dataAp, unsigned lu_nblocks,
					struct piv_s *piv_description,
					unsigned k, unsigned i,
			      starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned),
			      struct starpu_sched_ctx *sched_ctx)
{
	struct starpu_task *task = create_task(PIVOT(k, i));

	task->cl = &cl_pivot;

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = get_block(dataAp, lu_nblocks, k, i);
	task->buffers[0].mode = STARPU_RW;

	task->cl_arg = &piv_description[k];

	/* this is an important task */
	if (!no_prio && (i == k+1))
		task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k == 0) {
		starpu_tag_declare_deps(PIVOT(k, i), 1, TAG11(k));
	}
	else 
	{
		if (i > k) {
			starpu_tag_declare_deps(PIVOT(k, i), 2, TAG11(k), TAG22(k-1, i, k));
		}
		else {
			starpu_tag_t *tags = malloc((lu_nblocks - k)*sizeof(starpu_tag_t));
			
			tags[0] = TAG11(k);
			unsigned ind, ind2;
			for (ind = k + 1, ind2 = 0; ind < lu_nblocks; ind++, ind2++)
			{
				tags[1 + ind2] = TAG22(k-1, ind, k);
			}

			/* perhaps we could do better ... :/  */
			starpu_tag_declare_deps_array(PIVOT(k, i), (lu_nblocks-k), tags);
		}
	}

	starpu_task_submit_to_ctx(task, sched_ctx);
}

static struct starpu_task *create_task_11_pivot(starpu_data_handle *dataAp, unsigned lu_nblocks,
					unsigned k, struct piv_s *piv_description,
					starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned))
{
	struct starpu_task *task = create_task(TAG11(k));

	task->cl = &cl11_pivot;

	task->cl_arg = &piv_description[k];

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = get_block(dataAp, lu_nblocks, k, k);
	task->buffers[0].mode = STARPU_RW;

	/* this is an important task */
	if (!no_prio)
		task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0) {
		starpu_tag_declare_deps(TAG11(k), 1, TAG22(k-1, k, k));
	}

	return task;
}

static void create_task_12(starpu_data_handle *dataAp, unsigned lu_nblocks, unsigned k, unsigned j,
			   starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned),
			   struct starpu_sched_ctx *sched_ctx)
{
//	printf("task 12 k,i = %d,%d TAG = %llx\n", k,i, TAG12(k,i));

	struct starpu_task *task = create_task(TAG12(k, j));
	
	task->cl = &cl12;

	task->cl_arg = (void *)(task->tag_id);

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = get_block(dataAp, lu_nblocks, k, k);
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = get_block(dataAp, lu_nblocks, j, k);
	task->buffers[1].mode = STARPU_RW;

	if (!no_prio && (j == k+1)) {
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
#if 0
	starpu_tag_declare_deps(TAG12(k, i), 1, PIVOT(k, i));
#endif
	if (k > 0) {
		starpu_tag_declare_deps(TAG12(k, j), 2, TAG11(k), TAG22(k-1, k, j));
	}
	else {
		starpu_tag_declare_deps(TAG12(k, j), 1, TAG11(k));
	}

	starpu_task_submit_to_ctx(task, sched_ctx);
}

static void create_task_21(starpu_data_handle *dataAp, unsigned lu_nblocks, unsigned k, unsigned i,
			   starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned), 
			   struct starpu_sched_ctx *sched_ctx)
{
	struct starpu_task *task = create_task(TAG21(k, i));

	task->cl = &cl21;
	
	/* which sub-data is manipulated ? */
	task->buffers[0].handle = get_block(dataAp, lu_nblocks, k, k); 
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = get_block(dataAp, lu_nblocks, k, i); 
	task->buffers[1].mode = STARPU_RW;

	if (!no_prio && (i == k+1)) {
		task->priority = STARPU_MAX_PRIO;
	}

	task->cl_arg = (void *)(task->tag_id);

	/* enforce dependencies ... */
	starpu_tag_declare_deps(TAG21(k, i), 1, PIVOT(k, i));

	starpu_task_submit_to_ctx(task, sched_ctx);
}

static void create_task_22(starpu_data_handle *dataAp, unsigned lu_nblocks, unsigned k, unsigned i, unsigned j,
			   starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned),
			   struct starpu_sched_ctx *sched_ctx)
{
//	printf("task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG22(k,i,j));

	struct starpu_task *task = create_task(TAG22(k, i, j));

	task->cl = &cl22;

	task->cl_arg = (void *)(task->tag_id);

	/* which sub-data is manipulated ? */
	task->buffers[0].handle = get_block(dataAp, lu_nblocks, k, i); /* produced by TAG21(k, i) */
	task->buffers[0].mode = STARPU_R;
	task->buffers[1].handle = get_block(dataAp, lu_nblocks, j, k); /* produced by TAG12(k, j) */ 
	task->buffers[1].mode = STARPU_R;
	task->buffers[2].handle = get_block(dataAp, lu_nblocks, j, i);  /* produced by TAG22(k-1, i, j) */
	task->buffers[2].mode = STARPU_RW;

	if (!no_prio &&  (i == k + 1) && (j == k +1) ) {
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0) {
		starpu_tag_declare_deps(TAG22(k, i, j), 3, TAG22(k-1, i, j), TAG12(k, j), TAG21(k, i));
	}
	else {
		starpu_tag_declare_deps(TAG22(k, i, j), 2, TAG12(k, j), TAG21(k, i));
	}

	starpu_task_submit_to_ctx(task, sched_ctx);
}

/*
 *	code to bootstrap the factorization 
 */

static double dw_codelet_facto_pivot(starpu_data_handle *dataAp,
					struct piv_s *piv_description,
					unsigned lu_nblocks,
				     starpu_data_handle (* get_block)(starpu_data_handle *, unsigned, unsigned, unsigned), 
				     struct starpu_sched_ctx *sched_ctx)
{
	struct timeval start;
	struct timeval end;

	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	for (k = 0; k < lu_nblocks; k++)
	{
		struct starpu_task *task = create_task_11_pivot(dataAp, lu_nblocks, k, piv_description, get_block);

		/* we defer the launch of the first task */
		if (k == 0) {
			entry_task = task;
		}
		else {
		  starpu_task_submit_to_ctx(task, sched_ctx);
		}

		for (i = 0; i < lu_nblocks; i++)
		{
			if (i != k)
			  create_task_pivot(dataAp, lu_nblocks, piv_description, k, i, get_block, sched_ctx);
		}
	
		for (i = k+1; i<lu_nblocks; i++)
		{
		  create_task_12(dataAp, lu_nblocks, k, i, get_block, sched_ctx);
		  create_task_21(dataAp, lu_nblocks, k, i, get_block, sched_ctx);
		}

		for (i = k+1; i<lu_nblocks; i++)
		{
			for (j = k+1; j<lu_nblocks; j++)
			{
			  create_task_22(dataAp, lu_nblocks, k, i, j, get_block, sched_ctx);
			}
		}
	}

	/* we wait the last task (TAG11(lu_nblocks - 1)) and all the pivot tasks */
	starpu_tag_t *tags = malloc(lu_nblocks*lu_nblocks*sizeof(starpu_tag_t));
	unsigned ndeps = 0;

	tags[ndeps++] = TAG11(lu_nblocks - 1);

	for (j = 0; j < lu_nblocks; j++)
	{
		for (i = 0; i < j; i++)
		{
			tags[ndeps++] = PIVOT(j, i);
		}
	}

	/* schedule the codelet */
	gettimeofday(&start, NULL);
	int ret = starpu_task_submit_to_ctx(entry_task, sched_ctx);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
		fprintf(stderr, "No worker may execute this task\n");
		exit(-1);
	}

	/* stall the application until the end of computations */
	starpu_tag_wait_array(ndeps, tags);
	printf("lu pivot  finish waiting for %d blocks \n", lu_nblocks);
//	starpu_task_wait_for_all();

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	return timing;
}

starpu_data_handle get_block_with_striding(starpu_data_handle *dataAp,
			unsigned lu_nblocks __attribute__((unused)), unsigned j, unsigned i)
{
	/* we use filters */
	return starpu_data_get_sub_data(*dataAp, 2, j, i);
}


double STARPU_LU(lu_decomposition_pivot)(TYPE *matA, unsigned *ipiv, unsigned size, unsigned ld, unsigned lu_nblocks, struct starpu_sched_ctx *sched_ctx)
{
	starpu_data_handle dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	starpu_matrix_data_register(&dataA, 0, (uintptr_t)matA, ld, size, size, sizeof(TYPE));

	/* We already enforce deps by hand */
	starpu_data_set_sequential_consistency_flag(dataA, 0);

	struct starpu_data_filter f;
		f.filter_func = starpu_vertical_block_filter_func;
		f.nchildren = lu_nblocks;

	struct starpu_data_filter f2;
		f2.filter_func = starpu_block_filter_func;
		f2.nchildren = lu_nblocks;

	starpu_data_map_filters(dataA, 2, &f, &f2);

	unsigned i;
	for (i = 0; i < size; i++)
		ipiv[i] = i;

	struct piv_s *piv_description = malloc(lu_nblocks*sizeof(struct piv_s));
	unsigned block;
	for (block = 0; block < lu_nblocks; block++)
	{
		piv_description[block].piv = ipiv;
		piv_description[block].first = block * (size / lu_nblocks);
		piv_description[block].last = (block + 1) * (size / lu_nblocks);
	}

#if 0
	unsigned j;
	for (j = 0; j < lu_nblocks; j++)
	for (i = 0; i < lu_nblocks; i++)
	{
		printf("BLOCK %d %d	%p\n", i, j, &matA[i*(size/lu_nblocks) + j * (size/lu_nblocks)*ld]);
	}
#endif

	double timing;
	timing = dw_codelet_facto_pivot(&dataA, piv_description, lu_nblocks, get_block_with_striding, sched_ctx);

	fprintf(stderr, "Computation took (in ms)\n");
	fprintf(stderr, "%2.2f\n", timing/1000);

	unsigned n = starpu_matrix_get_nx(dataA);
	double flop = (2.0f*n*n*n)/3.0f;
	double gflops = flop/timing/1000.0f;

	/* gather all the data */
	starpu_data_unpartition(dataA, 0);
	return gflops;
}


starpu_data_handle get_block_with_no_striding(starpu_data_handle *dataAp, unsigned lu_nblocks, unsigned j, unsigned i)
{
	/* dataAp is an array of data handle */
	return dataAp[i+j*lu_nblocks];
}

double STARPU_LU(lu_decomposition_pivot_no_stride)(TYPE **matA, unsigned *ipiv, unsigned size, unsigned ld, unsigned lu_nblocks, struct starpu_sched_ctx *sched_ctx)
{
	starpu_data_handle *dataAp = malloc(lu_nblocks*lu_nblocks*sizeof(starpu_data_handle));

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	unsigned bi, bj;
	for (bj = 0; bj < lu_nblocks; bj++)
	for (bi = 0; bi < lu_nblocks; bi++)
	{
		starpu_matrix_data_register(&dataAp[bi+lu_nblocks*bj], 0,
			(uintptr_t)matA[bi+lu_nblocks*bj], size/lu_nblocks,
			size/lu_nblocks, size/lu_nblocks, sizeof(TYPE));

		/* We already enforce deps by hand */
		starpu_data_set_sequential_consistency_flag(dataAp[bi+lu_nblocks*bj], 0);
	}

	unsigned i;
	for (i = 0; i < size; i++)
		ipiv[i] = i;

	struct piv_s *piv_description = malloc(lu_nblocks*sizeof(struct piv_s));
	unsigned block;
	for (block = 0; block < lu_nblocks; block++)
	{
		piv_description[block].piv = ipiv;
		piv_description[block].first = block * (size / lu_nblocks);
		piv_description[block].last = (block + 1) * (size / lu_nblocks);
	}

	double timing;
	timing = dw_codelet_facto_pivot(dataAp, piv_description, lu_nblocks, get_block_with_no_striding, sched_ctx);

	unsigned n = starpu_matrix_get_nx(dataAp[0])*lu_nblocks;
	double flop = (2.0f*n*n*n)/3.0f;
	double gflops = flop/timing/1000.0f;

	for (bj = 0; bj < lu_nblocks; bj++)
	for (bi = 0; bi < lu_nblocks; bi++)
	{
		starpu_data_unregister(dataAp[bi+lu_nblocks*bj]);
	}
	return gflops;
}
