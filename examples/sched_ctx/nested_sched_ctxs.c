/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux
 * Copyright (C) 2010-2014  Centre National de la Recherche Scientifique
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

#include <starpu.h>
#include <omp.h>

#ifdef STARPU_QUICK_CHECK
#define NTASKS 64
#else
#define NTASKS 100
#endif

int tasks_executed[2];
starpu_pthread_mutex_t mut;

int parallel_code(int sched_ctx)
{
	int i;
	int t = 0;
	int *cpuids = NULL;
	int ncpuids = 0;
	starpu_sched_ctx_get_available_cpuids(sched_ctx, &cpuids, &ncpuids);

//	printf("execute task of %d threads \n", ncpuids);
#pragma omp parallel num_threads(ncpuids)
	{
		starpu_sched_ctx_bind_current_thread_to_cpuid(cpuids[omp_get_thread_num()]);
// 			printf("cpu = %d ctx%d nth = %d\n", sched_getcpu(), sched_ctx, omp_get_num_threads());
#pragma omp for
		for(i = 0; i < NTASKS; i++)
			t++;
	}

	free(cpuids);
	return t;
}

static void sched_ctx_func(void *descr[] STARPU_ATTRIBUTE_UNUSED, void *arg)
{
	int w = starpu_worker_get_id();
	unsigned sched_ctx = (unsigned)arg;
	int n = parallel_code(sched_ctx);
//	printf("w %d executed %d it \n", w, n);
}


static struct starpu_codelet sched_ctx_codelet =
{
	.cpu_funcs = {sched_ctx_func, NULL},
	.cuda_funcs = {NULL},
	.opencl_funcs = {NULL},
	.model = NULL,
	.nbuffers = 0,
	.name = "sched_ctx"
};

int main(int argc, char **argv)
{
	tasks_executed[0] = 0;
	tasks_executed[1] = 0;
	int ntasks = NTASKS;
	int ret, j, k;
	unsigned ncpus = 0;

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_pthread_mutex_init(&mut, NULL);
	int nprocs1 = 1;
	int nprocs2 = 1;
	int *procs1, *procs2;

#ifdef STARPU_USE_CPU
	ncpus =  starpu_cpu_worker_get_count();
	procs1 = (int*)malloc(ncpus*sizeof(int));
	starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs1, ncpus);

	if (ncpus > 1)
	{
		nprocs1 = ncpus/2;
		nprocs2 =  nprocs1;
		k = 0;
		procs2 = (int*)malloc(nprocs2*sizeof(int));
		for(j = nprocs1; j < nprocs1+nprocs2; j++)
			procs2[k++] = procs1[j];
	}
	else
	{
		procs2 = (int*)malloc(nprocs2*sizeof(int));
		procs2[0] = procs1[0];
	}
#endif

	if (ncpus == 0)
	{
#ifdef STARPU_USE_CPU
		free(procs1);
		free(procs2);
#endif
		starpu_shutdown();
		return 77;
	}

	/*create contexts however you want*/
	unsigned sched_ctx1 = starpu_sched_ctx_create(procs1, nprocs1, "ctx1", STARPU_SCHED_CTX_POLICY_NAME, "eager", 0);
	unsigned sched_ctx2 = starpu_sched_ctx_create(procs2, nprocs2, "ctx2", STARPU_SCHED_CTX_POLICY_NAME, "dmda", 0);

	/*indicate what to do with the resources when context 2 finishes (it depends on your application)*/
//	starpu_sched_ctx_set_inheritor(sched_ctx2, sched_ctx1);

	int nprocs3 = nprocs1/2;
	int nprocs4 = nprocs1/2;
	int nprocs5 = nprocs2/2;
	int nprocs6 = nprocs2/2;
	int procs3[nprocs3];
	int procs4[nprocs4];
	int procs5[nprocs5];
	int procs6[nprocs6];

	k = 0;
	for(j = 0; j < nprocs3; j++)
		procs3[k++] = procs1[j];
	k = 0;
	for(j = nprocs3; j < nprocs3+nprocs4; j++)
		procs4[k++] = procs1[j];

	k = 0;
	for(j = 0; j < nprocs5; j++)
		procs5[k++] = procs2[j];
	k = 0;
	for(j = nprocs5; j < nprocs5+nprocs6; j++)
		procs6[k++] = procs2[j];

	unsigned sched_ctx3 = starpu_sched_ctx_create(procs3, nprocs3, "ctx3", STARPU_SCHED_CTX_NESTED, sched_ctx1, 0);
	unsigned sched_ctx4 = starpu_sched_ctx_create(procs4, nprocs4, "ctx4", STARPU_SCHED_CTX_NESTED, sched_ctx1, 0);

	unsigned sched_ctx5 = starpu_sched_ctx_create(procs5, nprocs5, "ctx5", STARPU_SCHED_CTX_NESTED, sched_ctx2, 0);
	unsigned sched_ctx6 = starpu_sched_ctx_create(procs6, nprocs6, "ctx6", STARPU_SCHED_CTX_NESTED, sched_ctx2, 0);


	int i;
	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &sched_ctx_codelet;
		task->cl_arg = sched_ctx1;

		/*submit tasks to context*/
		ret = starpu_task_submit_to_ctx(task,sched_ctx1);

		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	for (i = 0; i < ntasks; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &sched_ctx_codelet;
		task->cl_arg = sched_ctx2;

		/*submit tasks to context*/
		ret = starpu_task_submit_to_ctx(task,sched_ctx2);

		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}


	/* tell starpu when you finished submitting tasks to this context
	   in order to allow moving resources from this context to the inheritor one
	   when its corresponding tasks finished executing */



	/* wait for all tasks at the end*/
	starpu_task_wait_for_all();

	starpu_sched_ctx_delete(sched_ctx3);
	starpu_sched_ctx_delete(sched_ctx4);

	starpu_sched_ctx_delete(sched_ctx5);
	starpu_sched_ctx_delete(sched_ctx6);

	starpu_sched_ctx_delete(sched_ctx1);
	starpu_sched_ctx_delete(sched_ctx2);

	printf("ctx%d: tasks starpu executed %d out of %d\n", sched_ctx1, tasks_executed[0], NTASKS);
	printf("ctx%d: tasks starpu executed %d out of %d\n", sched_ctx2, tasks_executed[1], NTASKS);

#ifdef STARPU_USE_CPU
	free(procs1);
	free(procs2);
#endif
	starpu_shutdown();
	return 0;
}
