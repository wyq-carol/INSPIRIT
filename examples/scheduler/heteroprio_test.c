/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015  INRIA
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
#include <schedulers/starpu_heteroprio.h>
#include <unistd.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void initSchedulerCallback()
{
	// CPU uses 3 buckets
	starpu_heteroprio_set_nb_prios(0, STARPU_CPU_IDX, 3);
	// It uses direct mapping idx => idx
	unsigned idx;
	for(idx = 0; idx < 3; ++idx)
	{
		starpu_heteroprio_set_mapping(0, STARPU_CPU_IDX, idx, idx);
		starpu_heteroprio_set_faster_arch(0, STARPU_CPU_IDX, idx);
	}
#ifdef STARPU_USE_OPENCL
	// OpenCL is enabled and uses 2 buckets
	starpu_heteroprio_set_nb_prios(0, STARPU_OPENCL_IDX, 2);
	// OpenCL will first look to priority 2
	starpu_heteroprio_set_mapping(0, STARPU_OPENCL_IDX, 0, 2);
	// For this bucket OpenCL is the fastest
	starpu_heteroprio_set_faster_arch(0, STARPU_OPENCL_IDX, 2);
	// And CPU is 4 times slower
	starpu_heteroprio_set_arch_slow_factor(0, STARPU_CPU_IDX, 2, 4.0f);

	starpu_heteroprio_set_mapping(0, STARPU_OPENCL_IDX, 1, 1);
	// We let the CPU as the fastest and tell that OpenCL is 1.7 times slower
	starpu_heteroprio_set_arch_slow_factor(0, STARPU_OPENCL_IDX, 1, 1.7f);
#endif
}

void callback_a_cpu(void *buffers[], void *cl_arg)
{
	usleep(100000);
	FPRINTF(stderr, "[COMMUTE_LOG] callback %s\n", __FUNCTION__); fflush(stdout);
}

void callback_b_cpu(void *buffers[], void *cl_arg)
{
	usleep(100000);
	FPRINTF(stderr, "[COMMUTE_LOG] callback %s\n", __FUNCTION__); fflush(stdout);
}

void callback_c_cpu(void *buffers[], void *cl_arg)
{
	usleep(100000);
	FPRINTF(stderr, "[COMMUTE_LOG] callback %s\n", __FUNCTION__); fflush(stdout);
}

#ifdef STARPU_USE_OPENCL
void callback_a_opencl(void *buffers[], void *cl_arg)
{
	usleep(100000);
	FPRINTF(stderr, "[COMMUTE_LOG] callback %s\n", __FUNCTION__); fflush(stdout);
}

void callback_b_opencl(void *buffers[], void *cl_arg)
{
	usleep(100000);
	FPRINTF(stderr, "[COMMUTE_LOG] callback %s\n", __FUNCTION__); fflush(stdout);
}

void callback_c_opencl(void *buffers[], void *cl_arg)
{
	usleep(100000);
	FPRINTF(stderr, "[COMMUTE_LOG] callback %s\n", __FUNCTION__); fflush(stdout);
}
#endif

int main(int argc, char** argv)
{
	unsigned ret;
	struct starpu_conf conf;
	ret = starpu_conf_init(&conf);
	assert(ret == 0);

	conf.sched_policy_name = "heteroprio";
	conf.sched_policy_init = &initSchedulerCallback;

	ret = starpu_init(&conf);
	assert(ret == 0);

	starpu_pause();

	FPRINTF(stderr, "Worker = %d\n",  starpu_worker_get_count());
	FPRINTF(stderr, "Worker CPU = %d\n", starpu_cpu_worker_get_count());
#ifdef STARPU_USE_OPENCL
	FPRINTF(stderr, "Worker OpenCL = %d\n", starpu_opencl_worker_get_count());
#endif

	struct starpu_codelet codeleteA;
	{
		memset(&codeleteA, 0, sizeof(codeleteA));
		codeleteA.nbuffers = 2;
		codeleteA.modes[0] = STARPU_RW;
		codeleteA.modes[1] = STARPU_RW;
		codeleteA.name = "codeleteA";
		codeleteA.where = STARPU_CPU;
		codeleteA.cpu_funcs[0] = callback_a_cpu;
#ifdef STARPU_USE_OPENCL
		codeleteA.where |= STARPU_OPENCL;
		codeleteA.opencl_funcs[0] = callback_a_opencl;
#endif
	}
	struct starpu_codelet codeleteB;
	{
		memset(&codeleteB, 0, sizeof(codeleteB));
		codeleteB.nbuffers = 2;
		codeleteB.modes[0] = STARPU_RW;
		codeleteB.modes[1] = STARPU_RW;
		codeleteB.name = "codeleteB";
		codeleteB.where = STARPU_CPU;
		codeleteB.cpu_funcs[0] = callback_b_cpu;
#ifdef STARPU_USE_OPENCL
		codeleteB.where |= STARPU_OPENCL;
		codeleteB.opencl_funcs[0] = callback_b_opencl;
#endif
	}
	struct starpu_codelet codeleteC;
	{
		memset(&codeleteC, 0, sizeof(codeleteC));
		codeleteC.nbuffers = 2;
		codeleteC.modes[0] = STARPU_RW;
		codeleteC.modes[1] = STARPU_RW;
		codeleteC.name = "codeleteC";
		codeleteC.where = STARPU_CPU;
		codeleteC.cpu_funcs[0] = callback_c_cpu;
#ifdef STARPU_USE_OPENCL
		codeleteC.where |= STARPU_OPENCL;
		codeleteC.opencl_funcs[0] = callback_c_opencl;
#endif
	}

	const int nbHandles = 10;
	FPRINTF(stderr, "Nb handles = %d\n", nbHandles);

	starpu_data_handle_t handles[nbHandles];
	memset(handles, 0, sizeof(handles[0])*nbHandles);
	int dataA[nbHandles];
	int idx;
	for(idx = 0; idx < nbHandles; ++idx){
		dataA[idx] = idx;
	}
	int idxHandle;
	for(idxHandle = 0; idxHandle < nbHandles; ++idxHandle){
		starpu_variable_data_register(&handles[idxHandle], 0, (uintptr_t)&dataA[idxHandle], sizeof(dataA[idxHandle]));
	}

	const int nbTasks = 40;
	FPRINTF(stderr, "Submit %d tasks \n", nbTasks);

	starpu_resume();

	int idxTask;
	for(idxTask = 0; idxTask < nbTasks; ++idxTask)
	{
		starpu_insert_task(&codeleteA,
				   STARPU_PRIORITY, 0,
				   (STARPU_RW), handles[(idxTask*2)%nbHandles],
				   (STARPU_RW), handles[(idxTask*3+1)%nbHandles],
				   0);
		starpu_insert_task(&codeleteB,
				   STARPU_PRIORITY, 1,
				   (STARPU_RW), handles[(idxTask*2 +1 )%nbHandles],
				   (STARPU_RW), handles[(idxTask*2)%nbHandles],
				   0);
		starpu_insert_task(&codeleteC,
				   STARPU_PRIORITY, 2,
				   (STARPU_RW), handles[(idxTask)%nbHandles],
				   (STARPU_RW), handles[(idxTask*idxTask)%nbHandles],
				   0);
	}

	FPRINTF(stderr, "Wait task\n");
	starpu_task_wait_for_all();

	FPRINTF(stderr, "Release data\n");
	for(idxHandle = 0 ; idxHandle < nbHandles ; ++idxHandle)
	{
		starpu_data_unregister(handles[idxHandle]);
	}

	FPRINTF(stderr, "Shutdown\n");

	starpu_shutdown();
	return 0;
}