/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
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
#include <sys/time.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#if !defined(STARPU_HAVE_HWLOC)
#warning hwloc is not enabled. Skipping test
int main(int argc, char **argv)
{
	return 77;
}
#else

int main()
{
	starpu_init(NULL);

	int procs[STARPU_NMAXWORKERS];
	unsigned ncpus =  starpu_cpu_worker_get_count();
        starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, procs, ncpus);

	struct starpu_worker_collection *co = (struct starpu_worker_collection*)malloc(sizeof(struct starpu_worker_collection));
	co->has_next = worker_tree.has_next;
	co->get_next = worker_tree.get_next;
	co->add = worker_tree.add;
	co->remove = worker_tree.remove;
	co->init = worker_tree.init;
	co->deinit = worker_tree.deinit;
	co->init_iterator = worker_tree.init_iterator;
	co->type = STARPU_WORKER_TREE;

	FPRINTF(stderr, "ncpus %d \n", ncpus);

	struct timeval start_time;
        struct timeval end_time;
        gettimeofday(&start_time, NULL);

	co->init(co);

	gettimeofday(&end_time, NULL);

        long diff_s = end_time.tv_sec  - start_time.tv_sec;
        long diff_us = end_time.tv_usec  - start_time.tv_usec;

	float timing = (float)(diff_s*1000000 + diff_us)/1000;

	int i;
	for(i = 0; i < ncpus; i++)
	{
		int added = co->add(co, procs[i]);
//		FPRINTF(stderr, "added proc %d to the tree \n", added);
	}

	struct starpu_sched_ctx_iterator it;
        if(co->init_iterator)
                co->init_iterator(co, &it);

	int pu;
	while(co->has_next(co, &it))
	{
		pu = co->get_next(co, &it);
//		FPRINTF(stderr, "pu = %d out of %d workers \n", pu, co->nworkers);
	}

	for(i = 0; i < 6; i++)
	{
		co->remove(co, i);
//		FPRINTF(stderr, "remove %d out of %d workers\n", i, co->nworkers);
	}

	while(co->has_next(co, &it))
	{
		pu = co->get_next(co, &it);
//		FPRINTF(stderr, "pu = %d out of %d workers \n", pu, co->nworkers);
	}

	FPRINTF(stderr, "timing init = %lf \n", timing);

	co->deinit(co);
	starpu_shutdown();
}
#endif
