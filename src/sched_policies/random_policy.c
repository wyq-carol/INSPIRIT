/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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

/* Policy attributing tasks randomly to workers */

#include <core/workers.h>
#include <sched_policies/fifo_queues.h>

//static unsigned nworkers;

static pthread_cond_t sched_cond[STARPU_NMAXWORKERS];
static pthread_mutex_t sched_mutex[STARPU_NMAXWORKERS];

static int _random_push_task(struct starpu_task *task, unsigned prio, struct starpu_sched_ctx *sched_ctx)
{
	/* find the queue */
        unsigned worker, worker_in_ctx;

	unsigned selected = 0;

	double alpha_sum = 0.0;

	unsigned nworkers = sched_ctx->nworkers_in_ctx;	

	for (worker_in_ctx = 0; worker_in_ctx < nworkers; worker_in_ctx++)
	{
                worker = sched_ctx->workerid[worker_in_ctx];

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		alpha_sum += starpu_worker_get_relative_speedup(perf_arch);
	}

	double random = starpu_drand48()*alpha_sum;
//	_STARPU_DEBUG("my rand is %e\n", random);

	double alpha = 0.0;
	for (worker_in_ctx = 0; worker_in_ctx < nworkers; worker_in_ctx++)
	{
                worker = sched_ctx->workerid[worker_in_ctx];

		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		double worker_alpha = starpu_worker_get_relative_speedup(perf_arch);

		if (alpha + worker_alpha > random) {
			/* we found the worker */
			selected = worker;
			break;
		}

		alpha += worker_alpha;
	}

	/* we should now have the best worker in variable "selected" */
	_starpu_increment_nsubmitted_tasks_of_worker(selected);
	int n = starpu_push_local_task(selected, task, prio);
	return n;
}

static int random_push_prio_task(struct starpu_task *task, int sched_ctx_id)
{	
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);

        return _random_push_task(task, 1, sched_ctx);
}

static int random_push_task(struct starpu_task *task, int sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);

        return _random_push_task(task, 0, sched_ctx);
}

static void initialize_random_policy(int sched_ctx_id) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);

	starpu_srand48(time(NULL));

	unsigned nworkers = sched_ctx->nworkers_in_ctx;	

	unsigned workerid, workerid_ctx;
	for (workerid_ctx = 0; workerid_ctx < nworkers; workerid_ctx++)
	{
                workerid = sched_ctx->workerid[workerid_ctx];
	
		PTHREAD_MUTEX_INIT(&sched_mutex[workerid], NULL);
		PTHREAD_COND_INIT(&sched_cond[workerid], NULL);
	
		starpu_worker_set_sched_condition(workerid, &sched_cond[workerid], &sched_mutex[workerid]);
	}
}

struct starpu_sched_policy_s _starpu_sched_random_policy = {
	.init_sched = initialize_random_policy,
	.deinit_sched = NULL,
	.push_task = random_push_task,
	.push_prio_task = random_push_prio_task,
	.push_task_notify = NULL,
	.pop_task = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "random",
	.policy_description = "weighted random"
};
