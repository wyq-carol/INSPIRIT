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

#ifndef __SCHED_CONTEXT_H__
#define __SCHED_CONTEXT_H__

#include <starpu.h>
#include <starpu_sched_ctx.h>
#include <starpu_scheduler.h>
#include <common/config.h>
#include <common/barrier_counter.h>
#include <profiling/profiling.h>

#define NO_RESIZE -1
#define REQ_RESIZE 0
#define DO_RESIZE 1

/* used when changes (delete, modify) are applyed to contexts */
_starpu_pthread_mutex_t changing_ctx_mutex[STARPU_NMAX_SCHED_CTXS];

struct _starpu_sched_ctx
{
	/* id of the context used in user mode*/
	unsigned id;

	/* name of context */
	const char *name;

	/* policy of the context */
	struct starpu_sched_policy *sched_policy;

	/* data necessary for the policy */
	void *policy_data;

	struct starpu_sched_ctx_worker_collection *workers;

	/* we keep an initial sched which we never delete */
	unsigned is_initial_sched;

	/* wait for the tasks submitted to the context to be executed */
	struct _starpu_barrier_counter tasks_barrier;

	/* table of sched cond corresponding to each worker in this ctx */
	_starpu_pthread_cond_t **sched_cond;

	/* table of sched mutex corresponding to each worker in this ctx */
	_starpu_pthread_mutex_t **sched_mutex;

	/* cond to block push when there are no workers in the ctx */
	_starpu_pthread_cond_t no_workers_cond;

	/* mutex to block push when there are no workers in the ctx */
	_starpu_pthread_mutex_t no_workers_mutex;

	/*ready tasks that couldn't be pushed because the ctx has no workers*/
	struct starpu_task_list empty_ctx_tasks;

	/* mutext protecting empty_ctx_tasks list */
	_starpu_pthread_mutex_t empty_ctx_mutex;

	/* min CPUs to execute*/
	int min_ncpus;

	/* max CPUs to execute*/
	int max_ncpus;

	/* min GPUs to execute*/
	int min_ngpus;

	/* max GPUs to execute*/
	int max_ngpus;

	/* needed for overlapping contexts to help the workers
	   determine which is the next context to pop tasks from */
	unsigned pop_counter[STARPU_NMAXWORKERS];

	/* in case we delete the context leave resources to the inheritor*/
	unsigned inheritor;

	/* indicates whether the application finished submitting tasks
	   to this context*/
	unsigned finished_submit;

#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
	/* a structure containing a series of performance counters determining the resize procedure */
	struct starpu_performance_counters *perf_counters;
#endif //STARPU_USE_SCHED_CTX_HYPERVISOR
};

struct _starpu_machine_config;

/* init sched_ctx_id of all contextes*/
void _starpu_init_all_sched_ctxs(struct _starpu_machine_config *config);

/* init the list of contextes of the worker */
void _starpu_init_sched_ctx_for_worker(unsigned workerid);

/* allocate all structures belonging to a context */
struct _starpu_sched_ctx*  _starpu_create_sched_ctx(const char *policy_name, int *workerid, int nworkerids, unsigned is_init_sched, const char *sched_name);

/* delete all sched_ctx */
void _starpu_delete_all_sched_ctxs();

/* This function waits until all the tasks that were already submitted to a specific
 * context have been executed. */
int _starpu_wait_for_all_tasks_of_sched_ctx(unsigned sched_ctx_id);

/* In order to implement starpu_wait_for_all_tasks_of_ctx, we keep track of the number of
 * task currently submitted to the context */
void _starpu_decrement_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id);
void _starpu_increment_nsubmitted_tasks_of_sched_ctx(unsigned sched_ctx_id);

/* Return the corresponding index of the workerid in the ctx table */
int _starpu_get_index_in_ctx_of_workerid(unsigned sched_ctx, unsigned workerid);

/* Get the total number of sched_ctxs created till now */
unsigned _starpu_get_nsched_ctxs();

/* Get the mutex corresponding to the global workerid */
_starpu_pthread_mutex_t *_starpu_get_sched_mutex(struct _starpu_sched_ctx *sched_ctx, int worker);

/* Get workers belonging to a certain context, it returns the number of workers
 take care: no mutex taken, the list of workers might not be updated */
int starpu_get_workers_of_sched_ctx(unsigned sched_ctx_id, int *pus, enum starpu_archtype arch);

#if defined(_MSC_VER) || defined(STARPU_SIMGRID)
_starpu_pthread_mutex_t* starpu_get_changing_ctx_mutex(unsigned sched_ctx_id);
#endif

#endif // __SCHED_CONTEXT_H__
