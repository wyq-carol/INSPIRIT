/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
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
#include <starpu_scheduler.h>

struct starpu_sched_ctx {
	/* id of the context used in user mode*/
	unsigned sched_ctx_id;

	/* name of context */
	const char *sched_name;

	/* policy of the context */
	struct starpu_sched_policy_s *sched_policy;

	/* data necessary for the policy */
	void *policy_data;
	
	/* list of indices of workers */
	int workerid[STARPU_NMAXWORKERS]; 
	
	/* number of threads in contex */
	int nworkers_in_ctx; 

	/* we keep an initial sched which we never delete */
	unsigned is_initial_sched; 

	/* cond used for no of submitted tasks to a sched_ctx */
	pthread_cond_t submitted_cond; 
	
	/* mut used for no of submitted tasks to a sched_ctx */
	pthread_mutex_t submitted_mutex; 
	
	/* counter used for no of submitted tasks to a sched_ctx */
	int nsubmitted;	 

	/* table of sched cond corresponding to each worker in this ctx */
	pthread_cond_t **sched_cond;

	/* table of sched mutex corresponding to each worker in this ctx */
	pthread_mutex_t **sched_mutex;
};

unsigned _starpu_create_sched_ctx(const char *policy_name, int *workerid, int nworkerids, unsigned is_init_sched, const char *sched_name);

void _starpu_delete_all_sched_ctxs();

void _starpu_increment_nblocked_ths(int nworkers);
void _starpu_decrement_nblocked_ths(void);

/* Keeps track of the number of tasks currently submitted to a worker */
void _starpu_decrement_nsubmitted_tasks_of_worker(int workerid);
void _starpu_increment_nsubmitted_tasks_of_worker(int workerid);

/* In order to implement starpu_wait_for_all_tasks_of_ctx, we keep track of the number of 
 * task currently submitted to the context */
void _starpu_decrement_nsubmitted_tasks_of_sched_ctx(struct starpu_sched_ctx *sched_ctx);
void _starpu_increment_nsubmitted_tasks_of_sched_ctx(struct starpu_sched_ctx *sched_ctx);

/* Return the corresponding index of the workerid in the ctx table */
int _starpu_get_index_in_ctx_of_workerid(unsigned sched_ctx, unsigned workerid);

/* Get the mutex corresponding to the global workerid */
pthread_mutex_t *_starpu_get_sched_mutex(struct starpu_sched_ctx *sched_ctx, int worker);

/* Get the cond corresponding to the global workerid */
pthread_cond_t *_starpu_get_sched_cond(struct starpu_sched_ctx *sched_ctx, int worker);
#endif // __SCHED_CONTEXT_H__
