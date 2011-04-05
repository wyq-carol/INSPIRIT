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
#include <core/workers.h>
#include <starpu_scheduler.h>


void _starpu_create_sched_ctx(struct starpu_sched_ctx *sched_ctx, const char *policy_name, int *workerid, int nworkerids, unsigned is_init_sched, const char *sched_name);

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

#endif // __SCHED_CONTEXT_H__
