/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <pthread.h>
#include <core/mechanisms/fifo_queues.h>
#include <errno.h>

/* keep track of the total number of jobs to be scheduled to avoid infinite 
 * polling when there are really few jobs in the overall queue */
static unsigned total_number_of_jobs;

static pthread_cond_t *sched_cond;
static pthread_mutex_t *sched_mutex;

void _starpu_init_fifo_queues_mechanisms(void)
{
	total_number_of_jobs = 0;

	struct starpu_sched_policy_s *sched = _starpu_get_sched_policy();

	/* to access them more easily, we keep their address in local variables */
	sched_cond = &sched->sched_activity_cond;
	sched_mutex = &sched->sched_activity_mutex;
}

struct jobq_s *_starpu_create_fifo(void)
{
	struct jobq_s *jobq;
	jobq = malloc(sizeof(struct jobq_s));

	pthread_mutex_init(&jobq->activity_mutex, NULL);
	pthread_cond_init(&jobq->activity_cond, NULL);

	struct starpu_fifo_jobq_s *fifo;
	fifo = malloc(sizeof(struct starpu_fifo_jobq_s));

	/* note that not all mechanisms (eg. the semaphore) have to be used */
	fifo->jobq = starpu_job_list_new();
	fifo->njobs = 0;
	fifo->nprocessed = 0;

	fifo->exp_start = timing_now();
	fifo->exp_len = 0.0;
	fifo->exp_end = fifo->exp_start;

	jobq->queue = fifo;

	return jobq;
}

int _starpu_fifo_push_prio_task(struct jobq_s *q, starpu_job_t task)
{
#ifndef STARPU_NO_PRIO
	STARPU_ASSERT(q);
	struct starpu_fifo_jobq_s *fifo_queue = q->queue;

	/* if anyone is blocked on the entire machine, wake it up */
	pthread_mutex_lock(sched_mutex);
	total_number_of_jobs++;
	pthread_cond_signal(sched_cond);
	pthread_mutex_unlock(sched_mutex);
	
	/* wake people waiting locally */
	pthread_mutex_lock(&q->activity_mutex);

	TRACE_JOB_PUSH(task, 0);
	starpu_job_list_push_back(fifo_queue->jobq, task);
	fifo_queue->njobs++;
	fifo_queue->nprocessed++;

	pthread_cond_signal(&q->activity_cond);
	pthread_mutex_unlock(&q->activity_mutex);

	return 0;
#else
	return _starpu_fifo_push_task(q, task);
#endif
}

int _starpu_fifo_push_task(struct jobq_s *q, starpu_job_t task)
{
	STARPU_ASSERT(q);
	struct starpu_fifo_jobq_s *fifo_queue = q->queue;

	/* if anyone is blocked on the entire machine, wake it up */
	pthread_mutex_lock(sched_mutex);
	total_number_of_jobs++;
	pthread_cond_signal(sched_cond);
	pthread_mutex_unlock(sched_mutex);
	
	/* wake people waiting locally */
	pthread_mutex_lock(&q->activity_mutex);

	TRACE_JOB_PUSH(task, 0);
	starpu_job_list_push_front(fifo_queue->jobq, task);
	fifo_queue->njobs++;
	fifo_queue->nprocessed++;

	pthread_cond_signal(&q->activity_cond);
	pthread_mutex_unlock(&q->activity_mutex);

	return 0;
}

starpu_job_t _starpu_fifo_pop_task(struct jobq_s *q)
{
	starpu_job_t j = NULL;

	STARPU_ASSERT(q);
	struct starpu_fifo_jobq_s *fifo_queue = q->queue;

	if (fifo_queue->njobs == 0)
		return NULL;

	if (fifo_queue->njobs > 0) 
	{
		/* there is a task */
		j = starpu_job_list_pop_back(fifo_queue->jobq);
	
		STARPU_ASSERT(j);
		fifo_queue->njobs--;
		
		TRACE_JOB_POP(j, 0);

		/* we are sure that we got it now, so at worst, some people thought 
		 * there remained some work and will soon discover it is not true */
		pthread_mutex_lock(sched_mutex);
		total_number_of_jobs--;
		pthread_mutex_unlock(sched_mutex);
	}
	
	return j;
}

/* pop every task that can be executed on the calling driver */
struct starpu_job_list_s * _starpu_fifo_pop_every_task(struct jobq_s *q, uint32_t where)
{
	struct starpu_job_list_s *new_list, *old_list;
	unsigned size;
	
	STARPU_ASSERT(q);
	struct starpu_fifo_jobq_s *fifo_queue = q->queue;

	pthread_mutex_lock(&q->activity_mutex);

	size = fifo_queue->njobs;

	if (size == 0) {
		new_list = NULL;
	}
	else {
		old_list = fifo_queue->jobq;
		new_list = starpu_job_list_new();

		unsigned new_list_size = 0;

		starpu_job_itor_t i;
		starpu_job_t next_job;
		/* note that this starts at the _head_ of the list, so we put
 		 * elements at the back of the new list */
		for(i = starpu_job_list_begin(old_list);
			i != starpu_job_list_end(old_list);
			i  = next_job)
		{
			next_job = starpu_job_list_next(i);

			if (i->task->cl->where & where)
			{
				/* this elements can be moved into the new list */
				new_list_size++;
				
				starpu_job_list_erase(old_list, i);
				starpu_job_list_push_back(new_list, i);
			}
		}

		if (new_list_size == 0)
		{
			/* the new list is empty ... */
			starpu_job_list_delete(new_list);
			new_list = NULL;
		}
		else
		{
			fifo_queue->njobs -= new_list_size;
	
			/* we are sure that we got it now, so at worst, some people thought
			 * there remained some work and will soon discover it is not true */
			pthread_mutex_lock(sched_mutex);
			total_number_of_jobs -= new_list_size;
			pthread_mutex_unlock(sched_mutex);
		}
	}

	pthread_mutex_unlock(&q->activity_mutex);

	return new_list;
}
