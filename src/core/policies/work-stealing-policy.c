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

#include <core/policies/work-stealing-policy.h>

/* save the general machine configuration */
//static struct machine_config_s *machineconfig;

static unsigned nworkers;
static unsigned rr_worker;
static struct jobq_s *queue_array[STARPU_NMAXWORKERS];

/* keep track of the work performed from the beginning of the algorithm to make
 * better decisions about which queue to select when stealing or deferring work
 */
static unsigned performed_total = 0;
//static unsigned performed_local[16];

#ifdef USE_OVERLOAD
static float overload_metric(unsigned id)
{
	float execution_ratio = 0.0f;
	if (performed_total > 0) {
		execution_ratio = get_deque_nprocessed(queue_array[id])/performed_total;
	}

	unsigned performed_queue;
	performed_queue = get_deque_nprocessed(queue_array[id]);

	float current_ratio = 0.0f;
	if (performed_queue > 0) {
		current_ratio = get_deque_njobs(queue_array[id])/performed_queue;
	}
	
	return (current_ratio - execution_ratio);
}

/* who to steal work to ? */
static struct jobq_s *select_victimq(void)
{
	struct jobq_s *q;

	unsigned attempts = nworkers;

	unsigned worker = rr_worker;
	do {
		if (overload_metric(worker) > 0.0f)
		{
			q = queue_array[worker];
			return q;
		}
		else {
			worker = (worker + 1)%nworkers;
		}
	} while(attempts-- > 0);

	/* take one anyway ... */
	q = queue_array[rr_worker];
	rr_worker = (rr_worker + 1 )%nworkers;

	return q;
}

static struct jobq_s *select_workerq(void)
{
	struct jobq_s *q;

	unsigned attempts = nworkers;

	unsigned worker = rr_worker;
	do {
		if (overload_metric(worker) < 0.0f)
		{
			q = queue_array[worker];
			return q;
		}
		else {
			worker = (worker + 1)%nworkers;
		}
	} while(attempts-- > 0);

	/* take one anyway ... */
	q = queue_array[rr_worker];
	rr_worker = (rr_worker + 1 )%nworkers;

	return q;
}

#else

/* who to steal work to ? */
static struct jobq_s *select_victimq(void)
{

	struct jobq_s *q;

	q = queue_array[rr_worker];

	rr_worker = (rr_worker + 1 )%nworkers;

	return q;
}


/* when anonymous threads submit tasks, 
 * we need to select a queue where to dispose them */
static struct jobq_s *select_workerq(void)
{

	struct jobq_s *q;

	q = queue_array[rr_worker];

	rr_worker = (rr_worker + 1 )%nworkers;

	return q;
}

#endif

static job_t ws_pop_task(struct jobq_s *q)
{
	job_t j;

	j = deque_pop_task(q);
	if (j) {
		/* there was a local task */
		performed_total++;
		return j;
	}
	
	/* we need to steal someone's job */
	struct jobq_s *victimq;
	victimq = select_victimq();

	if (!jobq_trylock(victimq))
	{
		j = deque_pop_task(victimq);
		jobq_unlock(victimq);

		TRACE_WORK_STEALING(q, j);
		performed_total++;
	}

	return j;
}

static struct jobq_s *init_ws_deque(void)
{
	struct jobq_s *q;

	q = create_deque();

	q->push_task = deque_push_task; 
	q->push_prio_task = deque_push_prio_task; 
	q->pop_task = ws_pop_task;
	q->who = 0;

	queue_array[nworkers++] = q;

	return q;
}

static void initialize_ws_policy(struct machine_config_s *config, 
				__attribute__ ((unused)) struct sched_policy_s *_policy) 
{
	nworkers = 0;
	rr_worker = 0;

	//machineconfig = config;

	setup_queues(init_deque_queues_mechanisms, init_ws_deque, config);
}

static struct jobq_s *get_local_queue_ws(struct sched_policy_s *policy __attribute__ ((unused)))
{
	struct jobq_s *queue;
	queue = pthread_getspecific(policy->local_queue_key);

	if (!queue) {
		queue = select_workerq();
	}

	STARPU_ASSERT(queue);

	return queue;
}

struct sched_policy_s sched_ws_policy = {
	.init_sched = initialize_ws_policy,
	.deinit_sched = NULL,
	.get_local_queue = get_local_queue_ws,
	.policy_name = "ws",
	.policy_description = "work stealing"
};
