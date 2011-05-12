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

/* Work stealing policy */

#include <core/workers.h>
#include <sched_policies/deque_queues.h>

//static unsigned nworkers;
//static unsigned rr_worker;
//static struct starpu_deque_jobq_s *queue_array[STARPU_NMAXWORKERS];

/* static pthread_mutex_t global_sched_mutex; */
/* static pthread_cond_t global_sched_cond; */

/* keep track of the work performed from the beginning of the algorithm to make
 * better decisions about which queue to select when stealing or deferring work
 */
//static unsigned performed_total = 0;

typedef struct {
	struct starpu_deque_jobq_s **queue_array;
	unsigned rr_worker;
	unsigned performed_total;
} work_stealing_data;

#ifdef USE_OVERLOAD
static float overload_metric(struct starpu_deque_jobq_s *dequeue_queue, unsigned *performed_total)
{
	float execution_ratio = 0.0f;
	if (*performed_total > 0) {
		execution_ratio = _starpu_get_deque_nprocessed(dequeue_queue)/ *performed_total;
	}

	unsigned performed_queue;
	performed_queue = _starpu_get_deque_nprocessed(dequeue_queue);

	float current_ratio = 0.0f;
	if (performed_queue > 0) {
		current_ratio = _starpu_get_deque_njobs(dequeue_queue)/performed_queue;
	}
	
	return (current_ratio - execution_ratio);
}

/* who to steal work to ? */
static struct starpu_deque_jobq_s *select_victimq(work_stealing_data *ws, unsigned nworkers)
{
	struct starpu_deque_jobq_s *q;

	unsigned attempts = nworkers;

	unsigned worker = ws->rr_worker;
	do {
		if (overload_metric(worker) > 0.0f)
		{
			q = ws->queue_array[worker];
			return q;
		}
		else {
			worker = (worker + 1)%nworkers;
		}
	} while(attempts-- > 0);

	/* take one anyway ... */
	q = ws->queue_array[ws->rr_worker];
	ws->rr_worker = (ws->rr_worker + 1 )%nworkers;

	return q;
}

static struct starpu_deque_jobq_s *select_workerq(work_stealing_data *ws, unsigned nworkers)
{
	struct starpu_deque_jobq_s *q;

	unsigned attempts = nworkers;

	unsigned worker = ws->rr_worker;
	do {
		if (overload_metric(worker) < 0.0f)
		{
			q = ws->queue_array[worker];
			return q;
		}
		else {
			worker = (worker + 1)%nworkers;
		}
	} while(attempts-- > 0);

	/* take one anyway ... */
	q = ws->queue_array[ws->rr_worker];
	ws->rr_worker = (ws->rr_worker + 1 )%nworkers;

	return q;
}

#else

/* who to steal work to ? */
static struct starpu_deque_jobq_s *select_victimq(work_stealing_data *ws, unsigned nworkers)
{
	struct starpu_deque_jobq_s *q;

	q = ws->queue_array[ws->rr_worker];

	ws->rr_worker = (ws->rr_worker + 1 )%nworkers;

	return q;
}


/* when anonymous threads submit tasks, 
 * we need to select a queue where to dispose them */
static struct starpu_deque_jobq_s *select_workerq(work_stealing_data *ws, unsigned nworkers)
{
	struct starpu_deque_jobq_s *q;

	q = ws->queue_array[ws->rr_worker];

	ws->rr_worker = (ws->rr_worker + 1 )%nworkers;

	return q;
}

#endif

#warning TODO rewrite ... this will not scale at all now
static struct starpu_task *ws_pop_task(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	work_stealing_data *ws = (work_stealing_data*)sched_ctx->policy_data;

	struct starpu_task *task;

	int workerid = starpu_worker_get_id();
	int workerid_ctx =  _starpu_get_index_in_ctx_of_workerid(sched_ctx_id, workerid);

	struct starpu_deque_jobq_s *q;

	q = ws->queue_array[workerid_ctx];

	PTHREAD_MUTEX_LOCK(sched_ctx->sched_mutex[0]);

	task = _starpu_deque_pop_task(q, -1);
	if (task) {
		/* there was a local task */
		ws->performed_total++;
		PTHREAD_MUTEX_UNLOCK(sched_ctx->sched_mutex[0]);
		return task;
	}
	
	/* we need to steal someone's job */
	struct starpu_deque_jobq_s *victimq;
	victimq = select_victimq(ws, sched_ctx->nworkers_in_ctx);

	task = _starpu_deque_pop_task(victimq, workerid);
	if (task) {
		STARPU_TRACE_WORK_STEALING(q, victimq);
		ws->performed_total++;
	}

	PTHREAD_MUTEX_UNLOCK(sched_ctx->sched_mutex[0]);

	return task;
}

int ws_push_task(struct starpu_task *task, unsigned sched_ctx_id)
{
	starpu_job_t j = _starpu_get_job_associated_to_task(task);

	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	work_stealing_data *ws = (work_stealing_data*)sched_ctx->policy_data;

	int workerid = starpu_worker_get_id();
	int workerid_ctx =  _starpu_get_index_in_ctx_of_workerid(sched_ctx_id, workerid);

        struct starpu_deque_jobq_s *deque_queue;
	deque_queue = ws->queue_array[workerid_ctx];

        PTHREAD_MUTEX_LOCK(sched_ctx->sched_mutex[0]);
	// XXX reuse ?
        //total_number_of_jobs++;

        STARPU_TRACE_JOB_PUSH(task, 0);
        starpu_job_list_push_front(deque_queue->jobq, j);
        deque_queue->njobs++;
        deque_queue->nprocessed++;

        PTHREAD_COND_SIGNAL(sched_ctx->sched_cond[0]);
        PTHREAD_MUTEX_UNLOCK(sched_ctx->sched_mutex[0]);

        return 0;
}

static void initialize_ws_policy(unsigned sched_ctx_id) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	work_stealing_data *ws = (work_stealing_data*)malloc(sizeof(work_stealing_data));
	sched_ctx->policy_data = (void*)ws;
	
	unsigned nworkers = sched_ctx->nworkers_in_ctx;
	ws->rr_worker = 0;
	ws->queue_array = (struct starpu_deque_jobq_s**)malloc(nworkers*sizeof(struct starpu_deque_jobq_s*));


	pthread_mutex_t *sched_mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_cond_t *sched_cond = (pthread_cond_t*) malloc(sizeof(pthread_cond_t));
	PTHREAD_MUTEX_INIT(sched_mutex, NULL);
	PTHREAD_COND_INIT(sched_cond, NULL);

	unsigned workerid_ctx;
	for (workerid_ctx = 0; workerid_ctx < nworkers; workerid_ctx++)
	{
		ws->queue_array[workerid_ctx] = _starpu_create_deque();

		sched_ctx->sched_mutex[workerid_ctx] = sched_mutex;
		sched_ctx->sched_cond[workerid_ctx] = sched_cond;
	}
}

struct starpu_sched_policy_s _starpu_sched_ws_policy = {
	.init_sched = initialize_ws_policy,
	.deinit_sched = NULL,
	.push_task = ws_push_task,
	.push_prio_task = ws_push_task,
	.pop_task = ws_pop_task,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "ws",
	.policy_description = "work stealing"
};
