/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
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

/* Distributed queues using performance modeling to assign tasks */

#include <limits.h>

#include <core/workers.h>
#include <sched_policies/fifo_queues.h>
#include <core/perfmodel/perfmodel.h>
#include <starpu_parameters.h>

/* #ifdef STARPU_VERBOSE */
/* static long int total_task_cnt = 0; */
/* static long int ready_task_cnt = 0; */
/* #endif */

typedef struct {
	double alpha;
	double beta;
	double _gamma;
	double idle_power;

	struct starpu_fifo_taskq_s **queue_array;

	long int total_task_cnt;
	long int ready_task_cnt;
} dmda_data;


static int count_non_ready_buffers(struct starpu_task *task, uint32_t node)
{
	int cnt = 0;

	starpu_buffer_descr *descrs = task->buffers;
	unsigned nbuffers = task->cl->nbuffers;

	unsigned index;
	for (index = 0; index < nbuffers; index++)
	{
		starpu_buffer_descr *descr;
		starpu_data_handle handle;

		descr = &descrs[index];
		handle = descr->handle;
		
		int is_valid;
		starpu_data_query_status(handle, node, NULL, &is_valid, NULL);

		if (!is_valid)
			cnt++;
	}

	return cnt;
}

static struct starpu_task *_starpu_fifo_pop_first_ready_task(struct starpu_fifo_taskq_s *fifo_queue, unsigned node)
{
	struct starpu_task *task = NULL, *current;

	if (fifo_queue->ntasks == 0)
		return NULL;

	if (fifo_queue->ntasks > 0) 
	{
		fifo_queue->ntasks--;

		task = starpu_task_list_back(&fifo_queue->taskq);

		int first_task_priority = task->priority;

		current = task;

		int non_ready_best = INT_MAX;

		while (current)
		{
			int priority = current->priority;

			if (priority <= first_task_priority)
			{
				int non_ready = count_non_ready_buffers(current, node);
				if (non_ready < non_ready_best)
				{
					non_ready_best = non_ready;
					task = current;

					if (non_ready == 0)
						break;
				}
			}

			current = current->prev;
		}
		
		starpu_task_list_erase(&fifo_queue->taskq, task);

		STARPU_TRACE_JOB_POP(task, 0);
	}
	
	return task;
}

static struct starpu_task *dmda_pop_ready_task(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	dmda_data *dt = (dmda_data*)sched_ctx->policy_data;

	struct starpu_task *task;

	int workerid = starpu_worker_get_id();
	int workerid_ctx =  _starpu_get_index_in_ctx_of_workerid(sched_ctx_id, workerid);
	struct starpu_fifo_taskq_s *fifo = dt->queue_array[workerid_ctx];

	unsigned node = starpu_worker_get_memory_node(workerid);

	task = _starpu_fifo_pop_first_ready_task(fifo, node);
	if (task) {
		double model = task->predicted;
	
		fifo->exp_len -= model;
		fifo->exp_start = starpu_timing_now() + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;

#ifdef STARPU_VERBOSE
		if (task->cl)
		{
			int non_ready = count_non_ready_buffers(task, starpu_worker_get_memory_node(workerid));
			if (non_ready == 0)
				dt->ready_task_cnt++;
		}

		dt->total_task_cnt++;
#endif
	}

	return task;
}

static struct starpu_task *dmda_pop_task(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	dmda_data *dt = (dmda_data*)sched_ctx->policy_data;

	struct starpu_task *task;

	int workerid = starpu_worker_get_id();
	int workerid_ctx =  _starpu_get_index_in_ctx_of_workerid(sched_ctx_id, workerid);
	struct starpu_fifo_taskq_s *fifo = dt->queue_array[workerid_ctx];

	task = _starpu_fifo_pop_task(fifo, -1);
	if (task) {
		double model = task->predicted;
	
		fifo->exp_len -= model;
		fifo->exp_start = starpu_timing_now() + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;

#ifdef STARPU_VERBOSE
		if (task->cl)
		{
			int non_ready = count_non_ready_buffers(task, starpu_worker_get_memory_node(workerid));
			if (non_ready == 0)
				dt->ready_task_cnt++;
		}

		dt->total_task_cnt++;
#endif
	}

	return task;
}



static struct starpu_task *dmda_pop_every_task(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	dmda_data *dt = (dmda_data*)sched_ctx->policy_data;

	struct starpu_task *new_list;

	int workerid = starpu_worker_get_id();
	int workerid_ctx =  _starpu_get_index_in_ctx_of_workerid(sched_ctx_id, workerid);
	struct starpu_fifo_taskq_s *fifo = dt->queue_array[workerid_ctx];

	new_list = _starpu_fifo_pop_every_task(fifo, sched_ctx->sched_mutex[workerid_ctx], workerid);

	while (new_list)
	{
		double model = new_list->predicted;

		fifo->exp_len -= model;
		fifo->exp_start = starpu_timing_now() + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	
		new_list = new_list->next;
	}

	return new_list;
}

static
int _starpu_fifo_push_sorted_task(struct starpu_fifo_taskq_s *fifo_queue, pthread_mutex_t *sched_mutex, pthread_cond_t *sched_cond, struct starpu_task *task)
{
	struct starpu_task_list *list = &fifo_queue->taskq;

	PTHREAD_MUTEX_LOCK(sched_mutex);

	STARPU_TRACE_JOB_PUSH(task, 0);

	if (list->head == NULL)
	{
		list->head = task;
		list->tail = task;
		task->prev = NULL;
		task->next = NULL;
	}
	else {
		struct starpu_task *current = list->head;
		struct starpu_task *prev = NULL;

		while (current)
		{
			if (current->priority >= task->priority)
				break;

			prev = current;
			current = current->next;
		}

		if (prev == NULL)
		{
			/* Insert at the front of the list */
			list->head->prev = task;
			task->prev = NULL;
			task->next = list->head;
			list->head = task;
		}
		else {
			if (current)
			{
				/* Insert between prev and current */
				task->prev = prev;
				prev->next = task;
				task->next = current;
				current->prev = task;
			}
			else {
				/* Insert at the tail of the list */
				list->tail->next = task;
				task->next = NULL;
				task->prev = list->tail;
				list->tail = task;
			}
		}
	}

	fifo_queue->ntasks++;
	fifo_queue->nprocessed++;

	PTHREAD_COND_SIGNAL(sched_cond);
	PTHREAD_MUTEX_UNLOCK(sched_mutex);

	return 0;
}



static int push_task_on_best_worker(struct starpu_task *task, int best_workerid, double predicted, int prio, struct starpu_sched_ctx *sched_ctx)
{
	dmda_data *dt = (dmda_data*)sched_ctx->policy_data;
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	int best_workerid_ctx =  _starpu_get_index_in_ctx_of_workerid(sched_ctx->sched_ctx_id, best_workerid);

	struct starpu_fifo_taskq_s *fifo;
	fifo = dt->queue_array[best_workerid_ctx];

	fifo->exp_end += predicted;
	fifo->exp_len += predicted;

	task->predicted = predicted;

	unsigned memory_node = starpu_worker_get_memory_node(best_workerid);

	if (starpu_get_prefetch_flag())
		starpu_prefetch_task_input_on_node(task, memory_node);

	if (prio)
		return _starpu_fifo_push_sorted_task(dt->queue_array[best_workerid_ctx],
			sched_ctx->sched_mutex[best_workerid_ctx], sched_ctx->sched_cond[best_workerid_ctx], task);
	else
		return _starpu_fifo_push_task(dt->queue_array[best_workerid_ctx],
			sched_ctx->sched_mutex[best_workerid_ctx], sched_ctx->sched_cond[best_workerid_ctx], task);
}

static int _dm_push_task(struct starpu_task *task, unsigned prio, struct starpu_sched_ctx *sched_ctx)
{
	dmda_data *dt = (dmda_data*)sched_ctx->policy_data;
	/* find the queue */
	struct starpu_fifo_taskq_s *fifo;
	unsigned worker, worker_in_ctx;
	int best = -1;

	double best_exp_end = 0.0;
	double model_best = 0.0;

	int ntasks_best = -1;
	double ntasks_best_end = 0.0;
	int calibrating = 0;

	/* A priori, we know all estimations */
	int unknown = 0;

	unsigned best_impl = 0;
	unsigned nimpl;
	unsigned nworkers = sched_ctx->nworkers_in_ctx;
	for (worker_in_ctx = 0; worker_in_ctx < nworkers; worker_in_ctx++)
	{
		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
        	worker = sched_ctx->workerid[worker_in_ctx];
			double exp_end;
		
			fifo = dt->queue_array[worker_in_ctx];

			/* Sometimes workers didn't take the tasks as early as we expected */
			fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
			fifo->exp_end = fifo->exp_start + fifo->exp_len;

			if (!starpu_worker_may_execute_task(worker, task, nimpl))
			{
				/* no one on that queue may execute this task */
				continue;
			}

			enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
			double local_length = starpu_task_expected_length(task, perf_arch, nimpl);
			double ntasks_end = fifo->ntasks / starpu_worker_get_relative_speedup(perf_arch);

			//_STARPU_DEBUG("Scheduler dm: task length (%lf) worker (%u) kernel (%u) \n", local_length,worker,nimpl);

			if (ntasks_best == -1
					|| (!calibrating && ntasks_end < ntasks_best_end) /* Not calibrating, take better task */
					|| (!calibrating && local_length == -1.0) /* Not calibrating but this worker is being calibrated */
					|| (calibrating && local_length == -1.0 && ntasks_end < ntasks_best_end) /* Calibrating, compete this worker with other non-calibrated */
					) {
				ntasks_best_end = ntasks_end;
				ntasks_best = worker;
			}

			if (local_length == -1.0)
				/* we are calibrating, we want to speed-up calibration time
				 * so we privilege non-calibrated tasks (but still
				 * greedily distribute them to avoid dumb schedules) */
				calibrating = 1;

			if (local_length <= 0.0)
				/* there is no prediction available for that task
				 * with that arch yet, so switch to a greedy strategy */
				unknown = 1;

			if (unknown)
				continue;

			exp_end = fifo->exp_start + fifo->exp_len + local_length;

			if (best == -1 || exp_end < best_exp_end)
			{
				/* a better solution was found */
				best_exp_end = exp_end;
				best = worker;
				model_best = local_length;
				best_impl = nimpl;
			}
		}
	}

	if (unknown) {
		best = ntasks_best;
		model_best = 0.0;
	}
	
	_starpu_increment_nsubmitted_tasks_of_worker(best);

	//_STARPU_DEBUG("Scheduler dm: kernel (%u)\n", best_impl);

	 _starpu_get_job_associated_to_task(task)->nimpl = 0;//best_impl;

	/* we should now have the best worker in variable "best" */
	return push_task_on_best_worker(task, best, model_best, prio, sched_ctx);
}

static int _dmda_push_task(struct starpu_task *task, unsigned prio, struct starpu_sched_ctx *sched_ctx)
{
	dmda_data *dt = (dmda_data*)sched_ctx->policy_data;
	/* find the queue */
	struct starpu_fifo_taskq_s *fifo;
	unsigned worker, worker_in_ctx;
	int best = -1, best_in_ctx = -1;
	
	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_best = -1;

	unsigned nworkers_in_ctx = sched_ctx->nworkers_in_ctx;
	double local_task_length[nworkers_in_ctx];
	double local_data_penalty[nworkers_in_ctx];
	double local_power[nworkers_in_ctx];
	double exp_end[nworkers_in_ctx];
	double max_exp_end = 0.0;

	double fitness[nworkers_in_ctx];

	double best_exp_end = 10e240;
	double model_best = 0.0;
	//double penality_best = 0.0;

	int ntasks_best = -1;
	double ntasks_best_end = 0.0;
	int calibrating = 0;
	
	/* A priori, we know all estimations */
	int unknown = 0;

	unsigned best_impl = 0;
	unsigned nimpl=0;
	for (worker_in_ctx = 0; worker_in_ctx < nworkers_in_ctx; worker_in_ctx++)
	{
		worker = sched_ctx->workerid[worker_in_ctx];
		for(nimpl  = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
	 	{
			fifo = dt->queue_array[worker_in_ctx];

			/* Sometimes workers didn't take the tasks as early as we expected */
			fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
			fifo->exp_end = fifo->exp_start + fifo->exp_len;
			if (fifo->exp_end > max_exp_end)
				max_exp_end = fifo->exp_end;

			if (!starpu_worker_may_execute_task(worker, task, nimpl))
			{
				/* no one on that queue may execute this task */
				continue;
			}

			enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
			local_task_length[worker_in_ctx] = starpu_task_expected_length(task, perf_arch, nimpl);

			//_STARPU_DEBUG("Scheduler dmda: task length (%lf) worker (%u) kernel (%u) \n", local_task_length[worker],worker,nimpl);

			unsigned memory_node = starpu_worker_get_memory_node(worker);
			local_data_penalty[worker_in_ctx] = starpu_task_expected_data_transfer_time(memory_node, task);

			double ntasks_end = fifo->ntasks / starpu_worker_get_relative_speedup(perf_arch);

			if (ntasks_best == -1
					|| (!calibrating && ntasks_end < ntasks_best_end) /* Not calibrating, take better task */
					|| (!calibrating && local_task_length[worker] == -1.0) /* Not calibrating but this worker is being calibrated */
					|| (calibrating && local_task_length[worker] == -1.0 && ntasks_end < ntasks_best_end) /* Calibrating, compete this worker with other non-calibrated */
					) {
				ntasks_best_end = ntasks_end;
				ntasks_best = worker;

			}

			if (local_task_length[worker_in_ctx] == -1.0)
				/* we are calibrating, we want to speed-up calibration time
			 	* so we privilege non-calibrated tasks (but still
			 	* greedily distribute them to avoid dumb schedules) */
				calibrating = 1;

			if (local_task_length[worker_in_ctx] <= 0.0)
				/* there is no prediction available for that task
			 	* with that arch yet, so switch to a greedy strategy */
				unknown = 1;

			if (unknown)
					continue;

			exp_end[worker_in_ctx] = fifo->exp_start + fifo->exp_len + local_task_length[worker_in_ctx];

			if (exp_end[worker_in_ctx] < best_exp_end)
			{
				/* a better solution was found */
				best_exp_end = exp_end[worker_in_ctx];
				best_impl = nimpl;

			}

			local_power[worker_in_ctx] = starpu_task_expected_power(task, perf_arch, nimpl);
			if (local_power[worker_in_ctx] == -1.0)
				local_power[worker_in_ctx] = 0.;
			}	
		}

		if (unknown)
			forced_best = ntasks_best;

		double best_fitness = -1;
	
		if (forced_best == -1)
		{
	        for (worker_in_ctx = 0; worker_in_ctx < nworkers_in_ctx; worker_in_ctx++)
	        {
		        worker = sched_ctx->workerid[worker_in_ctx];

				fifo = dt->queue_array[worker_in_ctx];
	
			if (!starpu_worker_may_execute_task(worker, task, 0))
			{
				/* no one on that queue may execute this task */
				continue;
			}
	
			fitness[worker_in_ctx] = dt->alpha*(exp_end[worker_in_ctx] - best_exp_end) 
					+ dt->beta*(local_data_penalty[worker_in_ctx])
					+ dt->_gamma*(local_power[worker_in_ctx]);

			if (exp_end[worker_in_ctx] > max_exp_end)
				/* This placement will make the computation
				 * longer, take into account the idle
				 * consumption of other cpus */
				fitness[worker_in_ctx] += dt->_gamma * dt->idle_power * (exp_end[worker_in_ctx] - max_exp_end) / 1000000.0;

			if (best == -1 || fitness[worker_in_ctx] < best_fitness)
			{
				/* we found a better solution */
				best_fitness = fitness[worker_in_ctx];
				best = worker;
				best_in_ctx = worker_in_ctx;

	//			_STARPU_DEBUG("best fitness (worker %d) %e = alpha*(%e) + beta(%e) +gamma(%e)\n", worker, best_fitness, exp_end[worker] - best_exp_end, local_data_penalty[worker], local_power[worker]);
			}
		}
	}

	STARPU_ASSERT(forced_best != -1 || best != -1);
	
	if (forced_best != -1)
	{
		/* there is no prediction available for that task
		 * with that arch we want to speed-up calibration time
		 * so we force this measurement */
		best = forced_best;
		model_best = 0.0;
		//penality_best = 0.0;
	}
	else 
	{
		model_best = local_task_length[best];
		//penality_best = local_data_penalty[best];
	}


	//_STARPU_DEBUG("Scheduler dmda: kernel (%u)\n", best_impl);
	 _starpu_get_job_associated_to_task(task)->nimpl = best_impl;

	/* we should now have the best worker in variable "best" */
	return push_task_on_best_worker(task, best, model_best, prio, sched_ctx);
}

static int dmda_push_sorted_task(struct starpu_task *task, unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	return _dmda_push_task(task, 2, sched_ctx);
}

static int dm_push_task(struct starpu_task *task, unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	return _dm_push_task(task, 0, sched_ctx);
}

static int dmda_push_task(struct starpu_task *task, unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	return _dmda_push_task(task, 0, sched_ctx);
}

static void initialize_dmda_policy_for_workers(unsigned sched_ctx_id, unsigned nnew_workers) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	unsigned nworkers = sched_ctx->nworkers_in_ctx;
	dmda_data *dt = (dmda_data*)sched_ctx->policy_data;

	struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
	unsigned ntotal_workers = config->topology.nworkers;

	unsigned all_workers = nnew_workers == ntotal_workers ? ntotal_workers : nworkers + nnew_workers;

	unsigned workerid_ctx;
	for (workerid_ctx = nworkers; workerid_ctx < all_workers; workerid_ctx++)
	{
		dt->queue_array[workerid_ctx] = _starpu_create_fifo();
	
		sched_ctx->sched_mutex[workerid_ctx] = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		sched_ctx->sched_cond[workerid_ctx] = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
		PTHREAD_MUTEX_INIT(sched_ctx->sched_mutex[workerid_ctx], NULL);
		PTHREAD_COND_INIT(sched_ctx->sched_cond[workerid_ctx], NULL);
	}

	/* take into account the new number of threads at the next push */
	PTHREAD_MUTEX_LOCK(&sched_ctx->changing_ctx_mutex);
	sched_ctx->temp_nworkers_in_ctx = all_workers;
	PTHREAD_MUTEX_UNLOCK(&sched_ctx->changing_ctx_mutex);
}

static void initialize_dmda_policy(unsigned sched_ctx_id) 
{
	dmda_data *dt = (dmda_data*)malloc(sizeof(dmda_data));
	dt->alpha = STARPU_DEFAULT_ALPHA;
	dt->beta = STARPU_DEFAULT_BETA;
	dt->_gamma = STARPU_DEFAULT_GAMMA;
	dt->idle_power = 0.0;

	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	unsigned nworkers = sched_ctx->nworkers_in_ctx;
	sched_ctx->policy_data = (void*)dt;

	dt->queue_array = (struct starpu_fifo_taskq_s**)malloc(STARPU_NMAXWORKERS*sizeof(struct starpu_fifo_taskq_s*));

	const char *strval_alpha = getenv("STARPU_SCHED_ALPHA");
	if (strval_alpha)
		dt->alpha = atof(strval_alpha);

	const char *strval_beta = getenv("STARPU_SCHED_BETA");
	if (strval_beta)
		dt->beta = atof(strval_beta);

	const char *strval_gamma = getenv("STARPU_SCHED_GAMMA");
	if (strval_gamma)
		dt->_gamma = atof(strval_gamma);

	unsigned workerid_ctx;
	for (workerid_ctx = 0; workerid_ctx < nworkers; workerid_ctx++)
	{
		dt->queue_array[workerid_ctx] = _starpu_create_fifo();
	
		sched_ctx->sched_mutex[workerid_ctx] = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		sched_ctx->sched_cond[workerid_ctx] = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
		PTHREAD_MUTEX_INIT(sched_ctx->sched_mutex[workerid_ctx], NULL);
		PTHREAD_COND_INIT(sched_ctx->sched_cond[workerid_ctx], NULL);
	}
}

static void initialize_dmda_sorted_policy(unsigned sched_ctx_id)
{
	initialize_dmda_policy(sched_ctx_id);

	/* The application may use any integer */
	starpu_sched_set_min_priority(INT_MIN);
	starpu_sched_set_max_priority(INT_MAX);
}

static void deinitialize_dmda_policy(unsigned sched_ctx_id) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	dmda_data *dt = (dmda_data*)sched_ctx->policy_data;
	int workerid_in_ctx;
        int nworkers = sched_ctx->nworkers_in_ctx;
	for (workerid_in_ctx = 0; workerid_in_ctx < nworkers; workerid_in_ctx++){
		_starpu_destroy_fifo(dt->queue_array[workerid_in_ctx]);
		PTHREAD_MUTEX_DESTROY(sched_ctx->sched_mutex[workerid_in_ctx]);
                PTHREAD_COND_DESTROY(sched_ctx->sched_cond[workerid_in_ctx]);
		free(sched_ctx->sched_mutex[workerid_in_ctx]);
                free(sched_ctx->sched_cond[workerid_in_ctx]);
	}

	free(dt->queue_array);
	free(dt);

	_STARPU_DEBUG("total_task_cnt %ld ready_task_cnt %ld -> %f\n", total_task_cnt, ready_task_cnt, (100.0f*ready_task_cnt)/total_task_cnt);
}

/* TODO: use post_exec_hook to fix the expected start */
struct starpu_sched_policy_s _starpu_sched_dm_policy = {
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.push_task = dm_push_task, 
	.pop_task = dmda_pop_task,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dm",
	.policy_description = "performance model",
	.init_sched_for_workers = initialize_dmda_policy_for_workers
};

struct starpu_sched_policy_s _starpu_sched_dmda_policy = {
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.push_task = dmda_push_task, 
	.pop_task = dmda_pop_task,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dmda",
	.policy_description = "data-aware performance model",
	.init_sched_for_workers = initialize_dmda_policy_for_workers
};

struct starpu_sched_policy_s _starpu_sched_dmda_sorted_policy = {
	.init_sched = initialize_dmda_sorted_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.push_task = dmda_push_sorted_task, 
	.pop_task = dmda_pop_ready_task,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dmdas",
	.policy_description = "data-aware performance model (sorted)",
	.init_sched_for_workers = initialize_dmda_policy_for_workers
};

struct starpu_sched_policy_s _starpu_sched_dmda_ready_policy = {
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.push_task = dmda_push_task, 
	.pop_task = dmda_pop_ready_task,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dmdar",
	.policy_description = "data-aware performance model (ready)",
	.init_sched_for_workers = initialize_dmda_policy_for_workers
};
