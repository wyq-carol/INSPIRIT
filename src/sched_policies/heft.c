/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
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

/* Distributed queues using performance modeling to assign tasks */

#include <float.h>

#include <core/workers.h>
#include <core/sched_ctx.h>
#include <core/perfmodel/perfmodel.h>
#include <starpu_parameters.h>
#include <starpu_task_bundle.h>

typedef struct {
	double alpha;
	double beta;
	double _gamma;
	double idle_power;

	double *exp_start;
	double *exp_end;
	double *exp_len;
	double *ntasks;
} heft_data;

static void heft_init(unsigned sched_ctx_id)
{
	heft_data *hd = (heft_data*)malloc(sizeof(heft_data));
	hd->alpha = STARPU_DEFAULT_ALPHA;
	hd->beta = STARPU_DEFAULT_BETA;
	hd->_gamma = STARPU_DEFAULT_GAMMA;
	hd->idle_power = 0.0;
	
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);

	unsigned nworkers = sched_ctx->nworkers_in_ctx;
	sched_ctx->policy_data = (void*)hd;

	hd->exp_start = (double*)malloc(nworkers*sizeof(double));
	hd->exp_end = (double*)malloc(nworkers*sizeof(double));
	hd->exp_len = (double*)malloc(nworkers*sizeof(double));
	hd->ntasks = (double*)malloc(nworkers*sizeof(double));

	const char *strval_alpha = getenv("STARPU_SCHED_ALPHA");
	if (strval_alpha)
		hd->alpha = atof(strval_alpha);

	const char *strval_beta = getenv("STARPU_SCHED_BETA");
	if (strval_beta)
		hd->beta = atof(strval_beta);

	const char *strval_gamma = getenv("STARPU_SCHED_GAMMA");
	if (strval_gamma)
		hd->_gamma = atof(strval_gamma);

	const char *strval_idle_power = getenv("STARPU_IDLE_POWER");
	if (strval_idle_power)
		hd->idle_power = atof(strval_idle_power);

	unsigned workerid_ctx;

	for (workerid_ctx = 0; workerid_ctx < nworkers; workerid_ctx++)
	  {
	    hd->exp_start[workerid_ctx] = starpu_timing_now();
	    hd->exp_len[workerid_ctx] = 0.0;
	    hd->exp_end[workerid_ctx] = hd->exp_start[workerid_ctx]; 
	    hd->ntasks[workerid_ctx] = 0;
	    	    
	    sched_ctx->sched_mutex[workerid_ctx] = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
	    sched_ctx->sched_cond[workerid_ctx] = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
	    PTHREAD_MUTEX_INIT(sched_ctx->sched_mutex[workerid_ctx], NULL);
	    PTHREAD_COND_INIT(sched_ctx->sched_cond[workerid_ctx], NULL);
	  }
}

static void heft_post_exec_hook(struct starpu_task *task, unsigned sched_ctx_id)
{
	int workerid = starpu_worker_get_id();
	int workerid_ctx =  _starpu_get_index_in_ctx_of_workerid(sched_ctx_id, workerid);

	double model = task->predicted;
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	heft_data *hd = (heft_data*)sched_ctx->policy_data;

	/* Once we have executed the task, we can update the predicted amount
	 * of work. */
	PTHREAD_MUTEX_LOCK(sched_ctx->sched_mutex[workerid_ctx]);
	hd->exp_len[workerid_ctx] -= model;
	hd->exp_start[workerid_ctx] = starpu_timing_now() + model;
	hd->exp_end[workerid_ctx] = hd->exp_start[workerid_ctx] + hd->exp_len[workerid_ctx];
	hd->ntasks[workerid_ctx]--;
	PTHREAD_MUTEX_UNLOCK(sched_ctx->sched_mutex[workerid_ctx]);
}

static void heft_push_task_notify(struct starpu_task *task, int workerid, unsigned sched_ctx_id)
{
	int workerid_ctx =  _starpu_get_index_in_ctx_of_workerid(sched_ctx_id, workerid);
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	heft_data *hd = (heft_data*)sched_ctx->policy_data;

	/* Compute the expected penality */
	enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(workerid);
	double predicted = starpu_task_expected_length(task, perf_arch);

	/* Update the predictions */
	PTHREAD_MUTEX_LOCK(sched_ctx->sched_mutex[workerid_ctx]);

	/* Sometimes workers didn't take the tasks as early as we expected */
	hd->exp_start[workerid_ctx] = STARPU_MAX(hd->exp_start[workerid_ctx], starpu_timing_now());
	hd->exp_end[workerid_ctx] = STARPU_MAX(hd->exp_start[workerid_ctx], starpu_timing_now());

	/* If there is no prediction available, we consider the task has a null length */
	if (predicted != -1.0)
	{
		task->predicted = predicted;
		hd->exp_end[workerid_ctx] += predicted;
		hd->exp_len[workerid_ctx] += predicted;
	}

	hd->ntasks[workerid_ctx]++;

	PTHREAD_MUTEX_UNLOCK(sched_ctx->sched_mutex[workerid_ctx]);
}

static int push_task_on_best_worker(struct starpu_task *task, int best_workerid, double predicted, int prio, struct starpu_sched_ctx *sched_ctx)
{
	heft_data *hd = (heft_data*)sched_ctx->policy_data;

	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);
	int best_workerid_ctx =  _starpu_get_index_in_ctx_of_workerid(sched_ctx->sched_ctx_id, best_workerid);

	PTHREAD_MUTEX_LOCK(sched_ctx->sched_mutex[best_workerid_ctx]);
	hd->exp_end[best_workerid_ctx] += predicted;
	hd->exp_len[best_workerid_ctx] += predicted;
	hd->ntasks[best_workerid_ctx]++;
	PTHREAD_MUTEX_UNLOCK(sched_ctx->sched_mutex[best_workerid_ctx]);

	task->predicted = predicted;

	if (starpu_get_prefetch_flag())
	{
		unsigned memory_node = starpu_worker_get_memory_node(best_workerid);
		starpu_prefetch_task_input_on_node(task, memory_node);
	}
	return starpu_push_local_task(best_workerid, task, prio);
}

static void compute_all_performance_predictions(struct starpu_task *task,
						double *local_task_length, double *exp_end,
						double *max_exp_endp, double *best_exp_endp,
						double *local_data_penalty,
						double *local_power, int *forced_best,
						struct starpu_task_bundle *bundle,
						 struct starpu_sched_ctx *sched_ctx )
{
  int calibrating = 0;
  double max_exp_end = DBL_MIN;
  double best_exp_end = DBL_MAX;
  int ntasks_best = -1;
  double ntasks_best_end = 0.0;

  /* A priori, we know all estimations */
  int unknown = 0;

  unsigned nworkers = sched_ctx->nworkers_in_ctx;
  heft_data *hd = (heft_data*)sched_ctx->policy_data;

  unsigned worker, worker_in_ctx;
  for (worker_in_ctx = 0; worker_in_ctx < nworkers; worker_in_ctx++)
    {
      worker = sched_ctx->workerid[worker_in_ctx];
      /* Sometimes workers didn't take the tasks as early as we expected */
      hd->exp_start[worker_in_ctx] = STARPU_MAX(hd->exp_start[worker_in_ctx], starpu_timing_now());
      exp_end[worker_in_ctx] = hd->exp_start[worker_in_ctx] + hd->exp_len[worker_in_ctx];
      if (exp_end[worker_in_ctx] > max_exp_end)
 	max_exp_end = exp_end[worker_in_ctx];

      if (!starpu_worker_may_execute_task(worker, task))
	{
	  /* no one on that queue may execute this task */
	  continue;
	}

      enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
      unsigned memory_node = starpu_worker_get_memory_node(worker);

      if (bundle)
      	{
      	  local_task_length[worker_in_ctx] = starpu_task_bundle_expected_length(bundle, perf_arch);
      	  local_data_penalty[worker_in_ctx] = starpu_task_bundle_expected_data_transfer_time(bundle, memory_node);
      	  local_power[worker_in_ctx] = starpu_task_bundle_expected_power(bundle, perf_arch);
      	}
      else {
	local_task_length[worker_in_ctx] = starpu_task_expected_length(task, perf_arch);
	local_data_penalty[worker_in_ctx] = starpu_task_expected_data_transfer_time(memory_node, task);
	local_power[worker_in_ctx] = starpu_task_expected_power(task, perf_arch);
      }

      //      printf("%d: local task len = %2.2f perf model %d\n", worker, local_task_length[worker_in_ctx], task->cl->model->type);

      double ntasks_end = hd->ntasks[worker_in_ctx] / starpu_worker_get_relative_speedup(perf_arch);

      if (ntasks_best == -1
	  || (!calibrating && ntasks_end < ntasks_best_end) /* Not calibrating, take better task */
	  || (!calibrating && local_task_length[worker_in_ctx] == -1.0) /* Not calibrating but this worker is being calibrated */
	  || (calibrating && local_task_length[worker_in_ctx] == -1.0 && ntasks_end < ntasks_best_end) /* Calibrating, compete this worker with other non-calibrated */
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

      exp_end[worker_in_ctx] = hd->exp_start[worker_in_ctx] + hd->exp_len[worker_in_ctx] + local_task_length[worker_in_ctx];

      if (exp_end[worker_in_ctx] < best_exp_end)
	{
	  /* a better solution was found */
	  best_exp_end = exp_end[worker_in_ctx];
	}

      if (local_power[worker_in_ctx] == -1.0)
	local_power[worker_in_ctx] = 0.;
    }

  *forced_best = unknown?ntasks_best:-1;
  *best_exp_endp = best_exp_end;
  *max_exp_endp = max_exp_end;
}

static int _heft_push_task(struct starpu_task *task, unsigned prio, unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	heft_data *hd = (heft_data*)sched_ctx->policy_data;
	unsigned worker, worker_in_ctx;
	int best = -1, best_id_in_ctx = -1;
	
	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_best;

	unsigned nworkers_in_ctx = sched_ctx->nworkers_in_ctx;
	double local_task_length[nworkers_in_ctx];
	double local_data_penalty[nworkers_in_ctx];
	double local_power[nworkers_in_ctx];
	double exp_end[nworkers_in_ctx];
	double max_exp_end = 0.0;

	double best_exp_end;

	/*
	 *	Compute the expected end of the task on the various workers,
	 *	and detect if there is some calibration that needs to be done.
	 */

	struct starpu_task_bundle *bundle = task->bundle;

	compute_all_performance_predictions(task, local_task_length, exp_end,
					    &max_exp_end, &best_exp_end,
					    local_data_penalty,
					    local_power, &forced_best, bundle, sched_ctx);

	/* If there is no prediction available for that task with that arch we
	 * want to speed-up calibration time so we force this measurement */
	if (forced_best != -1){
		_starpu_increment_nsubmitted_tasks_of_worker(forced_best);
		return push_task_on_best_worker(task, forced_best, 0.0, prio, sched_ctx);
	}

	/*
	 *	Determine which worker optimizes the fitness metric which is a
	 *	trade-off between load-balacing, data locality, and energy
	 *	consumption.
	 */
	
	double fitness[nworkers_in_ctx];
	double best_fitness = -1;

	for (worker_in_ctx = 0; worker_in_ctx < nworkers_in_ctx; worker_in_ctx++)
	{
		worker = sched_ctx->workerid[worker_in_ctx];
		if (!starpu_worker_may_execute_task(worker, task))
		{
			/* no one on that queue may execute this task */
			continue;
		}

		fitness[worker_in_ctx] = hd->alpha*(exp_end[worker_in_ctx] - best_exp_end) 
				+ hd->beta*(local_data_penalty[worker_in_ctx])
				+ hd->_gamma*(local_power[worker_in_ctx]);

		if (exp_end[worker_in_ctx] > max_exp_end)
			/* This placement will make the computation
			 * longer, take into account the idle
			 * consumption of other cpus */
			fitness[worker_in_ctx] += hd->_gamma * hd->idle_power * (exp_end[worker_in_ctx] - max_exp_end) / 1000000.0;

		if (best == -1 || fitness[worker_in_ctx] < best_fitness)
		{
			/* we found a better solution */
			best_fitness = fitness[worker_in_ctx];
			best = worker;
			best_id_in_ctx = worker_in_ctx;
		}
	}

	/* By now, we must have found a solution */
	STARPU_ASSERT(best != -1);
	
	/* we should now have the best worker in variable "best" */
	double model_best;

	if (bundle)
	{
		/* If we have a task bundle, we have computed the expected
		 * length for the entire bundle, but not for the task alone. */
		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(best);
		model_best = starpu_task_expected_length(task, perf_arch);

		/* Remove the task from the bundle since we have made a
		 * decision for it, and that other tasks should not consider it
		 * anymore. */
		PTHREAD_MUTEX_LOCK(&bundle->mutex);
		int ret = starpu_task_bundle_remove(bundle, task);

		/* Perhaps the bundle was destroyed when removing the last
		 * entry */
		if (ret != 1)
			PTHREAD_MUTEX_UNLOCK(&bundle->mutex);

	}
	else {
		model_best = local_task_length[best_id_in_ctx];
	}

	_starpu_increment_nsubmitted_tasks_of_worker(best);
	return push_task_on_best_worker(task, best, model_best, prio, sched_ctx);
}

static int heft_push_prio_task(struct starpu_task *task, unsigned sched_ctx_id)
{
        return _heft_push_task(task, 1, sched_ctx_id);
}

static int heft_push_task(struct starpu_task *task, unsigned sched_ctx_id)
{
	if (task->priority > 0)
        	  return _heft_push_task(task, 1, sched_ctx_id);

	return _heft_push_task(task, 0, sched_ctx_id);
}

static void heft_deinit(unsigned sched_ctx_id) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	int workerid_in_ctx;
	int nworkers = sched_ctx->nworkers_in_ctx;
	heft_data *ht = (heft_data*)sched_ctx->policy_data;

	for (workerid_in_ctx = 0; workerid_in_ctx < nworkers; workerid_in_ctx++){
		PTHREAD_MUTEX_DESTROY(sched_ctx->sched_mutex[workerid_in_ctx]);
		PTHREAD_COND_DESTROY(sched_ctx->sched_cond[workerid_in_ctx]);
		free(sched_ctx->sched_mutex[workerid_in_ctx]);
		free(sched_ctx->sched_cond[workerid_in_ctx]);
	}

	free(ht->exp_start);
	free(ht->exp_end);
	free(ht->exp_len);
	free(ht->ntasks);
	  
	free(ht);
}

struct starpu_sched_policy_s heft_policy = {
	.init_sched = heft_init,
	.deinit_sched = heft_deinit,
	.push_task = heft_push_task, 
	.push_prio_task = heft_push_prio_task, 
	.push_task_notify = heft_push_task_notify,
	.pop_task = NULL,
	.pop_every_task = NULL,
	.post_exec_hook = heft_post_exec_hook,
	.policy_name = "heft",
	.policy_description = "Heterogeneous Earliest Finish Task"
};
