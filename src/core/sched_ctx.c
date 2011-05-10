#include <core/sched_ctx.h>
#include <common/config.h>
#include <common/utils.h>
#include <core/sched_policy.h>
#include <profiling/profiling.h>

static pthread_cond_t blocking_ths_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t wakeup_ths_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t blocking_ths_mutex = PTHREAD_MUTEX_INITIALIZER;
static int nblocked_ths = 0;

int _starpu_create_sched_ctx(const char *policy_name, int *workerids_in_ctx, 
			     int nworkerids_in_ctx, unsigned is_initial_sched,
			     const char *sched_name)
{
	struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
	STARPU_ASSERT(config->topology.nsched_ctxs < STARPU_NMAX_SCHED_CTXS - 1);

	struct starpu_sched_ctx *sched_ctx = &config->sched_ctxs[config->topology.nsched_ctxs];

	sched_ctx->sched_ctx_id = config->topology.nsched_ctxs;

	int nworkers = config->topology.nworkers;
	
	STARPU_ASSERT(nworkerids_in_ctx <= nworkers);
  
	sched_ctx->nworkers_in_ctx = nworkerids_in_ctx;
	sched_ctx->sched_policy = malloc(sizeof(struct starpu_sched_policy_s));
	sched_ctx->is_initial_sched = is_initial_sched;
	sched_ctx->sched_name = sched_name;

	PTHREAD_COND_INIT(&sched_ctx->submitted_cond, NULL);
	PTHREAD_MUTEX_INIT(&sched_ctx->submitted_mutex, NULL);
	sched_ctx->nsubmitted = 0;

	int j;
	/*all the workers are in this contex*/
	if(workerids_in_ctx == NULL)
	  {
		for(j = 0; j < nworkers; j++)
		  {
			sched_ctx->workerid[j] = j;
			struct starpu_worker_s *workerarg = _starpu_get_worker_struct(j);
			workerarg->sched_ctx[workerarg->nctxs++] = sched_ctx;
		}
		sched_ctx->nworkers_in_ctx = nworkers;
	  } 
	else 
	  {
		int i;
		for(i = 0; i < nworkerids_in_ctx; i++)
		  {
			/*take care the user does not ask for a resource that does not exist*/
			STARPU_ASSERT( workerids_in_ctx[i] >= 0 &&  workerids_in_ctx[i] <= nworkers);
		    
			sched_ctx->workerid[i] = workerids_in_ctx[i];
			for(j = 0; j < nworkers; j++)
			  {
				if(sched_ctx->workerid[i] == j)
				  {
					struct starpu_worker_s *workerarg = _starpu_get_worker_struct(j);
					workerarg->sched_ctx[workerarg->nctxs++] = sched_ctx;
				  }
			  }
		  }
	  }

	_starpu_init_sched_policy(config, sched_ctx, policy_name);

	config->topology.nsched_ctxs++;	

 	return sched_ctx->sched_ctx_id;
}

void _starpu_decrement_nblocked_ths(void)
{
	PTHREAD_MUTEX_LOCK(&blocking_ths_mutex);

	if(--nblocked_ths == 0)
		PTHREAD_COND_BROADCAST(&wakeup_ths_cond);

	PTHREAD_MUTEX_UNLOCK(&blocking_ths_mutex);
}

void _starpu_increment_nblocked_ths(int nworkers)
{
	PTHREAD_MUTEX_LOCK(&blocking_ths_mutex);
	if (++nblocked_ths == nworkers)
		PTHREAD_COND_BROADCAST(&blocking_ths_cond);

	PTHREAD_MUTEX_UNLOCK(&blocking_ths_mutex);
}

static int _starpu_wait_for_all_threads_to_block(int nworkers)
{
	PTHREAD_MUTEX_LOCK(&blocking_ths_mutex);

	while (nblocked_ths < nworkers)
		PTHREAD_COND_WAIT(&blocking_ths_cond, &blocking_ths_mutex);

	PTHREAD_MUTEX_UNLOCK(&blocking_ths_mutex);
	
	return 0;
}

static int _starpu_wait_for_all_threads_to_wake_up(void)
{
	PTHREAD_MUTEX_LOCK(&blocking_ths_mutex);
	
	while (nblocked_ths > 0)
		PTHREAD_COND_WAIT(&wakeup_ths_cond, &blocking_ths_mutex);

	PTHREAD_MUTEX_UNLOCK(&blocking_ths_mutex);
	
	return 0;
}

static int set_changing_ctx_flag(starpu_worker_status changing_ctx, int nworkerids_in_ctx, int *workerids_in_ctx)
{
	struct starpu_machine_config_s *config = _starpu_get_machine_config();

	int i;
	int nworkers = nworkerids_in_ctx == -1 ? (int)config->topology.nworkers : nworkerids_in_ctx;
  
	struct starpu_worker_s *worker = NULL;
	pthread_mutex_t *changing_ctx_mutex = NULL;
	pthread_cond_t *changing_ctx_cond = NULL;
	
	int workerid = -1;

	for(i = 0; i < nworkers; i++)
	  {
		workerid = workerids_in_ctx == NULL ? i : workerids_in_ctx[i];
		worker = _starpu_get_worker_struct(workerid);

		changing_ctx_mutex = &worker->changing_ctx_mutex;
		changing_ctx_cond = &worker->changing_ctx_cond;
		
		/*if the status is CHANGING_CTX let the thread know that it must block*/
		PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
		worker->status = changing_ctx;
		worker->nworkers_of_next_ctx = nworkers;
		PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);

		/*if we have finished changing the ctx wake up the blocked threads*/
		if(changing_ctx == STATUS_UNKNOWN)
		  {
			PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
			PTHREAD_COND_SIGNAL(changing_ctx_cond);
			PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
		  }
	  }
	
	/*after letting know all the concerned threads about the change                        
	  wait for them to take into account the info*/
	if(changing_ctx == STATUS_CHANGING_CTX)
		_starpu_wait_for_all_threads_to_block(nworkers);
	else
		_starpu_wait_for_all_threads_to_wake_up();

  return 0;
}

int starpu_create_sched_ctx(const char *policy_name, int *workerids_in_ctx, 
			    int nworkerids_in_ctx, const char *sched_name)
{
	int ret;
	/* block the workers until the contex is switched */
	set_changing_ctx_flag(STATUS_CHANGING_CTX, nworkerids_in_ctx, workerids_in_ctx);
	ret = _starpu_create_sched_ctx(policy_name, workerids_in_ctx, nworkerids_in_ctx, 0, sched_name);
	/* also wait the workers to wake up before using the context */
	set_changing_ctx_flag(STATUS_UNKNOWN, nworkerids_in_ctx, workerids_in_ctx);
	
	return ret;
}

static unsigned _starpu_worker_belongs_to_ctx(struct starpu_worker_s *workerarg, struct starpu_sched_ctx *sched_ctx)
{
	unsigned i;
	for(i = 0; i < workerarg->nctxs; i++)
		if(sched_ctx != NULL && workerarg->sched_ctx[i] == sched_ctx
		   && workerarg->status != STATUS_JOINED)
		  return 1;
	return 0;
}

static void _starpu_remove_sched_ctx_from_worker(struct starpu_worker_s *workerarg, struct starpu_sched_ctx *sched_ctx)
{
	unsigned i;
	unsigned to_remove = 0;
	for(i = 0; i < workerarg->nctxs; i++)
	  {
		if(sched_ctx != NULL && workerarg->sched_ctx[i] == sched_ctx
		    && workerarg->status != STATUS_JOINED)
		  {
			workerarg->sched_ctx[i] = NULL;
			to_remove = 1;
		  }
	  }
	
	/* if the the worker had belonged to the context it would have been found in the worker's list of sched_ctxs, so it can be removed */
	if(to_remove)
		workerarg->nctxs--;
	return;
}

void starpu_delete_sched_ctx(unsigned sched_ctx_id)
{
	if(!starpu_wait_for_all_tasks_of_sched_ctx(sched_ctx_id))
	  {
		struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);

		int nworkers = sched_ctx->nworkers_in_ctx;
		int workerid;

		int i;
		for(i = 0; i < nworkers; i++)
		  {
			workerid = sched_ctx->workerid[i];
			struct starpu_worker_s *workerarg = _starpu_get_worker_struct(workerid);
			_starpu_remove_sched_ctx_from_worker(workerarg, sched_ctx);
		  }
	
		free(sched_ctx->sched_policy);
		sched_ctx->sched_policy = NULL;
	  }		
	return;	
}

void _starpu_delete_all_sched_ctxs()
{
	struct starpu_machine_config_s *config = _starpu_get_machine_config();
	unsigned nsched_ctxs = config->topology.nsched_ctxs;
	unsigned i;

	for(i = 0; i < nsched_ctxs; i++)
	  {
	    starpu_delete_sched_ctx((int)i);
	  }
	return;
}

int starpu_wait_for_all_tasks_of_worker(int workerid)
{
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

	struct starpu_worker_s *worker = _starpu_get_worker_struct(workerid);
	
	PTHREAD_MUTEX_LOCK(&worker->submitted_mutex);

	while (worker->nsubmitted > 0)
		PTHREAD_COND_WAIT(&worker->submitted_cond, &worker->submitted_mutex);

	PTHREAD_MUTEX_UNLOCK(&worker->submitted_mutex);
	
	return 0;
}

int starpu_wait_for_all_tasks_of_workers(int *workerids_in_ctx, int nworkerids_in_ctx){
	int ret_val = 0;
	
	struct starpu_machine_config_s *config = _starpu_get_machine_config();
	int nworkers = nworkerids_in_ctx == -1 ? (int)config->topology.nworkers : nworkerids_in_ctx;
	
	int workerid = -1;
	int i, n;
	
	for(i = 0; i < nworkers; i++)
	  {
		workerid = workerids_in_ctx == NULL ? i : workerids_in_ctx[i];
		n = starpu_wait_for_all_tasks_of_worker(workerid);
		ret_val = (ret_val && n);
	  }
	
	return ret_val;
}

void _starpu_decrement_nsubmitted_tasks_of_worker(int workerid)
{
	struct starpu_worker_s *worker = _starpu_get_worker_struct(workerid);
	
	PTHREAD_MUTEX_LOCK(&worker->submitted_mutex);

	if (--worker->nsubmitted == 0)
		PTHREAD_COND_BROADCAST(&worker->submitted_cond);

	PTHREAD_MUTEX_UNLOCK(&worker->submitted_mutex);
	return;
}

void _starpu_increment_nsubmitted_tasks_of_worker(int workerid)
{
	struct starpu_worker_s *worker = _starpu_get_worker_struct(workerid);

	PTHREAD_MUTEX_LOCK(&worker->submitted_mutex);

	worker->nsubmitted++;
	
	PTHREAD_MUTEX_UNLOCK(&worker->submitted_mutex);
	return;
}

static void _starpu_add_workers_to_sched_ctx(int *workerids_in_ctx, int nworkerids_in_ctx, 
				     struct starpu_sched_ctx *sched_ctx)
{
	struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
	int nworkers = config->topology.nworkers;
	
	STARPU_ASSERT((nworkerids_in_ctx + sched_ctx->nworkers_in_ctx) <= nworkers);
	
	int nworkerids_already_in_ctx =  sched_ctx->nworkers_in_ctx;
	int j;

	/*if null add the rest of the workers which don't already belong to this ctx*/
	if(workerids_in_ctx == NULL)
	  {
		for(j = 0; j < nworkers; j++)
		  {
			struct starpu_worker_s *workerarg = _starpu_get_worker_struct(j);
			if(!_starpu_worker_belongs_to_ctx(workerarg, sched_ctx))
			  {
				sched_ctx->workerid[++nworkerids_already_in_ctx] = j;
				workerarg->sched_ctx[workerarg->nctxs++] = sched_ctx;
			  }
			sched_ctx->nworkers_in_ctx = nworkers;
		  }
	  } 
	else 
	  {
		int i;
		for(i = 0; i < nworkerids_in_ctx; i++)
		  {
			/*take care the user does not ask for a resource that does not exist*/
			STARPU_ASSERT( workerids_in_ctx[i] >= 0 &&  workerids_in_ctx[i] <= nworkers);
		    
			sched_ctx->workerid[ nworkerids_already_in_ctx + i] = workerids_in_ctx[i];
			for(j = 0; j < nworkers; j++)
			  {
				if(sched_ctx->workerid[i] == j)
				  {
					struct starpu_worker_s *workerarg = _starpu_get_worker_struct(j);
					workerarg->sched_ctx[workerarg->nctxs++] = sched_ctx;
				  }
			  }
		  }
		sched_ctx->nworkers_in_ctx = nworkerids_in_ctx;
	  }
	return;
}

void starpu_add_workers_to_sched_ctx(int *workerids_in_ctx, int nworkerids_in_ctx,
				     unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);

	/* block the workers until the contex is switched */
	set_changing_ctx_flag(STATUS_CHANGING_CTX, nworkerids_in_ctx, workerids_in_ctx);
	_starpu_add_workers_to_sched_ctx(workerids_in_ctx, nworkerids_in_ctx, sched_ctx);
	/* also wait the workers to wake up before using the context */
	set_changing_ctx_flag(STATUS_UNKNOWN, nworkerids_in_ctx, workerids_in_ctx);

	return;
}

static int _starpu_get_first_free_space(int *workerids, int old_nworkerids_in_ctx)
{
  int i;
  for(i = 0; i < old_nworkerids_in_ctx; i++)
    if(workerids[i] == -1)
      return i;
  return -1;
}

/* rearange array of workerids in order not to have {-1, -1, 5, -1, 7}
   and have instead {5, 7, -1, -1, -1} 
   it is easier afterwards to iterate the array
*/
static void _starpu_rearange_sched_ctx_workerids(struct starpu_sched_ctx *sched_ctx, int old_nworkerids_in_ctx)
{
	int first_free_id = -1;
	int i;
	for(i = 0; i < old_nworkerids_in_ctx; i++)
	  {
		if(sched_ctx->workerid[i] != -1)
		  {
			first_free_id = _starpu_get_first_free_space(sched_ctx->workerid, 
								     old_nworkerids_in_ctx);
			if(first_free_id != -1)
			  {
				sched_ctx->workerid[first_free_id] = sched_ctx->workerid[i];
				sched_ctx->workerid[i] = -1;
			  }
		  }
	  }
}

static void _starpu_remove_workers_from_sched_ctx(int *workerids_in_ctx, int nworkerids_in_ctx, 
					  struct starpu_sched_ctx *sched_ctx)
{
  	struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
	int nworkers = config->topology.nworkers;
	
	int nworkerids_already_in_ctx =  sched_ctx->nworkers_in_ctx;

	STARPU_ASSERT(nworkerids_in_ctx  <= nworkerids_already_in_ctx);

	int i, workerid;

	/*if null remove all the workers that belong to this ctx*/
	if(workerids_in_ctx == NULL)
	  {
		for(i = 0; i < nworkerids_already_in_ctx; i++)
		  {
			workerid = sched_ctx->workerid[i];
			struct starpu_worker_s *workerarg = _starpu_get_worker_struct(workerid);
			_starpu_remove_sched_ctx_from_worker(workerarg, sched_ctx);
			sched_ctx->workerid[i] = -1;
		  }

		sched_ctx->nworkers_in_ctx = 0;
	  } 
	else 
	  {
		for(i = 0; i < nworkerids_in_ctx; i++)
		  {
		    	workerid = workerids_in_ctx[i]; 
			/* take care the user does not ask for a resource that does not exist */
			STARPU_ASSERT( workerid >= 0 &&  workerid <= nworkers);

			struct starpu_worker_s *workerarg = _starpu_get_worker_struct(workerid);
			_starpu_remove_sched_ctx_from_worker(workerarg, sched_ctx);
			int j;

			/* don't leave the workerid with a correct value even if we don't use it anymore */
			for(j = 0; j < nworkerids_already_in_ctx; j++)
				if(sched_ctx->workerid[j] == workerid)				 
					sched_ctx->workerid[j] = -1;
		  }

		sched_ctx->nworkers_in_ctx -= nworkerids_in_ctx;
		_starpu_rearange_sched_ctx_workerids(sched_ctx, nworkerids_already_in_ctx);
	  }

	return;

}

void starpu_remove_workers_from_sched_ctx(int *workerids_in_ctx, int nworkerids_in_ctx, 
					  unsigned sched_ctx_id)
{
	  /* wait for the workers concerned by the change of contex                       
	   * to finish their work in the previous context */
	if(!starpu_wait_for_all_tasks_of_workers(workerids_in_ctx, nworkerids_in_ctx))
	  {
		struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);

		/* block the workers until the contex is switched */
		set_changing_ctx_flag(STATUS_CHANGING_CTX, nworkerids_in_ctx, workerids_in_ctx);
		_starpu_remove_workers_from_sched_ctx(workerids_in_ctx, nworkerids_in_ctx, sched_ctx);
		/* also wait the workers to wake up before using the context */
		set_changing_ctx_flag(STATUS_UNKNOWN, nworkerids_in_ctx, workerids_in_ctx);
	  }
	return;

}

int starpu_wait_for_all_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
	  return -EDEADLK;
	
	PTHREAD_MUTEX_LOCK(&sched_ctx->submitted_mutex);
	
	
	while (sched_ctx->nsubmitted > 0)
	  PTHREAD_COND_WAIT(&sched_ctx->submitted_cond, &sched_ctx->submitted_mutex);
	
	PTHREAD_MUTEX_UNLOCK(&sched_ctx->submitted_mutex);
	
	return 0;
}

void _starpu_decrement_nsubmitted_tasks_of_sched_ctx(struct starpu_sched_ctx *sched_ctx)
{
  PTHREAD_MUTEX_LOCK(&sched_ctx->submitted_mutex);

  if (--sched_ctx->nsubmitted == 0)
    PTHREAD_COND_BROADCAST(&sched_ctx->submitted_cond);

  PTHREAD_MUTEX_UNLOCK(&sched_ctx->submitted_mutex);
}

void _starpu_increment_nsubmitted_tasks_of_sched_ctx(struct starpu_sched_ctx *sched_ctx)
{
  PTHREAD_MUTEX_LOCK(&sched_ctx->submitted_mutex);

  sched_ctx->nsubmitted++;

  PTHREAD_MUTEX_UNLOCK(&sched_ctx->submitted_mutex);
}
