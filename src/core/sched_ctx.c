#include <core/sched_policy.h>
#include <core/sched_ctx.h>

static struct _starpu_barrier_counter_t workers_barrier;

static unsigned _starpu_get_first_available_sched_ctx_id(struct starpu_machine_config_s *config);
static unsigned _starpu_get_first_free_sched_ctx_in_worker_list(struct starpu_worker_s *worker);
static void _starpu_rearange_sched_ctx_workerids(struct starpu_sched_ctx *sched_ctx, int old_nworkerids_in_ctx);
int set_changing_ctx_flag(starpu_worker_status changing_ctx, int nworkerids_in_ctx, int *workerids_in_ctx);

unsigned _starpu_create_sched_ctx(const char *policy_name, int *workerids_in_ctx, 
			     int nworkerids_in_ctx, unsigned is_initial_sched,
			     const char *sched_name)
{
	struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
	STARPU_ASSERT(config->topology.nsched_ctxs < STARPU_NMAX_SCHED_CTXS - 1);

	struct starpu_sched_ctx *sched_ctx = &config->sched_ctxs[config->topology.nsched_ctxs];

	sched_ctx->sched_ctx_id = _starpu_get_first_available_sched_ctx_id(config);

	int nworkers = config->topology.nworkers;
	
	STARPU_ASSERT(nworkerids_in_ctx <= nworkers);
  
	sched_ctx->nworkers_in_ctx = nworkerids_in_ctx;
	sched_ctx->sched_policy = malloc(sizeof(struct starpu_sched_policy_s));
	sched_ctx->is_initial_sched = is_initial_sched;
	sched_ctx->sched_name = sched_name;

	_starpu_barrier_counter_init(&sched_ctx->tasks_barrier, 0);

	int j;
	/*all the workers are in this contex*/
	if(workerids_in_ctx == NULL)
	  {
		for(j = 0; j < nworkers; j++)
		  {
			sched_ctx->workerid[j] = j;
			struct starpu_worker_s *workerarg = _starpu_get_worker_struct(j);
			workerarg->sched_ctx[_starpu_get_first_free_sched_ctx_in_worker_list(workerarg)] = sched_ctx;
		}
		sched_ctx->nworkers_in_ctx = nworkers;
	  } 
	else 
	  {
		int i;
		for(i = 0; i < nworkerids_in_ctx; i++)
		  {
			/* the user should not ask for a resource that does not exist */
			STARPU_ASSERT( workerids_in_ctx[i] >= 0 &&  workerids_in_ctx[i] <= nworkers);
		    
			sched_ctx->workerid[i] = workerids_in_ctx[i];
			struct starpu_worker_s *workerarg = _starpu_get_worker_struct(sched_ctx->workerid[i]);
			workerarg->sched_ctx[_starpu_get_first_free_sched_ctx_in_worker_list(workerarg)] = sched_ctx;
		  }
	  }

	/* initialise all sync structures bc the number of workers can modify */
	sched_ctx->sched_mutex = (pthread_mutex_t**)malloc(STARPU_NMAXWORKERS* sizeof(pthread_mutex_t*));
	sched_ctx->sched_cond = (pthread_cond_t**)malloc(STARPU_NMAXWORKERS *sizeof(pthread_cond_t*));


	_starpu_init_sched_policy(config, sched_ctx, policy_name);

	config->topology.nsched_ctxs++;	

 	return sched_ctx->sched_ctx_id;
}


unsigned starpu_create_sched_ctx(const char *policy_name, int *workerids_in_ctx, 
			    int nworkerids_in_ctx, const char *sched_name)
{
	unsigned ret;
	/* block the workers until the contex is switched */
	set_changing_ctx_flag(STATUS_CHANGING_CTX, nworkerids_in_ctx, workerids_in_ctx);
	ret = _starpu_create_sched_ctx(policy_name, workerids_in_ctx, nworkerids_in_ctx, 0, sched_name);
	/* also wait the workers to wake up before using the context */
	set_changing_ctx_flag(STATUS_UNKNOWN, nworkerids_in_ctx, workerids_in_ctx);
	
	return ret;
}

static unsigned _starpu_worker_belongs_to_ctx(int workerid, struct starpu_sched_ctx *sched_ctx)
{
	int i;
	for(i = 0; i < sched_ctx->nworkers_in_ctx; i++)
	  if(sched_ctx->workerid[i] == workerid)
		  return 1;
	return 0;
}

static void _starpu_remove_sched_ctx_from_worker(struct starpu_worker_s *workerarg, struct starpu_sched_ctx *sched_ctx)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	  {
		if(sched_ctx != NULL && workerarg->sched_ctx[i] == sched_ctx
		    && workerarg->status != STATUS_JOINED)
		  {
			workerarg->sched_ctx[i] = NULL;
			workerarg->nctxs--;
			break;
		  }
	  }
	
	return;
}

static void _starpu_manage_delete_sched_ctx(struct starpu_sched_ctx *sched_ctx)
{
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
	free(sched_ctx->sched_mutex);
	free(sched_ctx->sched_cond);
	sched_ctx->sched_policy = NULL;
	struct starpu_machine_config_s *config = _starpu_get_machine_config();
	config->topology.nsched_ctxs--;
	sched_ctx->sched_ctx_id = STARPU_NMAX_SCHED_CTXS;
}

static void _starpu_add_workers_to_sched_ctx(int *new_workers, int nnew_workers,
                                     struct starpu_sched_ctx *sched_ctx)
{
        struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
        int ntotal_workers = config->topology.nworkers;

	// STARPU_ASSERT((nnew_workers + sched_ctx->nworkers_in_ctx) <= ntotal_workers);

        int nworkerids_already_in_ctx =  sched_ctx->nworkers_in_ctx;
        int j;
	
	int n_added_workers = 0;
		
        /*if null add the rest of the workers which don't already belong to this ctx*/
        if(new_workers == NULL)
          {
                for(j = 0; j < ntotal_workers; j++)
                  {
                        struct starpu_worker_s *workerarg = _starpu_get_worker_struct(j);
                        if(!_starpu_worker_belongs_to_ctx(j, sched_ctx))
                          {
                                sched_ctx->workerid[++nworkerids_already_in_ctx] = j;
                                workerarg->sched_ctx[_starpu_get_first_free_sched_ctx_in_worker_list(workerarg)] = sched_ctx;
                          }
                  }
		//                sched_ctx->nworkers_in_ctx = ntotal_workers;
		n_added_workers = ntotal_workers;
          }
        else
          {
                int i;
		printf("%d redim worker:", nnew_workers);
                for(i = 0; i < nnew_workers; i++)
                  {
                        /*take care the user does not ask for a resource that does not exist*/
                        STARPU_ASSERT( new_workers[i] >= 0 &&  new_workers[i] <= ntotal_workers);
			printf(" %d", new_workers[i]);

			struct starpu_worker_s *workerarg = _starpu_get_worker_struct(new_workers[i]);
			if(!_starpu_worker_belongs_to_ctx(new_workers[i], sched_ctx))
			  {
			    /* add worker to context */
			    sched_ctx->workerid[ nworkerids_already_in_ctx + n_added_workers] = new_workers[i];
			    /* add context to worker */
			    workerarg->sched_ctx[_starpu_get_first_free_sched_ctx_in_worker_list(workerarg)] = sched_ctx;
			    n_added_workers++;
			  }
                  }
		//                sched_ctx->nworkers_in_ctx += n_added_workers;
		printf("\n");
          }

        sched_ctx->sched_policy->init_sched_for_workers(sched_ctx->sched_ctx_id, n_added_workers);

        return;
}

void starpu_delete_sched_ctx(unsigned sched_ctx_id, unsigned inheritor_sched_ctx_id)
{
	if(!starpu_wait_for_all_tasks_of_sched_ctx(sched_ctx_id))
	  {

		struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
		struct starpu_sched_ctx *inheritor_sched_ctx = _starpu_get_sched_ctx(inheritor_sched_ctx_id);

		/* block the workers until the contex is switched */
		set_changing_ctx_flag(STATUS_CHANGING_CTX, sched_ctx->nworkers_in_ctx, sched_ctx->workerid);
		_starpu_manage_delete_sched_ctx(sched_ctx);

		/*if both of them have all the ressources is pointless*/
		/*trying to transfer ressources from one ctx to the other*/
		struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
		int ntotal_workers = config->topology.nworkers;

		if(!(sched_ctx->nworkers_in_ctx == ntotal_workers && sched_ctx->nworkers_in_ctx == inheritor_sched_ctx->nworkers_in_ctx))
		  _starpu_add_workers_to_sched_ctx(sched_ctx->workerid, sched_ctx->nworkers_in_ctx, inheritor_sched_ctx);
		/* also wait the workers to wake up before using the context */
		set_changing_ctx_flag(STATUS_UNKNOWN, sched_ctx->nworkers_in_ctx, sched_ctx->workerid);
	  }		
	return;	
}

void _starpu_delete_all_sched_ctxs()
{
	unsigned i;

	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	  {
	    struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(i);
	    if(sched_ctx->sched_ctx_id != STARPU_NMAX_SCHED_CTXS){
	      if(!starpu_wait_for_all_tasks_of_sched_ctx(i))
		{
		  _starpu_manage_delete_sched_ctx(sched_ctx);
		}		
	    }
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

/* unused sched_ctx have the id STARPU_NMAX_SCHED_CTXS */
void _starpu_init_all_sched_ctx(struct starpu_machine_config_s *config)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		config->sched_ctxs[i].sched_ctx_id = STARPU_NMAX_SCHED_CTXS;
	return;
}

/* unused sched_ctx pointers of a worker are NULL */
void _starpu_init_sched_ctx_for_worker(unsigned workerid)
{
	struct starpu_worker_s *worker = _starpu_get_worker_struct(workerid);
	worker->sched_ctx = (struct starpu_sched_ctx**)malloc(STARPU_NMAX_SCHED_CTXS * sizeof(struct starpu_sched_ctx*));
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		worker->sched_ctx[i] = NULL;
	return;
}

/* sched_ctx aren't necessarly one next to another */
/* for eg when we remove one its place is free */
/* when we add  new one we reuse its place */
static unsigned _starpu_get_first_available_sched_ctx_id(struct starpu_machine_config_s *config)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		if(config->sched_ctxs[i].sched_ctx_id == STARPU_NMAX_SCHED_CTXS)
			return i;

	STARPU_ASSERT(0);
	return STARPU_NMAX_SCHED_CTXS;
}

static unsigned _starpu_get_first_free_sched_ctx_in_worker_list(struct starpu_worker_s *worker)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		if(worker->sched_ctx[i] == NULL)
			return i;
	STARPU_ASSERT(0);
	return STARPU_NMAX_SCHED_CTXS;
}

static int _starpu_get_first_free_worker_space(int *workerids, int old_nworkerids_in_ctx)
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
			first_free_id = _starpu_get_first_free_worker_space(sched_ctx->workerid, old_nworkerids_in_ctx);
			if(first_free_id != -1)
			  {
				sched_ctx->workerid[first_free_id] = sched_ctx->workerid[i];
				sched_ctx->workerid[i] = -1;
			  }
		  }
	  }
}

/* manage blocking and waking up threads when constructing/modifying contexts */
int set_changing_ctx_flag(starpu_worker_status changing_ctx, int nworkerids_in_ctx, int *workerids_in_ctx)
{
	struct starpu_machine_config_s *config = _starpu_get_machine_config();

	int i;
	int nworkers = nworkerids_in_ctx == -1 ? (int)config->topology.nworkers : nworkerids_in_ctx;
  
	struct starpu_worker_s *worker = NULL;
	pthread_mutex_t *changing_ctx_mutex = NULL;
	pthread_cond_t *changing_ctx_cond = NULL;
	
	int workerid = -1;

	if(changing_ctx == STATUS_CHANGING_CTX)
	  _starpu_barrier_counter_init(&workers_barrier, nworkers);
	
	for(i = 0; i < nworkers; i++)
	  {
		workerid = workerids_in_ctx == NULL ? i : workerids_in_ctx[i];
		worker = _starpu_get_worker_struct(workerid);

		changing_ctx_mutex = &worker->changing_ctx_mutex;
		changing_ctx_cond = &worker->changing_ctx_cond;
		
		/*if the status is CHANGING_CTX let the thread know that it must block*/
		PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
		worker->blocking_status = changing_ctx;
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
	
	/* after letting know all the concerned threads about the change
	   wait for them to take into account the info */
	if(changing_ctx == STATUS_CHANGING_CTX)
		_starpu_wait_for_all_threads_to_block();
	else
	  {
		_starpu_wait_for_all_threads_to_wake_up();
		_starpu_barrier_counter_destroy(&workers_barrier);
	  }

	return 0;
}


int starpu_wait_for_all_tasks_of_worker(int workerid)
{
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
		return -EDEADLK;

	struct starpu_worker_s *worker = _starpu_get_worker_struct(workerid);
	
	_starpu_barrier_counter_wait_for_empty_counter(&worker->tasks_barrier);
	
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
	
	_starpu_barrier_counter_decrement_until_empty_counter(&worker->tasks_barrier);

	return;
}

void _starpu_increment_nsubmitted_tasks_of_worker(int workerid)
{
	struct starpu_worker_s *worker = _starpu_get_worker_struct(workerid);

	_starpu_barrier_counter_increment(&worker->tasks_barrier);

	return;
}

void _starpu_decrement_nblocked_ths(void)
{
	_starpu_barrier_counter_decrement_until_empty_counter(&workers_barrier);
}

void _starpu_increment_nblocked_ths(void)
{
	_starpu_barrier_counter_increment_until_full_counter(&workers_barrier);
}

int _starpu_wait_for_all_threads_to_block(void)
{
	return _starpu_barrier_counter_wait_for_full_counter(&workers_barrier);
}

int _starpu_wait_for_all_threads_to_wake_up(void)
{
	return _starpu_barrier_counter_wait_for_empty_counter(&workers_barrier);
}

int starpu_wait_for_all_tasks_of_sched_ctx(unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	
	if (STARPU_UNLIKELY(!_starpu_worker_may_perform_blocking_calls()))
	  return -EDEADLK;
	
	return _starpu_barrier_counter_wait_for_empty_counter(&sched_ctx->tasks_barrier);
}

void _starpu_decrement_nsubmitted_tasks_of_sched_ctx(struct starpu_sched_ctx *sched_ctx)
{
	_starpu_barrier_counter_decrement_until_empty_counter(&sched_ctx->tasks_barrier);
}

void _starpu_increment_nsubmitted_tasks_of_sched_ctx(struct starpu_sched_ctx *sched_ctx)
{
	_starpu_barrier_counter_increment(&sched_ctx->tasks_barrier);
}


int _starpu_get_index_in_ctx_of_workerid(unsigned sched_ctx_id, unsigned workerid)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx(sched_ctx_id);
	
	int nworkers_in_ctx = sched_ctx->nworkers_in_ctx;

	int i;
	for(i = 0; i < nworkers_in_ctx; i++)
		if(sched_ctx->workerid[i] == (int)workerid)
			return i;
	
	return -1;
}

pthread_mutex_t *_starpu_get_sched_mutex(struct starpu_sched_ctx *sched_ctx, int worker)
{
	int workerid_ctx = _starpu_get_index_in_ctx_of_workerid(sched_ctx->sched_ctx_id, worker);
	return sched_ctx->sched_mutex[workerid_ctx];
}

pthread_cond_t *_starpu_get_sched_cond(struct starpu_sched_ctx *sched_ctx, int worker)
{
	int workerid_ctx = _starpu_get_index_in_ctx_of_workerid(sched_ctx->sched_ctx_id, worker);
	return sched_ctx->sched_cond[workerid_ctx];
}
