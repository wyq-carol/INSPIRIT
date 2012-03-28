#include <sched_ctx_hypervisor_intern.h>

unsigned imposed_resize = 0;
struct starpu_performance_counters* perf_counters = NULL;

static void notify_idle_cycle(unsigned sched_ctx, int worker, double idle_time);
static void notify_pushed_task(unsigned sched_ctx, int worker);
static void notify_poped_task(unsigned sched_ctx, int worker, double flops);
static void notify_post_exec_hook(unsigned sched_ctx, int taskid);
static void notify_idle_end(unsigned sched_ctx, int  worker);


extern struct hypervisor_policy idle_policy;
extern struct hypervisor_policy app_driven_policy;
extern struct hypervisor_policy gflops_rate_policy;

static struct hypervisor_policy *predefined_policies[] = {
        &idle_policy,
	&app_driven_policy,
	&gflops_rate_policy
};

static void _load_hypervisor_policy(struct hypervisor_policy *policy)
{
        STARPU_ASSERT(policy);

#ifdef STARPU_VERBOSE
        if (policy->name)
        {
		_STARPU_DEBUG("Use %s hypervisor policy \n", policy->name);
        }
#endif
	hypervisor.policy.name = policy->name;
	hypervisor.policy.handle_poped_task = policy->handle_poped_task;
	hypervisor.policy.handle_pushed_task = policy->handle_pushed_task;
	hypervisor.policy.handle_idle_cycle = policy->handle_idle_cycle;
	hypervisor.policy.handle_idle_end = policy->handle_idle_end;
	hypervisor.policy.handle_post_exec_hook = policy->handle_post_exec_hook;
}


static struct hypervisor_policy *_find_hypervisor_policy_from_name(const char *policy_name)
{

        if (!policy_name)
                return NULL;

        unsigned i;
        for (i = 0; i < sizeof(predefined_policies)/sizeof(predefined_policies[0]); i++)
        {
                struct hypervisor_policy *p;
                p = predefined_policies[i];
                if (p->name)
                {
                        if (strcmp(policy_name, p->name) == 0) {
                                /* we found a policy with the requested name */
                                return p;
                        }
                }
        }
        fprintf(stderr, "Warning: hypervisor policy \"%s\" was not found, try \"help\" to get a list\n", policy_name);

        /* nothing was found */
        return NULL;
}

static struct hypervisor_policy *_select_hypervisor_policy(struct hypervisor_policy* hypervisor_policy)
{
	struct hypervisor_policy *selected_policy = NULL;

	if(hypervisor_policy && hypervisor_policy->custom)
		return hypervisor_policy;

        /* we look if the application specified the name of a policy to load */
        const char *policy_name;
        if (hypervisor_policy && hypervisor_policy->name)
        {
                policy_name = hypervisor_policy->name;
        }
        else 
	{
                policy_name = getenv("HYPERVISOR_POLICY");
        }

        if (policy_name)
                selected_policy = _find_hypervisor_policy_from_name(policy_name);

        /* Perhaps there was no policy that matched the name */
        if (selected_policy)
                return selected_policy;

        /* If no policy was specified, we use the idle policy as a default */

        return &idle_policy;
}


/* initializez the performance counters that starpu will use to retrive hints for resizing */
struct starpu_performance_counters* sched_ctx_hypervisor_init(struct hypervisor_policy *hypervisor_policy)
{
	hypervisor.min_tasks = 0;
	hypervisor.nsched_ctxs = 0;
	pthread_mutex_init(&act_hypervisor_mutex, NULL);

	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
		hypervisor.resize[i] = 0;
		hypervisor.configurations[i] = NULL;
		hypervisor.sched_ctxs[i] = STARPU_NMAX_SCHED_CTXS;
		hypervisor.sched_ctx_w[i].sched_ctx = STARPU_NMAX_SCHED_CTXS;
		hypervisor.sched_ctx_w[i].config = NULL;
		hypervisor.sched_ctx_w[i].total_flops = 0.0;
		hypervisor.sched_ctx_w[i].remaining_flops = 0.0;
		hypervisor.sched_ctx_w[i].start_time = 0.0;
		hypervisor.sched_ctx_w[i].resize_ack.receiver_sched_ctx = -1;
		hypervisor.sched_ctx_w[i].resize_ack.moved_workers = NULL;
		hypervisor.sched_ctx_w[i].resize_ack.nmoved_workers = 0;
		hypervisor.sched_ctx_w[i].resize_ack.acked_workers = NULL;


		int j;
		for(j = 0; j < STARPU_NMAXWORKERS; j++)
		{
			hypervisor.sched_ctx_w[i].current_idle_time[j] = 0.0;
			hypervisor.sched_ctx_w[i].pushed_tasks[j] = 0;
			hypervisor.sched_ctx_w[i].poped_tasks[j] = 0;
			hypervisor.sched_ctx_w[i].elapsed_flops[j] = 0.0;
			hypervisor.sched_ctx_w[i].total_elapsed_flops[j] = 0.0;

		}
	}

	struct hypervisor_policy *selected_hypervisor_policy = _select_hypervisor_policy(hypervisor_policy);
	_load_hypervisor_policy(selected_hypervisor_policy);

	perf_counters = (struct starpu_performance_counters*)malloc(sizeof(struct starpu_performance_counters));
	perf_counters->notify_idle_cycle = notify_idle_cycle;
	perf_counters->notify_pushed_task = notify_pushed_task;
	perf_counters->notify_poped_task = notify_poped_task;
	perf_counters->notify_post_exec_hook = notify_post_exec_hook;
	perf_counters->notify_idle_end = notify_idle_end;

	starpu_notify_hypervisor_exists();

	return perf_counters;
}

char* sched_ctx_hypervisor_get_policy()
{
	return hypervisor.policy.name;
}

/* the user can forbid the resizing process*/
void sched_ctx_hypervisor_stop_resize(unsigned sched_ctx)
{
	imposed_resize = 1;
	hypervisor.resize[sched_ctx] = 0;
}

/* the user can restart the resizing process*/
void sched_ctx_hypervisor_start_resize(unsigned sched_ctx)
{
	imposed_resize = 1;
	hypervisor.resize[sched_ctx] = 1;
}

void sched_ctx_hypervisor_shutdown(void)
{
	printf("shutdown\n");
	int i;
	for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
	{
                if(hypervisor.sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS && hypervisor.nsched_ctxs > 0)
		{
			sched_ctx_hypervisor_stop_resize(hypervisor.sched_ctxs[i]);
			sched_ctx_hypervisor_unregister_ctx(hypervisor.sched_ctxs[i]);
		}
	}
	perf_counters->notify_idle_cycle = NULL;
	perf_counters->notify_pushed_task = NULL;
	perf_counters->notify_poped_task = NULL;
	perf_counters->notify_post_exec_hook = NULL;
	perf_counters->notify_idle_end = NULL;

	free(perf_counters);
	perf_counters = NULL;

	pthread_mutex_destroy(&act_hypervisor_mutex);
}

/* the hypervisor is in charge only of the contexts registered to it*/
void sched_ctx_hypervisor_register_ctx(unsigned sched_ctx, double total_flops)
{	
	hypervisor.configurations[sched_ctx] = (struct starpu_htbl32_node*)malloc(sizeof(struct starpu_htbl32_node));
	hypervisor.resize_requests[sched_ctx] = (struct starpu_htbl32_node*)malloc(sizeof(struct starpu_htbl32_node));

	_add_config(sched_ctx);
	hypervisor.sched_ctx_w[sched_ctx].sched_ctx = sched_ctx;
	hypervisor.sched_ctxs[hypervisor.nsched_ctxs++] = sched_ctx;

	hypervisor.sched_ctx_w[sched_ctx].total_flops = total_flops;
	hypervisor.sched_ctx_w[sched_ctx].remaining_flops = total_flops;
	if(strcmp(hypervisor.policy.name, "app_driven") == 0)
		hypervisor.resize[sched_ctx] = 1;
}

static int _get_first_free_sched_ctx(int *sched_ctxs, unsigned nsched_ctxs)
{
        int i;
        for(i = 0; i < nsched_ctxs; i++)
                if(sched_ctxs[i] == STARPU_NMAX_SCHED_CTXS)
                        return i;

        return STARPU_NMAX_SCHED_CTXS;
}

/* rearange array of sched_ctxs in order not to have {MAXVAL, MAXVAL, 5, MAXVAL, 7}    
   and have instead {5, 7, MAXVAL, MAXVAL, MAXVAL}                                    
   it is easier afterwards to iterate the array                           
*/
static void _rearange_sched_ctxs(int *sched_ctxs, int old_nsched_ctxs)
{
        int first_free_id = STARPU_NMAX_SCHED_CTXS;
        int i;
        for(i = 0; i < old_nsched_ctxs; i++)
        {
                if(sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS)
                {
                        first_free_id = _get_first_free_sched_ctx(sched_ctxs, old_nsched_ctxs);
                        if(first_free_id != STARPU_NMAX_SCHED_CTXS)
			{
                                sched_ctxs[first_free_id] = sched_ctxs[i];
				sched_ctxs[i] = STARPU_NMAX_SCHED_CTXS;
			}
                }
	}
}

/* unregistered contexts will no longer be resized */
void sched_ctx_hypervisor_unregister_ctx(unsigned sched_ctx)
{
        unsigned i;
        for(i = 0; i < hypervisor.nsched_ctxs; i++)
        {
                if(hypervisor.sched_ctxs[i] == sched_ctx)
                {
                        hypervisor.sched_ctxs[i] = STARPU_NMAX_SCHED_CTXS;
			break;
                }
        }

        _rearange_sched_ctxs(hypervisor.sched_ctxs, hypervisor.nsched_ctxs);
	hypervisor.nsched_ctxs--;
	hypervisor.sched_ctx_w[sched_ctx].sched_ctx = STARPU_NMAX_SCHED_CTXS;
	_remove_config(sched_ctx);

	free(hypervisor.configurations[sched_ctx]);
	free(hypervisor.resize_requests[sched_ctx]);
}

static int get_ntasks( int *tasks)
{
	int ntasks = 0;
	int j;
	for(j = 0; j < STARPU_NMAXWORKERS; j++)
	{
		ntasks += tasks[j];
	}
	return ntasks;
}


static void _get_cpus(int *workers, int nworkers, int *cpus, int *ncpus)
{
	int i, worker;
	*ncpus = 0;

	for(i = 0; i < nworkers; i++)
	{
		worker = workers[i];
		enum starpu_archtype arch = starpu_worker_get_type(worker);
		if(arch == STARPU_CPU_WORKER)
			cpus[(*ncpus)++] = worker;
	}
}

/* actually move the workers: the cpus are moved, gpus are only shared  */
/* forbids another resize request before this one is take into account */
void sched_ctx_hypervisor_move_workers(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, int* workers_to_move, unsigned nworkers_to_move)
{
	if(nworkers_to_move > 0 && hypervisor.resize[sender_sched_ctx] && hypervisor.resize[receiver_sched_ctx])
	{
		int j;
		printf("resize ctx %d with", sender_sched_ctx);
		for(j = 0; j < nworkers_to_move; j++)
			printf(" %d", workers_to_move[j]);
		printf("\n");

		int *cpus = (int*) malloc(nworkers_to_move * sizeof(int));
		int ncpus;

		_get_cpus(workers_to_move, nworkers_to_move, cpus, &ncpus);

		if(ncpus != 0)
			starpu_remove_workers_from_sched_ctx(cpus, ncpus, sender_sched_ctx);

		starpu_add_workers_to_sched_ctx(workers_to_move, nworkers_to_move, receiver_sched_ctx);

		hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.receiver_sched_ctx = receiver_sched_ctx;
		hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.moved_workers = (int*)malloc(nworkers_to_move * sizeof(int));
		hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.nmoved_workers = nworkers_to_move;
		hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.acked_workers = (int*)malloc(nworkers_to_move * sizeof(int));


		int i;
		for(i = 0; i < nworkers_to_move; i++)
		{
			hypervisor.sched_ctx_w[sender_sched_ctx].current_idle_time[workers_to_move[i]] = 0.0;
			hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.moved_workers[i] = workers_to_move[i];	
			hypervisor.sched_ctx_w[sender_sched_ctx].resize_ack.acked_workers[i] = 0;	
		}

		hypervisor.resize[sender_sched_ctx] = 0;
		hypervisor.resize[receiver_sched_ctx] = 0;
	}

	return;
}

static void _set_elapsed_flops_per_sched_ctx(unsigned sched_ctx, double val)
{
	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		hypervisor.sched_ctx_w[sched_ctx].elapsed_flops[i] = val;
}

double sched_ctx_hypervisor_get_elapsed_flops_per_sched_ctx(struct sched_ctx_wrapper* sc_w)
{
	double ret_val = 0.0;
	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		ret_val += sc_w->elapsed_flops[i];
	return ret_val;
}

static unsigned _ack_resize_completed(unsigned sched_ctx, int worker)
{
	struct resize_ack *resize_ack = NULL;
	unsigned sender_sched_ctx = STARPU_NMAX_SCHED_CTXS;

	if(hypervisor.nsched_ctxs > 0)
	{
		int i;
		for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
		{
			if(hypervisor.sched_ctxs[i] != STARPU_NMAX_SCHED_CTXS)
			{
				struct sched_ctx_wrapper *sc_w = &hypervisor.sched_ctx_w[hypervisor.sched_ctxs[i]];
				if(sc_w->resize_ack.receiver_sched_ctx != -1 && 
				   sc_w->resize_ack.receiver_sched_ctx == sched_ctx) 
				{
					resize_ack = &sc_w->resize_ack;
					sender_sched_ctx = hypervisor.sched_ctxs[i];
					break;
				}
			}
		}
	}

	/* if there is no ctx waiting for its ack return 1*/
	if(resize_ack == NULL)
		return 1;
	else
	{
		int *moved_workers = resize_ack->moved_workers;
		int nmoved_workers = resize_ack->nmoved_workers;
		int *acked_workers = resize_ack->acked_workers;
		int i;
		
		if(worker != -1)
		{
			for(i = 0; i < nmoved_workers; i++)
			{
				int moved_worker = moved_workers[i];
				if(moved_worker == worker && acked_workers[i] == 0)
					acked_workers[i] = 1;
			}
		}
		
		int nacked_workers = 0;
		for(i = 0; i < nmoved_workers; i++)
		{
			nacked_workers += (acked_workers[i] == 1);
		}
		
		unsigned resize_completed = (nacked_workers == nmoved_workers);
		unsigned receiver_sched_ctx = resize_ack->receiver_sched_ctx;
		/* if the permission to resize is not allowed by the user don't do it
		   whatever the application says */
		if(resize_completed && !((hypervisor.resize[sched_ctx] == 0 || hypervisor.resize[receiver_sched_ctx] == 0) && imposed_resize) && worker == moved_workers[0])
		{				
			/* info concerning only the gflops_rate strateg */
			struct sched_ctx_wrapper *sender_sc_w = &hypervisor.sched_ctx_w[sender_sched_ctx];
			struct sched_ctx_wrapper *receiver_sc_w = &hypervisor.sched_ctx_w[receiver_sched_ctx];
			
			double start_time =  starpu_timing_now();
			sender_sc_w->start_time = start_time;
			sender_sc_w->remaining_flops = sender_sc_w->remaining_flops - sched_ctx_hypervisor_get_elapsed_flops_per_sched_ctx(sender_sc_w);
			_set_elapsed_flops_per_sched_ctx(sender_sched_ctx, 0.0);
			
			receiver_sc_w->start_time = start_time;
			receiver_sc_w->remaining_flops = receiver_sc_w->remaining_flops - sched_ctx_hypervisor_get_elapsed_flops_per_sched_ctx(receiver_sc_w);
			_set_elapsed_flops_per_sched_ctx(receiver_sched_ctx, 0.0);

			hypervisor.resize[sender_sched_ctx] = 1;
			hypervisor.resize[receiver_sched_ctx] = 1;
			/* if the user allowed resizing leave the decisions to the application */
			if(imposed_resize)  imposed_resize = 0;

			resize_ack->receiver_sched_ctx = -1;
			resize_ack->nmoved_workers = 0;
			free(resize_ack->moved_workers);
			free(resize_ack->acked_workers);
		}
		return resize_completed;
	}
	return 0;
}

void sched_ctx_hypervisor_resize(unsigned sched_ctx, int task_tag)
{
	_starpu_htbl_insert_32(&hypervisor.resize_requests[sched_ctx], (uint32_t)task_tag, (void*)sched_ctx);	
}

/* notifies the hypervisor that the worker is no longer idle and a new task was pushed on its queue */
static void notify_idle_end(unsigned sched_ctx, int worker)
{
	if(hypervisor.nsched_ctxs > 1)
	{

		if(hypervisor.resize[sched_ctx])
			hypervisor.sched_ctx_w[sched_ctx].current_idle_time[worker] = 0.0;
		
		if(hypervisor.policy.handle_idle_end)
			hypervisor.policy.handle_idle_end(sched_ctx, worker);
		
		_ack_resize_completed(sched_ctx, worker);
	}
}

/* notifies the hypervisor that the worker spent another cycle in idle time */
static void notify_idle_cycle(unsigned sched_ctx, int worker, double idle_time)
{
	if(hypervisor.nsched_ctxs > 1)
	{
		if(hypervisor.resize[sched_ctx])
		{
			struct sched_ctx_wrapper *sc_w = &hypervisor.sched_ctx_w[sched_ctx];
			sc_w->current_idle_time[worker] += idle_time;
			if(hypervisor.policy.handle_idle_cycle)
				hypervisor.policy.handle_idle_cycle(sched_ctx, worker);
		}		
		else 
			_ack_resize_completed(sched_ctx, worker);
	}
	return;
}

/* notifies the hypervisor that a new task was pushed on the queue of the worker */
static void notify_pushed_task(unsigned sched_ctx, int worker)
{	
	hypervisor.sched_ctx_w[sched_ctx].pushed_tasks[worker]++;
	if(hypervisor.sched_ctx_w[sched_ctx].total_flops != 0.0 && hypervisor.sched_ctx_w[sched_ctx].start_time == 0.0)
		hypervisor.sched_ctx_w[sched_ctx].start_time = starpu_timing_now();
	
	int ntasks = get_ntasks(hypervisor.sched_ctx_w[sched_ctx].pushed_tasks);
	
	if(!(hypervisor.resize[sched_ctx] == 0 && imposed_resize) && ntasks == hypervisor.min_tasks)
	{
		hypervisor.resize[sched_ctx] = 1;
		if(imposed_resize) imposed_resize = 0;
	}

	if(hypervisor.policy.handle_pushed_task)
		hypervisor.policy.handle_pushed_task(sched_ctx, worker);
}

/* notifies the hypervisor that a task was poped from the queue of the worker */
static void notify_poped_task(unsigned sched_ctx, int worker, double elapsed_flops)
{
	hypervisor.sched_ctx_w[sched_ctx].poped_tasks[worker]++;
	hypervisor.sched_ctx_w[sched_ctx].elapsed_flops[worker] += elapsed_flops;
	hypervisor.sched_ctx_w[sched_ctx].total_elapsed_flops[worker] += elapsed_flops;

	if(hypervisor.nsched_ctxs > 1)
	{
		if(hypervisor.resize[sched_ctx])
		{
			if(hypervisor.policy.handle_poped_task)
				hypervisor.policy.handle_poped_task(sched_ctx, worker);
		}
		else 
			_ack_resize_completed(sched_ctx, worker);
	}
}

/* notifies the hypervisor that a tagged task has just been executed */
static void notify_post_exec_hook(unsigned sched_ctx, int task_tag)
{
	STARPU_ASSERT(task_tag > 0);

	if(hypervisor.nsched_ctxs > 1)
	{
		unsigned conf_sched_ctx;
		int i;
		for(i = 0; i < hypervisor.nsched_ctxs; i++)
		{
			conf_sched_ctx = hypervisor.sched_ctxs[i];
			void *config = _starpu_htbl_search_32(hypervisor.configurations[conf_sched_ctx], (uint32_t)task_tag);
			if(config && config != hypervisor.configurations[conf_sched_ctx])
			{
				sched_ctx_hypervisor_set_config(conf_sched_ctx, config);
				free(config);
				_starpu_htbl_insert_32(&hypervisor.configurations[sched_ctx], (uint32_t)task_tag, NULL);
			}
		}	

		/* for the app driven we have to wait for the resize to be available
		   because the event is required to be executed at this specific moment */
		while(!_ack_resize_completed(sched_ctx, -1));

		if(hypervisor.resize[sched_ctx])
		{
			struct starpu_htbl32_node* resize_requests = hypervisor.resize_requests[sched_ctx];

			if(hypervisor.policy.handle_post_exec_hook)
				hypervisor.policy.handle_post_exec_hook(sched_ctx, resize_requests, task_tag);
		}
	}
}
struct sched_ctx_wrapper* sched_ctx_hypervisor_get_wrapper(unsigned sched_ctx)
{
	return &hypervisor.sched_ctx_w[sched_ctx];
}

int* sched_ctx_hypervisor_get_sched_ctxs()
{
	return hypervisor.sched_ctxs;
}

int sched_ctx_hypervisor_get_nsched_ctxs()
{
	return hypervisor.nsched_ctxs;
}
