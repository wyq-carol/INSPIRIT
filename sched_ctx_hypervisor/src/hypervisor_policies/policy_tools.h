#include <sched_ctx_hypervisor.h>
#include <pthread.h>

#define HYPERVISOR_REDIM_SAMPLE 0.01
#define HYPERVISOR_START_REDIM_SAMPLE 0.005

struct bound_task_pool
{
	/* Which codelet has been executed */
	struct starpu_codelet *cl;
	/* Task footprint key */
	uint32_t footprint;
	/* Context the task belongs to */
	unsigned sched_ctx_id;
	/* Number of tasks of this kind */
	unsigned long n;
	/* Other task kinds */
	struct bound_task_pool *next;
};

unsigned _find_poor_sched_ctx(unsigned req_sched_ctx, int nworkers_to_move);

int* _get_first_workers(unsigned sched_ctx, unsigned *nworkers, enum starpu_archtype arch);

int* _get_first_workers_in_list(int *workers, int nall_workers,  unsigned *nworkers, enum starpu_archtype arch);

unsigned _get_potential_nworkers(struct policy_config *config, unsigned sched_ctx, enum starpu_archtype arch);

unsigned _get_nworkers_to_move(unsigned req_sched_ctx);

unsigned _resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, unsigned force_resize, unsigned now);

unsigned _resize_to_unknown_receiver(unsigned sender_sched_ctx, unsigned now);

double _get_ctx_velocity(struct sched_ctx_wrapper* sc_w);

double _get_velocity_per_worker_type(struct sched_ctx_wrapper* sc_w, enum starpu_archtype arch);

int _velocity_gap_btw_ctxs(void);

void _get_total_nw(int *workers, int nworkers, int ntypes_of_workers, double total_nw[ntypes_of_workers]);
