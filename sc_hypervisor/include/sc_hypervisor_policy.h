/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2012  INRIA
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

#ifndef SCHED_CTX_HYPERVISOR_POLICY_H
#define SCHED_CTX_HYPERVISOR_POLICY_H

#include <sc_hypervisor.h>

#ifdef __cplusplus
extern "C"
{
#endif


#define HYPERVISOR_REDIM_SAMPLE 0.02
#define HYPERVISOR_START_REDIM_SAMPLE 0.1

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

int* _get_first_workers(unsigned sched_ctx, int *nworkers, enum starpu_archtype arch);

int* _get_first_workers_in_list(int *start, int *workers, int nall_workers,  int *nworkers, enum starpu_archtype arch);

unsigned _get_potential_nworkers(struct sc_hypervisor_policy_config *config, unsigned sched_ctx, enum starpu_archtype arch);

int _get_nworkers_to_move(unsigned req_sched_ctx);

unsigned _resize(unsigned sender_sched_ctx, unsigned receiver_sched_ctx, unsigned force_resize, unsigned now);

unsigned _resize_to_unknown_receiver(unsigned sender_sched_ctx, unsigned now);

double _get_ctx_velocity(struct sc_hypervisor_wrapper* sc_w);

double _get_slowest_ctx_exec_time(void);

double _get_fastest_ctx_exec_time(void);

double _get_velocity_per_worker(struct sc_hypervisor_wrapper *sc_w, unsigned worker); 

double _get_velocity_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_archtype arch);

double _get_ref_velocity_per_worker_type(struct sc_hypervisor_wrapper* sc_w, enum starpu_archtype arch);

int _velocity_gap_btw_ctxs(void);

void _get_total_nw(int *workers, int nworkers, int ntypes_of_workers, int total_nw[ntypes_of_workers]);

#ifdef __cplusplus
}
#endif

#endif
