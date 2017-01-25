/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2016  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2014, 2015  CNRS
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

#include <common/config.h>
#include <core/sched_policy.h>
#include <datawizard/datastats.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>
#include <common/fxt.h>
#include "copy_driver.h"
#include "memalloc.h"

struct _starpu_memory_node_descr _starpu_descr;
starpu_pthread_key_t _starpu_memory_node_key STARPU_ATTRIBUTE_INTERNAL;

void _starpu_memory_nodes_init(void)
{
	/* there is no node yet, subsequent nodes will be
	 * added using _starpu_memory_node_register */
	_starpu_descr.nnodes = 0;

	STARPU_PTHREAD_KEY_CREATE(&_starpu_memory_node_key, NULL);

	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		_starpu_descr.nodes[i] = STARPU_UNUSED;
		_starpu_descr.nworkers[i] = 0;
	}

	_starpu_init_mem_chunk_lists();
	_starpu_init_data_request_lists();
	_starpu_memory_manager_init();

	STARPU_PTHREAD_RWLOCK_INIT(&_starpu_descr.conditions_rwlock, NULL);
	_starpu_descr.total_condition_count = 0;
}

void _starpu_memory_nodes_deinit(void)
{
	_starpu_deinit_data_request_lists();
	_starpu_deinit_mem_chunk_lists();

	STARPU_PTHREAD_RWLOCK_DESTROY(&_starpu_descr.conditions_rwlock);
	STARPU_PTHREAD_KEY_DELETE(_starpu_memory_node_key);
}

#undef starpu_node_get_kind
enum starpu_node_kind starpu_node_get_kind(unsigned node)
{
	return _starpu_node_get_kind(node);
}

#undef starpu_memory_nodes_get_count
unsigned starpu_memory_nodes_get_count(void)
{
	return _starpu_memory_nodes_get_count();
}

void _starpu_memory_node_get_name(unsigned node, char *name, int size)
{
	const char *prefix;
	switch (_starpu_descr.nodes[node])
	{
	case STARPU_CPU_RAM:
		prefix = "RAM";
		break;
	case STARPU_CUDA_RAM:
		prefix = "CUDA";
		break;
	case STARPU_OPENCL_RAM:
		prefix = "OpenCL";
		break;
	case STARPU_DISK_RAM:
		prefix = "Disk";
		break;
	case STARPU_MIC_RAM:
		prefix = "MIC";
		break;
	case STARPU_SCC_RAM:
		prefix = "SCC_RAM";
		break;
	case STARPU_SCC_SHM:
		prefix = "SCC_shared";
		break;
	case STARPU_UNUSED:
	default:
		prefix = "unknown";
		STARPU_ASSERT(0);
	}
	snprintf(name, size, "%s %u", prefix, _starpu_descr.devid[node]);
}

unsigned _starpu_memory_node_register(enum starpu_node_kind kind, int devid)
{
	unsigned node;
	/* ATOMIC_ADD returns the new value ... */
	node = STARPU_ATOMIC_ADD(&_starpu_descr.nnodes, 1) - 1;
	STARPU_ASSERT_MSG(node < STARPU_MAXNODES,"Too many nodes (%u) for maximum %u. Use configure option --enable-maxnodes=xxx to update the maximum number of nodes.", node, STARPU_MAXNODES);
	_starpu_descr.nodes[node] = kind;
	_STARPU_TRACE_NEW_MEM_NODE(node);

	_starpu_descr.devid[node] = devid;

	/* for now, there is no condition associated to that newly created node */
	_starpu_descr.condition_count[node] = 0;

	_starpu_malloc_init(node);

	return node;
}

/* TODO move in a more appropriate file  !! */
/* Register a condition variable associated to worker which is associated to a
 * memory node itself. */
void _starpu_memory_node_register_condition(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex, unsigned nodeid)
{
	unsigned cond_id;
	unsigned nconds_total, nconds;

	STARPU_PTHREAD_RWLOCK_WRLOCK(&_starpu_descr.conditions_rwlock);

	/* we only insert the queue if it's not already in the list */
	nconds = _starpu_descr.condition_count[nodeid];
	for (cond_id = 0; cond_id < nconds; cond_id++)
	{
		if (_starpu_descr.conditions_attached_to_node[nodeid][cond_id].cond == cond)
		{
			STARPU_ASSERT(_starpu_descr.conditions_attached_to_node[nodeid][cond_id].mutex == mutex);

			/* the condition is already in the list */
			STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_descr.conditions_rwlock);
			return;
		}
	}

	/* it was not found locally */
	_starpu_descr.conditions_attached_to_node[nodeid][cond_id].cond = cond;
	_starpu_descr.conditions_attached_to_node[nodeid][cond_id].mutex = mutex;
	_starpu_descr.condition_count[nodeid]++;

	/* do we have to add it in the global list as well ? */
	nconds_total = _starpu_descr.total_condition_count;
	for (cond_id = 0; cond_id < nconds_total; cond_id++)
	{
		if (_starpu_descr.conditions_all[cond_id].cond == cond)
		{
			/* the queue is already in the global list */
			STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_descr.conditions_rwlock);
			return;
		}
	}

	/* it was not in the global list either */
	_starpu_descr.conditions_all[nconds_total].cond = cond;
	_starpu_descr.conditions_all[nconds_total].mutex = mutex;
	_starpu_descr.total_condition_count++;

	STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_descr.conditions_rwlock);
}

#undef starpu_worker_get_memory_node
unsigned starpu_worker_get_memory_node(unsigned workerid)
{
	return _starpu_worker_get_memory_node(workerid);
}

