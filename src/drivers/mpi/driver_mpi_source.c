/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015  Mathieu Lirzin <mthl@openmailbox.org>
 * Copyright (C) 2016  Inria
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


#include <mpi.h>
#include <errno.h>

#include <starpu.h>
#include <drivers/mpi/driver_mpi_source.h>
#include <drivers/mpi/driver_mpi_common.h>

#include <datawizard/memory_nodes.h>

#include <drivers/driver_common/driver_common.h>
#include <drivers/mp_common/source_common.h>

/* Mutex for concurrent access to the table.
 */
starpu_pthread_mutex_t htbl_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;

/* Structure used by host to store informations about a kernel executable on
 * a MPI MS device : its name, and its address on each device.
 * If a kernel has been initialized, then a lookup has already been achieved and the
 * device knows how to call it, else the host still needs to do a lookup.
 */
struct _starpu_mpi_ms_kernel
{
	UT_hash_handle hh;
	char *name;
	starpu_mpi_ms_kernel_t func[STARPU_MAXMPIDEVS];
} *kernels;


/* Array of structures containing all the informations useful to send
 * and receive informations with devices */
struct _starpu_mp_node *mpi_ms_nodes[STARPU_MAXMPIDEVS];

void _starpu_mpi_source_init(struct _starpu_mp_node *node)
{
    _starpu_mpi_common_mp_initialize_src_sink(node);
    //TODO
}

void _starpu_mpi_source_deinit(struct _starpu_mp_node *node)
{

}

struct _starpu_mp_node *_starpu_mpi_src_get_mp_node_from_memory_node(int memory_node)
{
    int devid = _starpu_memory_node_get_devid(memory_node);
    STARPU_ASSERT_MSG(devid >= 0 && devid < STARPU_MAXMPIDEVS, "bogus devid %d for memory node %d\n", devid, memory_node);

    return mpi_ms_nodes[devid];
}

int _starpu_mpi_src_allocate_memory(void ** addr, size_t size, unsigned memory_node)
{
    const struct _starpu_mp_node *mp_node = _starpu_mpi_src_get_mp_node_from_memory_node(memory_node);
    return _starpu_src_common_allocate(mp_node, addr, size);
}

void _starpu_mpi_source_free_memory(void *addr, unsigned memory_node)
{
	struct _starpu_mp_node *mp_node = _starpu_mpi_src_get_mp_node_from_memory_node(memory_node);
    _starpu_src_common_free(mp_node, addr);
}

 /* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
  * node to the address pointed by DST in the DST_NODE memory node
  */
int _starpu_mpi_copy_ram_to_mpi_sync(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size)
{
    struct _starpu_mp_node *mp_node = _starpu_mpi_src_get_mp_node_from_memory_node(dst_node);
    return _starpu_src_common_copy_host_to_sink_sync(mp_node, src, dst, size);
}   
 
 /* Transfert SIZE bytes from the address pointed by SRC in the SRC_NODE memory
  * node to the address pointed by DST in the DST_NODE memory node
  */    
int _starpu_mpi_copy_mpi_to_ram_sync(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size)
{
    struct _starpu_mp_node *mp_node = _starpu_mpi_src_get_mp_node_from_memory_node(src_node);
    return _starpu_src_common_copy_sink_to_host_sync(mp_node, src, dst, size);
}   

int _starpu_mpi_copy_sink_to_sink_sync(void *src, unsigned src_node, void *dst, unsigned dst_node, size_t size)
{
    return _starpu_src_common_copy_sink_to_sink_sync(_starpu_mpi_src_get_mp_node_from_memory_node(src_node),
            _starpu_mpi_src_get_mp_node_from_memory_node(dst_node),
            src, dst, size);
}

int _starpu_mpi_copy_mpi_to_ram_async(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size, void * event)
{
    struct _starpu_mp_node *mp_node = _starpu_mpi_src_get_mp_node_from_memory_node(src_node);
    return _starpu_src_common_copy_sink_to_host_async(mp_node, src, dst, size, event);
}

int _starpu_mpi_copy_ram_to_mpi_async(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size, void * event)
{
    struct _starpu_mp_node *mp_node = _starpu_mpi_src_get_mp_node_from_memory_node(dst_node);
    return _starpu_src_common_copy_host_to_sink_async(mp_node, src, dst, size, event);
}

int _starpu_mpi_copy_sink_to_sink_async(void *src, unsigned src_node, void *dst, unsigned dst_node, size_t size, void * event)
{
    return _starpu_src_common_copy_sink_to_sink_async(_starpu_mpi_src_get_mp_node_from_memory_node(src_node),
            _starpu_mpi_src_get_mp_node_from_memory_node(dst_node),
            src, dst, size, event);
}


int _starpu_mpi_ms_src_register_kernel(starpu_mpi_ms_func_symbol_t *symbol, const char *func_name)
{
	unsigned int func_name_size = (strlen(func_name) + 1) * sizeof(char);

	STARPU_PTHREAD_MUTEX_LOCK(&htbl_mutex);
	struct _starpu_mpi_ms_kernel *kernel;
	
	HASH_FIND_STR(kernels, func_name, kernel);

	if (kernel != NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);
		// Function already in the table.
		*symbol = kernel;
		return 0;
	}

	kernel = malloc(sizeof(*kernel));
	if (kernel == NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);
		return -ENOMEM;
	}

	kernel->name = malloc(func_name_size);
	if (kernel->name == NULL)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);
		free(kernel);
		return -ENOMEM;
	}

	memcpy(kernel->name, func_name, func_name_size);

	HASH_ADD_STR(kernels, name, kernel);

	unsigned int nb_mpi_devices = _starpu_mpi_src_get_device_count();
	unsigned int i;
	for (i = 0; i < nb_mpi_devices; ++i)
		kernel->func[i] = NULL;

	STARPU_PTHREAD_MUTEX_UNLOCK(&htbl_mutex);

	*symbol = kernel;

	return 0;
}


starpu_mpi_ms_kernel_t _starpu_mpi_ms_src_get_kernel(starpu_mpi_ms_func_symbol_t symbol)
{
	int workerid = starpu_worker_get_id();
	
	/* This function has to be called in the codelet only, by the thread
	 * which will handle the task */
	if (workerid < 0)
		return NULL;

	int devid = starpu_worker_get_devid(workerid);

	struct _starpu_mpi_ms_kernel *kernel = symbol;

	if (kernel->func[devid] == NULL)
	{
		struct _starpu_mp_node *node = mpi_ms_nodes[devid];
		int ret = _starpu_src_common_lookup(node, (void (**)(void))&kernel->func[devid], kernel->name);
		if (ret)
			return NULL;
	}

	return kernel->func[devid];
}

void(* _starpu_mpi_ms_src_get_kernel_from_job(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *j))(void)
{
	starpu_mpi_ms_kernel_t kernel = NULL;

	starpu_mpi_ms_func_t func = _starpu_task_get_mpi_ms_nth_implementation(j->task->cl, j->nimpl);
	if (func)
	{
		/* We execute the function contained in the codelet, it must return a
		 * pointer to the function to execute on the device, either specified
		 * directly by the user or by a call to starpu_mpi_ms_get_func().
		 */
		kernel = func();
	}
	else
	{
		/* If user dont define any starpu_mpi_ms_fun_t in cl->mpi_ms_func we try to use
		 * cpu_func_name.
		 */
		const char *func_name = _starpu_task_get_cpu_name_nth_implementation(j->task->cl, j->nimpl);
		if (func_name)
		{
			starpu_mpi_ms_func_symbol_t symbol;

			_starpu_mpi_ms_src_register_kernel(&symbol, func_name);

			kernel = _starpu_mpi_ms_src_get_kernel(symbol);
		}
	}
	STARPU_ASSERT(kernel);

	return (void (*)(void))kernel;
}



unsigned _starpu_mpi_src_get_device_count()
{
    int nb_mpi_devices;

    if (!_starpu_mpi_common_is_mp_initialized())
        return 0;
    
    MPI_Comm_size(MPI_COMM_WORLD, &nb_mpi_devices);

    //Remove one for master
    nb_mpi_devices = nb_mpi_devices - 1;

    return nb_mpi_devices;

}

void *_starpu_mpi_src_worker(void *arg)
{
#ifndef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
    struct _starpu_worker_set *worker_set_mpi = (struct _starpu_worker_set *) arg;
    int nbsinknodes = _starpu_mpi_src_get_device_count();

    int workersetnum;
    for (workersetnum = 0; workersetnum < nbsinknodes; workersetnum++)
    {
        struct _starpu_worker_set * worker_set = &worker_set_mpi[workersetnum];
#else
        struct _starpu_worker_set *worker_set = arg;
#endif

        /* As all workers of a set share common data, we just use the first
         *       * one for intializing the following stuffs. */
        struct _starpu_worker *baseworker = &worker_set->workers[0];
        struct _starpu_machine_config *config = baseworker->config;
        unsigned baseworkerid = baseworker - config->workers;
        unsigned devid = baseworker->devid;
        unsigned i;

        /* unsigned memnode = baseworker->memory_node; */

        _starpu_driver_start(baseworker, _STARPU_FUT_MPI_KEY, 0);

#ifdef STARPU_USE_FXT             
        for (i = 1; i < worker_set->nworkers; i++)
            _starpu_worker_start(&worker_set->workers[i], _STARPU_FUT_MPI_KEY, 0);
#endif          

        // Current task for a thread managing a worker set has no sense.
        _starpu_set_current_task(NULL);

        for (i = 0; i < config->topology.nmpicores[devid]; i++)
        {
            struct _starpu_worker *worker = &config->workers[baseworkerid+i];
            snprintf(worker->name, sizeof(worker->name), "MPI_MS %d core %u", devid, i);
            snprintf(worker->short_name, sizeof(worker->short_name), "MPI_MS %d.%u", devid, i);
        }

#ifndef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
        {
            char thread_name[16];
            snprintf(thread_name, sizeof(thread_name), "MPI_MS");
            starpu_pthread_setname(thread_name);
        }
#else
        {
            char thread_name[16];
            snprintf(thread_name, sizeof(thread_name), "MPI_MS %d", devid);
            starpu_pthread_setname(thread_name);
        }
#endif

        for (i = 0; i < worker_set->nworkers; i++)
        {
            struct _starpu_worker *worker = &worker_set->workers[i];
            _STARPU_TRACE_WORKER_INIT_END(worker->workerid);
        }
    
#ifndef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
        _starpu_src_common_init_switch_env(workersetnum);
    }  /* for */

    /* set the worker zero for the main thread */
    for (workersetnum = 0; workersetnum < nbsinknodes; workersetnum++)
    {
        struct _starpu_worker_set * worker_set = &worker_set_mpi[workersetnum];
        struct _starpu_worker *baseworker = &worker_set->workers[0];
#endif

        /* tell the main thread that this one is ready */
        STARPU_PTHREAD_MUTEX_LOCK(&worker_set->mutex);
        baseworker->status = STATUS_UNKNOWN;
        worker_set->set_is_initialized = 1;
        STARPU_PTHREAD_COND_SIGNAL(&worker_set->ready_cond);
        STARPU_PTHREAD_MUTEX_UNLOCK(&worker_set->mutex);

#ifndef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
    }
#endif


#ifndef STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD
    _starpu_src_common_workers_set(worker_set_mpi, nbsinknodes, mpi_ms_nodes);
#else
    _starpu_src_common_worker(worker_set, baseworkerid, mpi_ms_nodes[devid]);
#endif

    return NULL;

    
}
