/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2012-2015  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2013, 2015  CNRS
 * Copyright (C) 2016  INRIA
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

#ifndef __COPY_DRIVER_H__
#define __COPY_DRIVER_H__

#ifdef HAVE_AIO_H
#include <aio.h>
#endif

#include <common/config.h>
#include <common/list.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#endif

#ifdef STARPU_USE_OPENCL
#include <starpu_opencl.h>
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

struct _starpu_data_request;
struct _starpu_data_replicate;

#ifdef STARPU_USE_MIC
/* MIC needs memory_node to know which MIC is concerned.
 * mark is used to wait asynchronous request.
 * signal is used to test asynchronous request. */
struct _starpu_mic_async_event
{
	unsigned memory_node;
	int mark;
	uint64_t *signal;
};
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
LIST_TYPE(_starpu_mpi_ms_event_request,
        MPI_Request request;
);

struct _starpu_mpi_ms_async_event
{
    int is_sender;
    struct _starpu_mpi_ms_event_request_list * requests;
};
#endif

struct _starpu_disk_async_event
{
	unsigned memory_node;
	void *backend_event;
};

/* this is a structure that can be queried to see whether an asynchronous
 * transfer has terminated or not */
union _starpu_async_channel_event
{
#ifdef STARPU_SIMGRID
	struct
	{
		unsigned finished;
		starpu_pthread_mutex_t mutex;
		starpu_pthread_cond_t cond;
	};
#endif
#ifdef STARPU_USE_CUDA
	cudaEvent_t cuda_event;
#endif
#ifdef STARPU_USE_OPENCL
    cl_event opencl_event;
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
    struct _starpu_mpi_ms_async_event mpi_ms_event;
#endif
#ifdef STARPU_USE_MIC
	struct _starpu_mic_async_event mic_event;
#endif
	struct _starpu_disk_async_event disk_event;
};

struct _starpu_async_channel
{
	union _starpu_async_channel_event event;
	enum starpu_node_kind type;
    /* Which node to polling when needing ACK msg */
    struct _starpu_mp_node *polling_node;
    /* Used to know if the acknowlegdment msg is arrived from sinks */
    volatile int starpu_mp_common_finished_sender; 
    volatile int starpu_mp_common_finished_receiver; 
};

void _starpu_wake_all_blocked_workers_on_node(unsigned nodeid);

int _starpu_driver_copy_data_1_to_1(starpu_data_handle_t handle,
				    struct _starpu_data_replicate *src_replicate,
				    struct _starpu_data_replicate *dst_replicate,
				    unsigned donotread,
				    struct _starpu_data_request *req,
				    unsigned may_alloc,
				    unsigned prefetch);

unsigned _starpu_driver_test_request_completion(struct _starpu_async_channel *async_channel);
void _starpu_driver_wait_request_completion(struct _starpu_async_channel *async_channel);

#ifdef __cplusplus
}
#endif

#endif // __COPY_DRIVER_H__
