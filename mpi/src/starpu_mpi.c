/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2017  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017  CNRS
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

#include <stdlib.h>
#include <starpu_mpi.h>
#include <starpu_mpi_datatype.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_cache.h>
#include <starpu_profiling.h>
#include <starpu_mpi_stats.h>
#include <starpu_mpi_cache.h>
#include <starpu_mpi_sync_data.h>
#include <starpu_mpi_early_data.h>
#include <starpu_mpi_early_request.h>
#include <starpu_mpi_select_node.h>
#include <starpu_mpi_tag.h>
#include <starpu_mpi_comm.h>
#include <starpu_mpi_init.h>
#include <common/config.h>
#include <common/thread.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/coherency.h>
#include <core/simgrid.h>
#include <core/task.h>

/* Number of ready requests to process before polling for completed requests */
#define NREADY_PROCESS 10

static void _starpu_mpi_add_sync_point_in_fxt(void);
static void _starpu_mpi_submit_ready_request(void *arg);
static void _starpu_mpi_handle_ready_request(struct _starpu_mpi_req *req);
static void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req);
#ifdef STARPU_VERBOSE
static char *_starpu_mpi_request_type(enum _starpu_mpi_request_type request_type);
#endif
static struct _starpu_mpi_req *_starpu_mpi_isend_common(starpu_data_handle_t data_handle,
							int dest, int data_tag, MPI_Comm comm,
							unsigned detached, unsigned sync, void (*callback)(void *), void *arg,
							int sequential_consistency);
static struct _starpu_mpi_req *_starpu_mpi_irecv_common(starpu_data_handle_t data_handle,
							int source, int data_tag, MPI_Comm comm,
							unsigned detached, unsigned sync, void (*callback)(void *), void *arg,
							int sequential_consistency, int is_internal_req,
							starpu_ssize_t count);
static void _starpu_mpi_handle_detached_request(struct _starpu_mpi_req *req);
static void _starpu_mpi_early_data_cb(void* arg);

/* The list of ready requests */
static struct _starpu_mpi_req_list *ready_requests;

/* The list of detached requests that have already been submitted to MPI */
static struct _starpu_mpi_req_list *detached_requests;
static starpu_pthread_mutex_t detached_requests_mutex;

/* Condition to wake up progression thread */
static starpu_pthread_cond_t progress_cond;
/* Condition to wake up waiting for all current MPI requests to finish */
static starpu_pthread_cond_t barrier_cond;
static starpu_pthread_mutex_t progress_mutex;
#ifndef STARPU_SIMGRID
static starpu_pthread_t progress_thread;
#endif
static int running = 0;

#ifdef STARPU_SIMGRID
static int wait_counter;
static starpu_pthread_cond_t wait_counter_cond;
static starpu_pthread_mutex_t wait_counter_mutex;
#endif
int _starpu_mpi_fake_world_size = -1;
int _starpu_mpi_fake_world_rank = -1;

/* Count requests posted by the application and not yet submitted to MPI */
static starpu_pthread_mutex_t mutex_posted_requests;
static int posted_requests = 0, newer_requests, barrier_running = 0;

#define _STARPU_MPI_INC_POSTED_REQUESTS(value) { STARPU_PTHREAD_MUTEX_LOCK(&mutex_posted_requests); posted_requests += value; STARPU_PTHREAD_MUTEX_UNLOCK(&mutex_posted_requests); }

#pragma weak smpi_simulated_main_
extern int smpi_simulated_main_(int argc, char *argv[]);

#ifdef HAVE_SMPI_PROCESS_SET_USER_DATA
#if !HAVE_DECL_SMPI_PROCESS_SET_USER_DATA
extern void smpi_process_set_user_data(void *);
#endif
#endif

static void _starpu_mpi_request_init(struct _starpu_mpi_req **req)
{
	_STARPU_MPI_CALLOC(*req, 1, sizeof(struct _starpu_mpi_req));

	/* Initialize the request structure */
	(*req)->data_handle = NULL;

	(*req)->datatype = 0;
	(*req)->datatype_name = NULL;
	(*req)->ptr = NULL;
	(*req)->count = -1;
	(*req)->registered_datatype = -1;

	(*req)->node_tag.rank = -1;
	(*req)->node_tag.data_tag = -1;
	(*req)->node_tag.comm = 0;

	(*req)->func = NULL;

	(*req)->status = NULL;
	(*req)->data_request = 0;
	(*req)->flag = NULL;

	(*req)->ret = -1;
	STARPU_PTHREAD_MUTEX_INIT(&((*req)->req_mutex), NULL);
	STARPU_PTHREAD_COND_INIT(&((*req)->req_cond), NULL);
	STARPU_PTHREAD_MUTEX_INIT(&((*req)->posted_mutex), NULL);
	STARPU_PTHREAD_COND_INIT(&((*req)->posted_cond), NULL);

	(*req)->request_type = UNKNOWN_REQ;

	(*req)->submitted = 0;
	(*req)->completed = 0;
	(*req)->posted = 0;

	(*req)->other_request = NULL;

	(*req)->sync = 0;
	(*req)->detached = -1;
	(*req)->callback = NULL;
	(*req)->callback_arg = NULL;

	(*req)->size_req = 0;
	(*req)->internal_req = NULL;
	(*req)->is_internal_req = 0;
	(*req)->early_data_handle = NULL;
	(*req)->envelope = NULL;
	(*req)->sequential_consistency = 1;

#ifdef STARPU_SIMGRID
	starpu_pthread_queue_init(&((*req)->queue));
	starpu_pthread_queue_register(&wait, &((*req)->queue));
	(*req)->done = 0;
#endif
}

static void _starpu_mpi_request_destroy(struct _starpu_mpi_req *req)
{
	STARPU_PTHREAD_MUTEX_DESTROY(&req->req_mutex);
	STARPU_PTHREAD_COND_DESTROY(&req->req_cond);
	STARPU_PTHREAD_MUTEX_DESTROY(&req->posted_mutex);
	STARPU_PTHREAD_COND_DESTROY(&req->posted_cond);
	free(req->datatype_name);
	req->datatype_name = NULL;
#ifdef STARPU_SIMGRID
	starpu_pthread_queue_unregister(&wait, &req->queue);
	starpu_pthread_queue_destroy(&req->queue);
#endif
	free(req);
	req = NULL;
}

 /********************************************************/
 /*                                                      */
 /*  Send/Receive functionalities                        */
 /*                                                      */
 /********************************************************/

struct _starpu_mpi_early_data_cb_args
{
	starpu_data_handle_t data_handle;
	starpu_data_handle_t early_handle;
	struct _starpu_mpi_req *req;
	void *buffer;
};

static void _starpu_mpi_submit_ready_request(void *arg)
{
	_STARPU_MPI_LOG_IN();
	struct _starpu_mpi_req *req = arg;

	_STARPU_MPI_INC_POSTED_REQUESTS(-1);

	_STARPU_MPI_DEBUG(3, "new req %p srcdst %d tag %d and type %s %d\n", req, req->node_tag.rank, req->node_tag.data_tag, _starpu_mpi_request_type(req->request_type), req->is_internal_req);

	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

	if (req->request_type == RECV_REQ)
	{
		/* Case : the request is the internal receive request submitted
		 * by StarPU-MPI to receive incoming data without a matching
		 * early_request from the application. We immediately allocate the
		 * pointer associated to the data_handle, and push it into the
		 * ready_requests list, so as the real MPI request can be submitted
		 * before the next submission of the envelope-catching request. */
		if (req->is_internal_req)
		{
			_starpu_mpi_datatype_allocate(req->data_handle, req);
			if (req->registered_datatype == 1)
			{
				req->count = 1;
				req->ptr = starpu_data_get_local_ptr(req->data_handle);
			}
			else
			{
				STARPU_ASSERT(req->count);
				_STARPU_MPI_MALLOC(req->ptr, req->count);
			}

			_STARPU_MPI_DEBUG(3, "Pushing internal starpu_mpi_irecv request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
					  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle, req->ptr,
					  req->datatype_name, (int)req->count, req->registered_datatype);
			_starpu_mpi_req_list_push_front(ready_requests, req);

			/* inform the starpu mpi thread that the request has been pushed in the ready_requests list */
			STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
			STARPU_PTHREAD_MUTEX_LOCK(&req->posted_mutex);
			req->posted = 1;
			STARPU_PTHREAD_COND_BROADCAST(&req->posted_cond);
			STARPU_PTHREAD_MUTEX_UNLOCK(&req->posted_mutex);
			STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
		}
		else
		{
			/* test whether some data with the given tag and source have already been received by StarPU-MPI*/
			struct _starpu_mpi_early_data_handle *early_data_handle = _starpu_mpi_early_data_find(&req->node_tag);

			/* Case: a receive request for a data with the given tag and source has already been
			 * posted by StarPU. Asynchronously requests a Read permission over the temporary handle ,
			 * so as when the internal receive is completed, the _starpu_mpi_early_data_cb function
			 * will be called to bring the data back to the original data handle associated to the request.*/
			if (early_data_handle)
			{
				STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
				STARPU_PTHREAD_MUTEX_LOCK(&(early_data_handle->req_mutex));
				while (!(early_data_handle->req_ready))
					STARPU_PTHREAD_COND_WAIT(&(early_data_handle->req_cond), &(early_data_handle->req_mutex));
				STARPU_PTHREAD_MUTEX_UNLOCK(&(early_data_handle->req_mutex));
				STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

				_STARPU_MPI_DEBUG(3, "The RECV request %p with tag %d has already been received, copying previously received data into handle's pointer..\n", req, req->node_tag.data_tag);
				STARPU_ASSERT(req->data_handle != early_data_handle->handle);

				req->internal_req = early_data_handle->req;
				req->early_data_handle = early_data_handle;

				struct _starpu_mpi_early_data_cb_args *cb_args;
				_STARPU_MPI_MALLOC(cb_args, sizeof(struct _starpu_mpi_early_data_cb_args));
				cb_args->data_handle = req->data_handle;
				cb_args->early_handle = early_data_handle->handle;
				cb_args->buffer = early_data_handle->buffer;
				cb_args->req = req;

				_STARPU_MPI_DEBUG(3, "Calling data_acquire_cb on starpu_mpi_copy_cb..\n");
				STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
				starpu_data_acquire_cb(early_data_handle->handle,STARPU_R,_starpu_mpi_early_data_cb,(void*) cb_args);
				STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
			}
			/* Case: no matching data has been received. Store the receive request as an early_request. */
			else
			{
				struct _starpu_mpi_req *sync_req = _starpu_mpi_sync_data_find(req->node_tag.data_tag, req->node_tag.rank, req->node_tag.comm);
				_STARPU_MPI_DEBUG(3, "----------> Looking for sync data for tag %d and src %d = %p\n", req->node_tag.data_tag, req->node_tag.rank, sync_req);
				if (sync_req)
				{
					req->sync = 1;
					_starpu_mpi_datatype_allocate(req->data_handle, req);
					if (req->registered_datatype == 1)
					{
						req->count = 1;
						req->ptr = starpu_data_get_local_ptr(req->data_handle);
					}
					else
					{
						req->count = sync_req->count;
						STARPU_ASSERT(req->count);
						_STARPU_MPI_MALLOC(req->ptr, req->count);
					}
					_starpu_mpi_req_list_push_front(ready_requests, req);
					_starpu_mpi_request_destroy(sync_req);
				}
				else
				{
					_STARPU_MPI_DEBUG(3, "Adding the pending receive request %p (srcdst %d tag %d) into the request hashmap\n", req, req->node_tag.rank, req->node_tag.data_tag);
					_starpu_mpi_early_request_enqueue(req);
				}
			}
		}
	}
	else
	{
		_starpu_mpi_req_list_push_front(ready_requests, req);
		_STARPU_MPI_DEBUG(3, "Pushing new request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
				  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle, req->ptr,
				  req->datatype_name, (int)req->count, req->registered_datatype);
	}

	newer_requests = 1;
	STARPU_PTHREAD_COND_BROADCAST(&progress_cond);
#ifdef STARPU_SIMGRID
	starpu_pthread_queue_signal(&dontsleep);
#endif
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
	_STARPU_MPI_LOG_OUT();
}

static void nop_acquire_cb(void *arg)
{
	starpu_data_release(arg);
}

static struct _starpu_mpi_req *_starpu_mpi_isend_irecv_common(starpu_data_handle_t data_handle,
							      int srcdst, int data_tag, MPI_Comm comm,
							      unsigned detached, unsigned sync, void (*callback)(void *), void *arg,
							      enum _starpu_mpi_request_type request_type, void (*func)(struct _starpu_mpi_req *),
							      enum starpu_data_access_mode mode,
							      int sequential_consistency,
							      int is_internal_req,
							      starpu_ssize_t count)
{
	struct _starpu_mpi_req *req;

	if (_starpu_mpi_fake_world_size != -1)
	{
		starpu_data_acquire_cb_sequential_consistency(data_handle, mode, nop_acquire_cb, data_handle, sequential_consistency);
		return NULL;
	}

	_STARPU_MPI_LOG_IN();
	_STARPU_MPI_INC_POSTED_REQUESTS(1);

	_starpu_mpi_comm_register(comm);

	/* Initialize the request structure */
	_starpu_mpi_request_init(&req);
	req->request_type = request_type;
	req->data_handle = data_handle;
	req->node_tag.rank = srcdst;
	req->node_tag.data_tag = data_tag;
	req->node_tag.comm = comm;
	req->detached = detached;
	req->sync = sync;
	req->callback = callback;
	req->callback_arg = arg;
	req->func = func;
	req->sequential_consistency = sequential_consistency;
	req->is_internal_req = is_internal_req;
	req->count = count;

	/* Asynchronously request StarPU to fetch the data in main memory: when
	 * it is available in main memory, _starpu_mpi_submit_ready_request(req) is called and
	 * the request is actually submitted */
	starpu_data_acquire_cb_sequential_consistency(data_handle, mode, _starpu_mpi_submit_ready_request, (void *)req, sequential_consistency);

	_STARPU_MPI_LOG_OUT();
	return req;
 }

#ifdef STARPU_SIMGRID
int _starpu_mpi_simgrid_mpi_test(unsigned *done, int *flag)
{
	*flag = 0;
	if (*done)
	{
		starpu_pthread_queue_signal(&dontsleep);
		*flag = 1;
	}
	return MPI_SUCCESS;
}

static void _starpu_mpi_simgrid_wait_req_func(void* arg)
{
	struct _starpu_simgrid_mpi_req *sim_req = arg;
	int ret;
	STARPU_PTHREAD_MUTEX_LOCK(&wait_counter_mutex);
	wait_counter++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&wait_counter_mutex);

	ret = MPI_Wait(sim_req->request, sim_req->status);

	STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Wait returning %s", _starpu_mpi_get_mpi_error_code(ret));

	*(sim_req->done) = 1;
	starpu_pthread_queue_signal(sim_req->queue);

	free(sim_req);

	STARPU_PTHREAD_MUTEX_LOCK(&wait_counter_mutex);
	if (--wait_counter == 0)
		STARPU_PTHREAD_COND_SIGNAL(&wait_counter_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&wait_counter_mutex);
}

void _starpu_mpi_simgrid_wait_req(MPI_Request *request, MPI_Status *status, starpu_pthread_queue_t *queue, unsigned *done)
{
	struct _starpu_simgrid_mpi_req *sim_req;
	_STARPU_MPI_CALLOC(sim_req, 1, sizeof(struct _starpu_simgrid_mpi_req));
	sim_req->request = request;
	sim_req->status = status;
	sim_req->queue = queue;
	sim_req->done = done;
	*done = 0;

	_starpu_simgrid_xbt_thread_create("wait for mpi transfer", _starpu_mpi_simgrid_wait_req_func, sim_req);
}
#endif

 /********************************************************/
 /*                                                      */
 /*  Send functionalities                                */
 /*                                                      */
 /********************************************************/

static void _starpu_mpi_isend_data_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(30, "post MPI isend request %p type %s tag %d src %d data %p datasize %ld ptr %p datatype '%s' count %d registered_datatype %d sync %d\n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle, starpu_data_get_size(req->data_handle), req->ptr, req->datatype_name, (int)req->count, req->registered_datatype, req->sync);

	_starpu_mpi_comm_amounts_inc(req->node_tag.comm, req->node_tag.rank, req->datatype, req->count);

	_STARPU_MPI_TRACE_ISEND_SUBMIT_BEGIN(req->node_tag.rank, req->node_tag.data_tag, 0);

	if (req->sync == 0)
	{
		_STARPU_MPI_COMM_TO_DEBUG(req->count, req->datatype, req->node_tag.rank, _STARPU_MPI_TAG_DATA, req->node_tag.data_tag, req->node_tag.comm);
		req->ret = MPI_Isend(req->ptr, req->count, req->datatype, req->node_tag.rank, _STARPU_MPI_TAG_DATA, req->node_tag.comm, &req->data_request);
		STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Isend returning %s", _starpu_mpi_get_mpi_error_code(req->ret));
	}
	else
	{
		_STARPU_MPI_COMM_TO_DEBUG(req->count, req->datatype, req->node_tag.rank, _STARPU_MPI_TAG_SYNC_DATA, req->node_tag.data_tag, req->node_tag.comm);
		req->ret = MPI_Issend(req->ptr, req->count, req->datatype, req->node_tag.rank, _STARPU_MPI_TAG_SYNC_DATA, req->node_tag.comm, &req->data_request);
		STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Issend returning %s", _starpu_mpi_get_mpi_error_code(req->ret));
	}

#ifdef STARPU_SIMGRID
	_starpu_mpi_simgrid_wait_req(&req->data_request, &req->status_store, &req->queue, &req->done);
#endif

	_STARPU_MPI_TRACE_ISEND_SUBMIT_END(req->node_tag.rank, req->node_tag.data_tag, starpu_data_get_size(req->data_handle));

	/* somebody is perhaps waiting for the MPI request to be posted */
	STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	req->submitted = 1;
	STARPU_PTHREAD_COND_BROADCAST(&req->req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	_starpu_mpi_handle_detached_request(req);

	_STARPU_MPI_LOG_OUT();
}

static void _starpu_mpi_isend_size_func(struct _starpu_mpi_req *req)
{
	_starpu_mpi_datatype_allocate(req->data_handle, req);

	_STARPU_MPI_CALLOC(req->envelope, 1,sizeof(struct _starpu_mpi_envelope));
	req->envelope->mode = _STARPU_MPI_ENVELOPE_DATA;
	req->envelope->data_tag = req->node_tag.data_tag;
	req->envelope->sync = req->sync;

	if (req->registered_datatype == 1)
	{
		int size;
		req->count = 1;
		req->ptr = starpu_data_get_local_ptr(req->data_handle);

		MPI_Type_size(req->datatype, &size);
		req->envelope->size = (starpu_ssize_t)req->count * size;
		_STARPU_MPI_DEBUG(20, "Post MPI isend count (%ld) datatype_size %ld request to %d\n",req->count,starpu_data_get_size(req->data_handle), req->node_tag.rank);
		_STARPU_MPI_COMM_TO_DEBUG(sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.rank, _STARPU_MPI_TAG_ENVELOPE, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.comm);
		MPI_Isend(req->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.rank, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.comm, &req->size_req);
	}
	else
	{
		int ret;

 		// Do not pack the data, just try to find out the size
		starpu_data_pack(req->data_handle, NULL, &(req->envelope->size));

		if (req->envelope->size != -1)
 		{
 			// We already know the size of the data, let's send it to overlap with the packing of the data
			_STARPU_MPI_DEBUG(20, "Sending size %ld (%ld %s) to node %d (first call to pack)\n", req->envelope->size, sizeof(req->count), "MPI_BYTE", req->node_tag.rank);
			req->count = req->envelope->size;
			_STARPU_MPI_COMM_TO_DEBUG(sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.rank, _STARPU_MPI_TAG_ENVELOPE, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.comm);
			ret = MPI_Isend(req->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.rank, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.comm, &req->size_req);
			STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "when sending size, MPI_Isend returning %s", _starpu_mpi_get_mpi_error_code(ret));
 		}

 		// Pack the data
 		starpu_data_pack(req->data_handle, &req->ptr, &req->count);
		if (req->envelope->size == -1)
 		{
 			// We know the size now, let's send it
			_STARPU_MPI_DEBUG(20, "Sending size %ld (%ld %s) to node %d (second call to pack)\n", req->envelope->size, sizeof(req->count), "MPI_BYTE", req->node_tag.rank);
			_STARPU_MPI_COMM_TO_DEBUG(sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.rank, _STARPU_MPI_TAG_ENVELOPE, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.comm);
			ret = MPI_Isend(req->envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.rank, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.comm, &req->size_req);
			STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "when sending size, MPI_Isend returning %s", _starpu_mpi_get_mpi_error_code(ret));
 		}
 		else
 		{
 			// We check the size returned with the 2 calls to pack is the same
			STARPU_MPI_ASSERT_MSG(req->count == req->envelope->size, "Calls to pack_data returned different sizes %ld != %ld", req->count, req->envelope->size);
 		}
		// We can send the data now
	}

	if (req->sync)
	{
		// If the data is to be sent in synchronous mode, we need to wait for the receiver ready message
		_starpu_mpi_sync_data_add(req);
	}
	else
	{
		// Otherwise we can send the data
		_starpu_mpi_isend_data_func(req);
	}
}

static struct _starpu_mpi_req *_starpu_mpi_isend_common(starpu_data_handle_t data_handle,
							int dest, int data_tag, MPI_Comm comm,
							unsigned detached, unsigned sync, void (*callback)(void *), void *arg,
							int sequential_consistency)
{
	return _starpu_mpi_isend_irecv_common(data_handle, dest, data_tag, comm, detached, sync, callback, arg, SEND_REQ, _starpu_mpi_isend_size_func,
#ifdef STARPU_MPI_PEDANTIC_ISEND
					      STARPU_RW,
#else
					      STARPU_R,
#endif
					      sequential_consistency, 0, 0);
}

int starpu_mpi_isend(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, int data_tag, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_isend needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req;
	_STARPU_MPI_TRACE_ISEND_COMPLETE_BEGIN(dest, data_tag, 0);
	req = _starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 0, 0, NULL, NULL, 1);
	_STARPU_MPI_TRACE_ISEND_COMPLETE_END(dest, data_tag, 0);

	STARPU_MPI_ASSERT_MSG(req, "Invalid return for _starpu_mpi_isend_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_isend_detached(starpu_data_handle_t data_handle,
			      int dest, int data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();
	_starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 1, 0, callback, arg, 1);
	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_send(starpu_data_handle_t data_handle, int dest, int data_tag, MPI_Comm comm)
{
	starpu_mpi_req req;
	MPI_Status status;

	_STARPU_MPI_LOG_IN();
	memset(&status, 0, sizeof(MPI_Status));

	starpu_mpi_isend(data_handle, &req, dest, data_tag, comm);
	starpu_mpi_wait(&req, &status);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_issend(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int dest, int data_tag, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_issend needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req;
	req = _starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 0, 1, NULL, NULL, 1);

	STARPU_MPI_ASSERT_MSG(req, "Invalid return for _starpu_mpi_isend_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_issend_detached(starpu_data_handle_t data_handle, int dest, int data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();

	_starpu_mpi_isend_common(data_handle, dest, data_tag, comm, 1, 1, callback, arg, 1);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

/********************************************************/
/*                                                      */
/*  receive functionalities                             */
/*                                                      */
/********************************************************/

static void _starpu_mpi_irecv_data_func(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(20, "post MPI irecv request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n", req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle, req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);

	_STARPU_MPI_TRACE_IRECV_SUBMIT_BEGIN(req->node_tag.rank, req->node_tag.data_tag);

	if (req->sync)
	{
		struct _starpu_mpi_envelope *_envelope;
		_STARPU_MPI_CALLOC(_envelope, 1, sizeof(struct _starpu_mpi_envelope));
		_envelope->mode = _STARPU_MPI_ENVELOPE_SYNC_READY;
		_envelope->data_tag = req->node_tag.data_tag;
		_STARPU_MPI_DEBUG(20, "Telling node %d it can send the data and waiting for the data back ...\n", req->node_tag.rank);
		_STARPU_MPI_COMM_TO_DEBUG(sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.rank, _STARPU_MPI_TAG_ENVELOPE, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.comm);
		req->ret = MPI_Send(_envelope, sizeof(struct _starpu_mpi_envelope), MPI_BYTE, req->node_tag.rank, _STARPU_MPI_TAG_ENVELOPE, req->node_tag.comm);
		STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Send returning %s", _starpu_mpi_get_mpi_error_code(req->ret));
		free(_envelope);
		_envelope = NULL;
	}

	if (req->sync)
	{
		_STARPU_MPI_COMM_FROM_DEBUG(req->count, req->datatype, req->node_tag.rank, _STARPU_MPI_TAG_SYNC_DATA, req->node_tag.data_tag, req->node_tag.comm);
		req->ret = MPI_Irecv(req->ptr, req->count, req->datatype, req->node_tag.rank, _STARPU_MPI_TAG_SYNC_DATA, req->node_tag.comm, &req->data_request);
	}
	else
	{
		_STARPU_MPI_COMM_FROM_DEBUG(req->count, req->datatype, req->node_tag.rank, _STARPU_MPI_TAG_DATA, req->node_tag.data_tag, req->node_tag.comm);
		req->ret = MPI_Irecv(req->ptr, req->count, req->datatype, req->node_tag.rank, _STARPU_MPI_TAG_DATA, req->node_tag.comm, &req->data_request);
#ifdef STARPU_SIMGRID
		_starpu_mpi_simgrid_wait_req(&req->data_request, &req->status_store, &req->queue, &req->done);
#endif
	}
	STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_IRecv returning %s", _starpu_mpi_get_mpi_error_code(req->ret));

	_STARPU_MPI_TRACE_IRECV_SUBMIT_END(req->node_tag.rank, req->node_tag.data_tag);

	/* somebody is perhaps waiting for the MPI request to be posted */
	STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	req->submitted = 1;
	STARPU_PTHREAD_COND_BROADCAST(&req->req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	_starpu_mpi_handle_detached_request(req);

	_STARPU_MPI_LOG_OUT();
}

static struct _starpu_mpi_req *_starpu_mpi_irecv_common(starpu_data_handle_t data_handle, int source, int data_tag, MPI_Comm comm, unsigned detached, unsigned sync, void (*callback)(void *), void *arg, int sequential_consistency, int is_internal_req, starpu_ssize_t count)
{
	return _starpu_mpi_isend_irecv_common(data_handle, source, data_tag, comm, detached, sync, callback, arg, RECV_REQ, _starpu_mpi_irecv_data_func, STARPU_W, sequential_consistency, is_internal_req, count);
}

int starpu_mpi_irecv(starpu_data_handle_t data_handle, starpu_mpi_req *public_req, int source, int data_tag, MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_irecv needs a valid starpu_mpi_req");

//	// We check if a tag is defined for the data handle, if not,
//	// we define the one given for the communication.
//	// A tag is necessary for the internal mpi engine.
//	int tag = starpu_data_get_tag(data_handle);
//	if (tag == -1)
//		starpu_data_set_tag(data_handle, data_tag);

	struct _starpu_mpi_req *req;
	_STARPU_MPI_TRACE_IRECV_COMPLETE_BEGIN(source, data_tag);
	req = _starpu_mpi_irecv_common(data_handle, source, data_tag, comm, 0, 0, NULL, NULL, 1, 0, 0);
	_STARPU_MPI_TRACE_IRECV_COMPLETE_END(source, data_tag);
	STARPU_MPI_ASSERT_MSG(req, "Invalid return for _starpu_mpi_irecv_common");
	*public_req = req;

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_irecv_detached(starpu_data_handle_t data_handle, int source, int data_tag, MPI_Comm comm, void (*callback)(void *), void *arg)
{
	_STARPU_MPI_LOG_IN();

//	// We check if a tag is defined for the data handle, if not,
//	// we define the one given for the communication.
//	// A tag is necessary for the internal mpi engine.
//	int tag = starpu_data_get_tag(data_handle);
//	if (tag == -1)
//		starpu_data_set_tag(data_handle, data_tag);

	_starpu_mpi_irecv_common(data_handle, source, data_tag, comm, 1, 0, callback, arg, 1, 0, 0);
	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_irecv_detached_sequential_consistency(starpu_data_handle_t data_handle, int source, int data_tag, MPI_Comm comm, void (*callback)(void *), void *arg, int sequential_consistency)
{
	_STARPU_MPI_LOG_IN();

//	// We check if a tag is defined for the data handle, if not,
//	// we define the one given for the communication.
//	// A tag is necessary for the internal mpi engine.
//	int tag = starpu_data_get_tag(data_handle);
//	if (tag == -1)
//		starpu_data_set_tag(data_handle, data_tag);

	_starpu_mpi_irecv_common(data_handle, source, data_tag, comm, 1, 0, callback, arg, sequential_consistency, 0, 0);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

int starpu_mpi_recv(starpu_data_handle_t data_handle, int source, int data_tag, MPI_Comm comm, MPI_Status *status)
{
	starpu_mpi_req req;
	_STARPU_MPI_LOG_IN();

//	// We check if a tag is defined for the data handle, if not,
//	// we define the one given for the communication.
//	// A tag is necessary for the internal mpi engine.
//	int tag = starpu_data_get_tag(data_handle);
//	if (tag == -1)
//		starpu_data_set_tag(data_handle, data_tag);

	starpu_mpi_irecv(data_handle, &req, source, data_tag, comm);
	starpu_mpi_wait(&req, status);

	_STARPU_MPI_LOG_OUT();
	return 0;
}

/********************************************************/
/*                                                      */
/*  Wait functionalities                                */
/*                                                      */
/********************************************************/

static void _starpu_mpi_wait_func(struct _starpu_mpi_req *waiting_req)
{
	_STARPU_MPI_LOG_IN();
	/* Which is the mpi request we are waiting for ? */
	struct _starpu_mpi_req *req = waiting_req->other_request;

	_STARPU_MPI_TRACE_UWAIT_BEGIN(req->node_tag.rank, req->node_tag.data_tag);
	if (req->data_request != MPI_REQUEST_NULL)
	{
		// TODO: Fix for STARPU_SIMGRID
#ifdef STARPU_SIMGRID
		STARPU_MPI_ASSERT_MSG(0, "Implement this in STARPU_SIMGRID");
#endif
		req->ret = MPI_Wait(&req->data_request, waiting_req->status);
		STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Wait returning %s", _starpu_mpi_get_mpi_error_code(req->ret));
	}
	_STARPU_MPI_TRACE_UWAIT_END(req->node_tag.rank, req->node_tag.data_tag);

	_starpu_mpi_handle_request_termination(req);

	_STARPU_MPI_LOG_OUT();
}

int starpu_mpi_wait(starpu_mpi_req *public_req, MPI_Status *status)
{
	int ret;
	struct _starpu_mpi_req *req = *public_req;
	struct _starpu_mpi_req *waiting_req;

	_STARPU_MPI_LOG_IN();
	_STARPU_MPI_INC_POSTED_REQUESTS(1);

	/* We cannot try to complete a MPI request that was not actually posted
	 * to MPI yet. */
	STARPU_PTHREAD_MUTEX_LOCK(&(req->req_mutex));
	while (!(req->submitted))
		STARPU_PTHREAD_COND_WAIT(&(req->req_cond), &(req->req_mutex));
	STARPU_PTHREAD_MUTEX_UNLOCK(&(req->req_mutex));

	/* Initialize the request structure */
	 _starpu_mpi_request_init(&waiting_req);
	waiting_req->status = status;
	waiting_req->other_request = req;
	waiting_req->func = _starpu_mpi_wait_func;
	waiting_req->request_type = WAIT_REQ;

	_starpu_mpi_submit_ready_request(waiting_req);

	/* We wait for the MPI request to finish */
	STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	while (!req->completed)
		STARPU_PTHREAD_COND_WAIT(&req->req_cond, &req->req_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	ret = req->ret;

	/* The internal request structure was automatically allocated */
	*public_req = NULL;
	if (req->internal_req)
	{
		_starpu_mpi_request_destroy(req->internal_req);
	}
	_starpu_mpi_request_destroy(req);
	_starpu_mpi_request_destroy(waiting_req);

	_STARPU_MPI_LOG_OUT();
	return ret;
}

/********************************************************/
/*                                                      */
/*  Test functionalities                                */
/*                                                      */
/********************************************************/

static void _starpu_mpi_test_func(struct _starpu_mpi_req *testing_req)
{
	_STARPU_MPI_LOG_IN();
	/* Which is the mpi request we are testing for ? */
	struct _starpu_mpi_req *req = testing_req->other_request;

	_STARPU_MPI_DEBUG(2, "Test request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle, req->ptr,
			  req->datatype_name, (int)req->count, req->registered_datatype);

	_STARPU_MPI_TRACE_UTESTING_BEGIN(req->node_tag.rank, req->node_tag.data_tag);

#ifdef STARPU_SIMGRID
	req->ret = _starpu_mpi_simgrid_mpi_test(&req->done, testing_req->flag);
	memcpy(testing_req->status, &req->status_store, sizeof(*testing_req->status));
#else
	req->ret = MPI_Test(&req->data_request, testing_req->flag, testing_req->status);
#endif

	STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Test returning %s", _starpu_mpi_get_mpi_error_code(req->ret));

	_STARPU_MPI_TRACE_UTESTING_END(req->node_tag.rank, req->node_tag.data_tag);

	if (*testing_req->flag)
	{
		testing_req->ret = req->ret;
		_starpu_mpi_handle_request_termination(req);
	}

	STARPU_PTHREAD_MUTEX_LOCK(&testing_req->req_mutex);
	testing_req->completed = 1;
	STARPU_PTHREAD_COND_SIGNAL(&testing_req->req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&testing_req->req_mutex);
	_STARPU_MPI_LOG_OUT();
}

int starpu_mpi_test(starpu_mpi_req *public_req, int *flag, MPI_Status *status)
{
	_STARPU_MPI_LOG_IN();
	int ret = 0;

	STARPU_MPI_ASSERT_MSG(public_req, "starpu_mpi_test needs a valid starpu_mpi_req");

	struct _starpu_mpi_req *req = *public_req;

	STARPU_MPI_ASSERT_MSG(!req->detached, "MPI_Test cannot be called on a detached request");

	STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	unsigned submitted = req->submitted;
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);

	if (submitted)
	{
		struct _starpu_mpi_req *testing_req;

		/* Initialize the request structure */
		_starpu_mpi_request_init(&testing_req);
		testing_req->flag = flag;
		testing_req->status = status;
		testing_req->other_request = req;
		testing_req->func = _starpu_mpi_test_func;
		testing_req->completed = 0;
		testing_req->request_type = TEST_REQ;

		_STARPU_MPI_INC_POSTED_REQUESTS(1);
		_starpu_mpi_submit_ready_request(testing_req);

		/* We wait for the test request to finish */
		STARPU_PTHREAD_MUTEX_LOCK(&(testing_req->req_mutex));
		while (!(testing_req->completed))
			STARPU_PTHREAD_COND_WAIT(&(testing_req->req_cond), &(testing_req->req_mutex));
		STARPU_PTHREAD_MUTEX_UNLOCK(&(testing_req->req_mutex));

		ret = testing_req->ret;

		if (*(testing_req->flag))
		{
			/* The request was completed so we free the internal
			 * request structure which was automatically allocated
			 * */
			*public_req = NULL;
			if (req->internal_req)
			{
				_starpu_mpi_request_destroy(req->internal_req);
			}
			_starpu_mpi_request_destroy(req);
		}

		_starpu_mpi_request_destroy(testing_req);
	}
	else
	{
		*flag = 0;
	}

	_STARPU_MPI_LOG_OUT();
	return ret;
}

/********************************************************/
/*                                                      */
/*  Barrier functionalities                             */
/*                                                      */
/********************************************************/

static void _starpu_mpi_barrier_func(struct _starpu_mpi_req *barrier_req)
{
	_STARPU_MPI_LOG_IN();

	barrier_req->ret = MPI_Barrier(barrier_req->node_tag.comm);
	STARPU_MPI_ASSERT_MSG(barrier_req->ret == MPI_SUCCESS, "MPI_Barrier returning %s", _starpu_mpi_get_mpi_error_code(barrier_req->ret));

	_starpu_mpi_handle_request_termination(barrier_req);
	_STARPU_MPI_LOG_OUT();
}

int _starpu_mpi_barrier(MPI_Comm comm)
{
	_STARPU_MPI_LOG_IN();

	int ret = posted_requests;
	struct _starpu_mpi_req *barrier_req;

	/* First wait for *both* all tasks and MPI requests to finish, in case
	 * some tasks generate MPI requests, MPI requests generate tasks, etc.
	 */
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
	STARPU_MPI_ASSERT_MSG(!barrier_running, "Concurrent starpu_mpi_barrier is not implemented, even on different communicators");
	barrier_running = 1;
	do
	{
		while (posted_requests)
			/* Wait for all current MPI requests to finish */
			STARPU_PTHREAD_COND_WAIT(&barrier_cond, &progress_mutex);
		/* No current request, clear flag */
		newer_requests = 0;
		STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
		/* Now wait for all tasks */
		starpu_task_wait_for_all();
		STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
		/* Check newer_requests again, in case some MPI requests
		 * triggered by tasks completed and triggered tasks between
		 * wait_for_all finished and we take the lock */
	} while (posted_requests || newer_requests);
	barrier_running = 0;
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

	/* Initialize the request structure */
	_starpu_mpi_request_init(&barrier_req);
	barrier_req->func = _starpu_mpi_barrier_func;
	barrier_req->request_type = BARRIER_REQ;
	barrier_req->node_tag.comm = comm;

	_STARPU_MPI_INC_POSTED_REQUESTS(1);
	_starpu_mpi_submit_ready_request(barrier_req);

	/* We wait for the MPI request to finish */
	STARPU_PTHREAD_MUTEX_LOCK(&barrier_req->req_mutex);
	while (!barrier_req->completed)
		STARPU_PTHREAD_COND_WAIT(&barrier_req->req_cond, &barrier_req->req_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&barrier_req->req_mutex);

	_starpu_mpi_request_destroy(barrier_req);
	_STARPU_MPI_LOG_OUT();

	return ret;
}

int starpu_mpi_barrier(MPI_Comm comm)
{
	_starpu_mpi_barrier(comm);
	return 0;
}

/********************************************************/
/*                                                      */
/*  Progression                                         */
/*                                                      */
/********************************************************/

#ifdef STARPU_VERBOSE
static char *_starpu_mpi_request_type(enum _starpu_mpi_request_type request_type)
{
	switch (request_type)
		{
		case SEND_REQ: return "SEND_REQ";
		case RECV_REQ: return "RECV_REQ";
		case WAIT_REQ: return "WAIT_REQ";
		case TEST_REQ: return "TEST_REQ";
		case BARRIER_REQ: return "BARRIER_REQ";
		case UNKNOWN_REQ: return "UNSET_REQ";
		default: return "unknown request type";
		}
}
#endif

static void _starpu_mpi_handle_request_termination(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();

	_STARPU_MPI_DEBUG(2, "complete MPI request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d registered_datatype %d internal_req %p\n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle, req->ptr,
			  req->datatype_name, (int)req->count, req->registered_datatype, req->internal_req);

	if (req->internal_req)
	{
		free(req->early_data_handle);
		req->early_data_handle = NULL;
	}
	else
	{
		if (req->request_type == RECV_REQ || req->request_type == SEND_REQ)
		{
			if (req->registered_datatype == 0)
			{
				if (req->request_type == SEND_REQ)
				{
					// We need to make sure the communication for sending the size
					// has completed, as MPI can re-order messages, let's call
					// MPI_Wait to make sure data have been sent
					int ret;
					ret = MPI_Wait(&req->size_req, MPI_STATUS_IGNORE);
					STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Wait returning %s", _starpu_mpi_get_mpi_error_code(ret));
					free(req->ptr);
					req->ptr = NULL;
				}
				else if (req->request_type == RECV_REQ)
				{
					// req->ptr is freed by starpu_data_unpack
					starpu_data_unpack(req->data_handle, req->ptr, req->count);
					starpu_memory_deallocate(STARPU_MAIN_RAM, req->count);
				}
			}
			else
			{
				_starpu_mpi_datatype_free(req->data_handle, &req->datatype);
			}
		}
	}

	if (req->data_handle)
		starpu_data_release(req->data_handle);

	if (req->envelope)
	{
		free(req->envelope);
		req->envelope = NULL;
	}

	/* Execute the specified callback, if any */
	if (req->callback)
		req->callback(req->callback_arg);

	/* tell anyone potentially waiting on the request that it is
	 * terminated now */
	STARPU_PTHREAD_MUTEX_LOCK(&req->req_mutex);
	req->completed = 1;
	STARPU_PTHREAD_COND_BROADCAST(&req->req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&req->req_mutex);
	_STARPU_MPI_LOG_OUT();
}

static void _starpu_mpi_early_data_cb(void* arg)
{
	struct _starpu_mpi_early_data_cb_args *args = arg;

	if (args->buffer)
	{
		/* Data has been received as a raw memory, it has to be unpacked */
		struct starpu_data_interface_ops *itf_src = starpu_data_get_interface_ops(args->early_handle);
		struct starpu_data_interface_ops *itf_dst = starpu_data_get_interface_ops(args->data_handle);
		STARPU_MPI_ASSERT_MSG(itf_dst->unpack_data, "The data interface does not define an unpack function\n");
		itf_dst->unpack_data(args->data_handle, STARPU_MAIN_RAM, args->buffer, itf_src->get_size(args->early_handle));
		free(args->buffer);
		args->buffer = NULL;
	}
	else
	{
		struct starpu_data_interface_ops *itf = starpu_data_get_interface_ops(args->early_handle);
		void* itf_src = starpu_data_get_interface_on_node(args->early_handle, STARPU_MAIN_RAM);
		void* itf_dst = starpu_data_get_interface_on_node(args->data_handle, STARPU_MAIN_RAM);

		if (!itf->copy_methods->ram_to_ram)
		{
			_STARPU_MPI_DEBUG(3, "Initiating any_to_any copy..\n");
			itf->copy_methods->any_to_any(itf_src, STARPU_MAIN_RAM, itf_dst, STARPU_MAIN_RAM, NULL);
		}
		else
		{
			_STARPU_MPI_DEBUG(3, "Initiating ram_to_ram copy..\n");
			itf->copy_methods->ram_to_ram(itf_src, STARPU_MAIN_RAM, itf_dst, STARPU_MAIN_RAM);
		}
	}

	_STARPU_MPI_DEBUG(3, "Done, handling release of early_handle..\n");
	starpu_data_release(args->early_handle);

	_STARPU_MPI_DEBUG(3, "Done, handling unregister of early_handle..\n");
	starpu_data_unregister_submit(args->early_handle);

	_STARPU_MPI_DEBUG(3, "Done, handling request %p termination of the already received request\n",args->req);
	// If the request is detached, we need to call _starpu_mpi_handle_request_termination
	// as it will not be called automatically as the request is not in the list detached_requests
	if (args->req)
	{
		if (args->req->detached)
		{
			_starpu_mpi_handle_request_termination(args->req);
			_starpu_mpi_request_destroy(args->req);
		}
		else
		{
			// else: If the request is not detached its termination will
			// be handled when calling starpu_mpi_wait
			// We store in the application request the internal MPI
			// request so that it can be used by starpu_mpi_wait
			args->req->data_request = args->req->internal_req->data_request;
			STARPU_PTHREAD_MUTEX_LOCK(&args->req->req_mutex);
			args->req->submitted = 1;
			STARPU_PTHREAD_COND_BROADCAST(&args->req->req_cond);
			STARPU_PTHREAD_MUTEX_UNLOCK(&args->req->req_mutex);
		}
	}

	free(args);
	args = NULL;
}

static void _starpu_mpi_test_detached_requests(void)
{
	//_STARPU_MPI_LOG_IN();
	int flag;
	struct _starpu_mpi_req *req;

	STARPU_PTHREAD_MUTEX_LOCK(&detached_requests_mutex);

	req = _starpu_mpi_req_list_begin(detached_requests);
	while (req != _starpu_mpi_req_list_end(detached_requests))
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);

		//_STARPU_MPI_DEBUG(3, "Test detached request %p - mpitag %d - TYPE %s %d\n", &req->data_request, req->node_tag.data_tag, _starpu_mpi_request_type(req->request_type), req->node_tag.rank);
#ifdef STARPU_SIMGRID
		req->ret = _starpu_mpi_simgrid_mpi_test(&req->done, &flag);
#else
		req->ret = MPI_Test(&req->data_request, &flag, MPI_STATUS_IGNORE);
#endif

		STARPU_MPI_ASSERT_MSG(req->ret == MPI_SUCCESS, "MPI_Test returning %s", _starpu_mpi_get_mpi_error_code(req->ret));

		if (!flag)
		{
			req = _starpu_mpi_req_list_next(req);
		}
		else
		{
		     	struct _starpu_mpi_req *next_req;
			next_req = _starpu_mpi_req_list_next(req);

			_STARPU_MPI_TRACE_COMPLETE_BEGIN(req->request_type, req->node_tag.rank, req->node_tag.data_tag);

			STARPU_PTHREAD_MUTEX_LOCK(&detached_requests_mutex);
			_starpu_mpi_req_list_erase(detached_requests, req);
			STARPU_PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);
			_starpu_mpi_handle_request_termination(req);

			_STARPU_MPI_TRACE_COMPLETE_END(req->request_type, req->node_tag.rank, req->node_tag.data_tag);

			if (req->is_internal_req == 0)
			{
				_starpu_mpi_request_destroy(req);
			}

			req = next_req;
		}

		STARPU_PTHREAD_MUTEX_LOCK(&detached_requests_mutex);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);
	//_STARPU_MPI_LOG_OUT();
}

static void _starpu_mpi_handle_detached_request(struct _starpu_mpi_req *req)
{
	if (req->detached)
	{
		/* put the submitted request into the list of pending requests
		 * so that it can be handled by the progression mechanisms */
		STARPU_PTHREAD_MUTEX_LOCK(&detached_requests_mutex);
		_starpu_mpi_req_list_push_front(detached_requests, req);
		STARPU_PTHREAD_MUTEX_UNLOCK(&detached_requests_mutex);

		starpu_wake_all_blocked_workers();

		STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
		STARPU_PTHREAD_COND_SIGNAL(&progress_cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
	}
}

static void _starpu_mpi_handle_ready_request(struct _starpu_mpi_req *req)
{
	_STARPU_MPI_LOG_IN();
	STARPU_MPI_ASSERT_MSG(req, "Invalid request");

	/* submit the request to MPI */
	_STARPU_MPI_DEBUG(2, "Handling new request %p type %s tag %d src %d data %p ptr %p datatype '%s' count %d registered_datatype %d \n",
			  req, _starpu_mpi_request_type(req->request_type), req->node_tag.data_tag, req->node_tag.rank, req->data_handle,
			  req->ptr, req->datatype_name, (int)req->count, req->registered_datatype);
	req->func(req);

	_STARPU_MPI_LOG_OUT();
}

static void _starpu_mpi_receive_early_data(struct _starpu_mpi_envelope *envelope, MPI_Status status, MPI_Comm comm)
{
	_STARPU_MPI_DEBUG(20, "Request with tag %d and source %d not found, creating a early_data_handle to receive incoming data..\n", envelope->data_tag, status.MPI_SOURCE);
	_STARPU_MPI_DEBUG(20, "Request sync %d\n", envelope->sync);

	struct _starpu_mpi_early_data_handle* early_data_handle = _starpu_mpi_early_data_create(envelope, status.MPI_SOURCE, comm);
	_starpu_mpi_early_data_add(early_data_handle);

	starpu_data_handle_t data_handle = NULL;
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
	data_handle = _starpu_mpi_tag_get_data_handle_from_tag(envelope->data_tag);
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

	if (data_handle && starpu_data_get_interface_id(data_handle) < STARPU_MAX_INTERFACE_ID)
	{
		/* We know which data will receive it and we won't have to unpack, use just the same kind of data.  */
		early_data_handle->buffer = NULL;
		starpu_data_register_same(&early_data_handle->handle, data_handle);
		//_starpu_mpi_early_data_add(early_data_handle);
	}
	else
	{
		/* The application has not registered yet a data with the tag,
		 * we are going to receive the data as a raw memory, and give it
		 * to the application when it post a receive for this tag
		 */
		_STARPU_MPI_DEBUG(3, "Posting a receive for a data of size %d which has not yet been registered\n", (int)early_data_handle->env->size);
		_STARPU_MPI_MALLOC(early_data_handle->buffer, early_data_handle->env->size);
		starpu_variable_data_register(&early_data_handle->handle, STARPU_MAIN_RAM, (uintptr_t) early_data_handle->buffer, early_data_handle->env->size);
		//_starpu_mpi_early_data_add(early_data_handle);
	}

	_STARPU_MPI_DEBUG(20, "Posting internal detached irecv on early_data_handle with tag %d from comm %ld src %d ..\n",
			  early_data_handle->node_tag.data_tag, (long int)comm, status.MPI_SOURCE);
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
	early_data_handle->req = _starpu_mpi_irecv_common(early_data_handle->handle, status.MPI_SOURCE,
							  early_data_handle->node_tag.data_tag, comm, 1, 0,
							  NULL, NULL, 1, 1, envelope->size);
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

	// We wait until the request is pushed in the
	// ready_request list, that ensures that the next loop
	// will call _starpu_mpi_handle_ready_request
	// on the request and post the corresponding mpi_irecv,
	// otherwise, it may lead to read data as envelop
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
	STARPU_PTHREAD_MUTEX_LOCK(&(early_data_handle->req->posted_mutex));
	while (!(early_data_handle->req->posted))
		STARPU_PTHREAD_COND_WAIT(&(early_data_handle->req->posted_cond), &(early_data_handle->req->posted_mutex));
	STARPU_PTHREAD_MUTEX_UNLOCK(&(early_data_handle->req->posted_mutex));

	STARPU_PTHREAD_MUTEX_LOCK(&early_data_handle->req_mutex);
	early_data_handle->req_ready = 1;
	STARPU_PTHREAD_COND_BROADCAST(&early_data_handle->req_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&early_data_handle->req_mutex);
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
}

static void *_starpu_mpi_progress_thread_func(void *arg)
{
	struct _starpu_mpi_argc_argv *argc_argv = (struct _starpu_mpi_argc_argv *) arg;

	starpu_pthread_setname("MPI");

#ifndef STARPU_SIMGRID
	_starpu_mpi_do_initialize(argc_argv);
#endif

	_starpu_mpi_fake_world_size = starpu_get_env_number("STARPU_MPI_FAKE_SIZE");
	_starpu_mpi_fake_world_rank = starpu_get_env_number("STARPU_MPI_FAKE_RANK");

#ifdef STARPU_SIMGRID
	/* Now that MPI is set up, let the rest of simgrid get initialized */
	char **argv_cpy;
	_STARPU_MPI_MALLOC(argv_cpy, *(argc_argv->argc) * sizeof(char*));
	int i;
	for (i = 0; i < *(argc_argv->argc); i++)
		argv_cpy[i] = strdup((*(argc_argv->argv))[i]);
	MSG_process_create_with_arguments("main", smpi_simulated_main_, NULL, _starpu_simgrid_get_host_by_name("MAIN"), *(argc_argv->argc), argv_cpy);
	/* And set TSD for us */
#ifdef HAVE_SMPI_PROCESS_SET_USER_DATA
	smpi_process_set_user_data(calloc(MAX_TSD + 1, sizeof(void*)));
#endif
#endif

#ifdef STARPU_USE_FXT
	_starpu_fxt_wait_initialisation();
#endif //STARPU_USE_FXT

	{
		_STARPU_MPI_TRACE_START(argc_argv->rank, argc_argv->world_size);
#ifdef STARPU_USE_FXT
		starpu_profiling_set_id(argc_argv->rank);
#endif //STARPU_USE_FXT
	}

	_starpu_mpi_add_sync_point_in_fxt();
	_starpu_mpi_comm_amounts_init(argc_argv->comm);
	_starpu_mpi_cache_init(argc_argv->comm);
	_starpu_mpi_select_node_init();
	_starpu_mpi_tag_init();
	_starpu_mpi_comm_init(argc_argv->comm);

	_starpu_mpi_early_request_init();
	_starpu_mpi_early_data_init();
	_starpu_mpi_sync_data_init();
	_starpu_mpi_datatype_init();

	/* notify the main thread that the progression thread is ready */
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
	running = 1;
	STARPU_PTHREAD_COND_SIGNAL(&progress_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

#ifdef STARPU_SIMGRID
	starpu_pthread_wait_init(&wait);
	starpu_pthread_queue_init(&dontsleep);
	starpu_pthread_queue_register(&wait, &dontsleep);
#endif

	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

 	int envelope_request_submitted = 0;

	while (running || posted_requests || !(_starpu_mpi_req_list_empty(ready_requests)) || !(_starpu_mpi_req_list_empty(detached_requests)))// || !(_starpu_mpi_early_request_count()) || !(_starpu_mpi_sync_data_count()))
	{
#ifdef STARPU_SIMGRID
		starpu_pthread_wait_reset(&wait);
#endif
		/* shall we block ? */
		unsigned block = _starpu_mpi_req_list_empty(ready_requests) && _starpu_mpi_early_request_count() == 0 && _starpu_mpi_sync_data_count() == 0 && _starpu_mpi_req_list_empty(detached_requests);

		if (block)
		{
			_STARPU_MPI_DEBUG(3, "NO MORE REQUESTS TO HANDLE\n");
			_STARPU_MPI_TRACE_SLEEP_BEGIN();

			if (barrier_running)
				/* Tell mpi_barrier */
				STARPU_PTHREAD_COND_SIGNAL(&barrier_cond);
			STARPU_PTHREAD_COND_WAIT(&progress_cond, &progress_mutex);

			_STARPU_MPI_TRACE_SLEEP_END();
		}

		/* get one request */
		int n = 0;
		while (!_starpu_mpi_req_list_empty(ready_requests))
		{
			struct _starpu_mpi_req *req;

			if (n++ == NREADY_PROCESS)
				/* Already spent some time on submitting ready requests, poll before processing more ready requests */
				break;

			req = _starpu_mpi_req_list_pop_back(ready_requests);

			/* handling a request is likely to block for a while
			 * (on a sync_data_with_mem call), we want to let the
			 * application submit requests in the meantime, so we
			 * release the lock. */
			STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
			_starpu_mpi_handle_ready_request(req);
			STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
		}

		/* If there is no currently submitted envelope_request submitted to
                 * catch envelopes from senders, and there is some pending
                 * receive requests on our side, we resubmit a header request. */
		if (((_starpu_mpi_early_request_count() > 0) || (_starpu_mpi_sync_data_count() > 0)) && (envelope_request_submitted == 0))// && (HASH_COUNT(_starpu_mpi_early_data_handle_hashmap) == 0))
		{
			_starpu_mpi_comm_post_recv();
			envelope_request_submitted = 1;
		}

		/* test whether there are some terminated "detached request" */
		STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
		_starpu_mpi_test_detached_requests();
		STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);

		if (envelope_request_submitted == 1)
		{
			int flag;
			struct _starpu_mpi_envelope *envelope;
			MPI_Status envelope_status;
			MPI_Comm envelope_comm;

			/* test whether an envelope has arrived. */
			flag = _starpu_mpi_comm_test_recv(&envelope_status, &envelope, &envelope_comm);

			if (flag)
			{
				_STARPU_MPI_DEBUG(4, "Envelope received with mode %d\n", envelope->mode);
				if (envelope->mode == _STARPU_MPI_ENVELOPE_SYNC_READY)
				{
					struct _starpu_mpi_req *_sync_req = _starpu_mpi_sync_data_find(envelope->data_tag, envelope_status.MPI_SOURCE, envelope_comm);
					_STARPU_MPI_DEBUG(20, "Sending data with tag %d to node %d\n", _sync_req->node_tag.data_tag, envelope_status.MPI_SOURCE);
					STARPU_MPI_ASSERT_MSG(envelope->data_tag == _sync_req->node_tag.data_tag, "Tag mismatch (envelope %d != req %d)\n", envelope->data_tag, _sync_req->node_tag.data_tag);
					STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
					_starpu_mpi_isend_data_func(_sync_req);
					STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
				}
				else
				{
					_STARPU_MPI_DEBUG(3, "Searching for application request with tag %d and source %d (size %ld)\n", envelope->data_tag, envelope_status.MPI_SOURCE, envelope->size);

					struct _starpu_mpi_req *early_request = _starpu_mpi_early_request_dequeue(envelope->data_tag, envelope_status.MPI_SOURCE, envelope_comm);

					/* Case: a data will arrive before a matching receive is
					 * posted by the application. Create a temporary handle to
					 * store the incoming data, submit a starpu_mpi_irecv_detached
					 * on this handle, and store it as an early_data
					 */
					if (early_request == NULL)
					{
						if (envelope->sync)
						{
							_STARPU_MPI_DEBUG(2000, "-------------------------> adding request for tag %d\n", envelope->data_tag);
							struct _starpu_mpi_req *new_req;
#ifdef STARPU_DEVEL
#warning creating a request is not really useful.
#endif
							/* Initialize the request structure */
							_starpu_mpi_request_init(&new_req);
							new_req->request_type = RECV_REQ;
							new_req->data_handle = NULL;
							new_req->node_tag.rank = envelope_status.MPI_SOURCE;
							new_req->node_tag.data_tag = envelope->data_tag;
							new_req->node_tag.comm = envelope_comm;
							new_req->detached = 1;
							new_req->sync = 1;
							new_req->callback = NULL;
							new_req->callback_arg = NULL;
							new_req->func = _starpu_mpi_irecv_data_func;
							new_req->sequential_consistency = 1;
							new_req->is_internal_req = 0; // ????
							new_req->count = envelope->size;
							_starpu_mpi_sync_data_add(new_req);
						}
						else
						{
							_starpu_mpi_receive_early_data(envelope, envelope_status, envelope_comm);
						}
					}
					/* Case: a matching application request has been found for
					 * the incoming data, we handle the correct allocation
					 * of the pointer associated to the data handle, then
					 * submit the corresponding receive with
					 * _starpu_mpi_handle_ready_request. */
					else
					{
						_STARPU_MPI_DEBUG(2000, "A matching application request has been found for the incoming data with tag %d\n", envelope->data_tag);
						_STARPU_MPI_DEBUG(2000, "Request sync %d\n", envelope->sync);

						early_request->sync = envelope->sync;
						_starpu_mpi_datatype_allocate(early_request->data_handle, early_request);
						if (early_request->registered_datatype == 1)
						{
							early_request->count = 1;
							early_request->ptr = starpu_data_get_local_ptr(early_request->data_handle);
						}
						else
						{
							early_request->count = envelope->size;
							_STARPU_MPI_MALLOC(early_request->ptr, early_request->count);
							starpu_memory_allocate(STARPU_MAIN_RAM, early_request->count, STARPU_MEMORY_OVERFLOW);

							STARPU_MPI_ASSERT_MSG(early_request->ptr, "cannot allocate message of size %ld\n", early_request->count);
						}

						_STARPU_MPI_DEBUG(3, "Handling new request... \n");
						/* handling a request is likely to block for a while
						 * (on a sync_data_with_mem call), we want to let the
						 * application submit requests in the meantime, so we
						 * release the lock. */
						STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
						_starpu_mpi_handle_ready_request(early_request);
						STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
					}
				}
				envelope_request_submitted = 0;
			}
			else
			{
				//_STARPU_MPI_DEBUG(4, "Nothing received, continue ..\n");
			}
		}
#ifdef STARPU_SIMGRID
		STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
		starpu_pthread_wait_wait(&wait);
		STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
#endif
	}

	if (envelope_request_submitted)
	{
		_starpu_mpi_comm_cancel_recv();
		envelope_request_submitted = 0;
	}


#ifdef STARPU_SIMGRID
	STARPU_PTHREAD_MUTEX_LOCK(&wait_counter_mutex);
	while (wait_counter != 0)
		STARPU_PTHREAD_COND_WAIT(&wait_counter_cond, &wait_counter_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&wait_counter_mutex);

	STARPU_PTHREAD_MUTEX_DESTROY(&wait_counter_mutex);
	STARPU_PTHREAD_COND_DESTROY(&wait_counter_cond);

	starpu_pthread_queue_unregister(&wait, &dontsleep);
	starpu_pthread_queue_destroy(&dontsleep);
	starpu_pthread_wait_destroy(&wait);
#endif

	STARPU_MPI_ASSERT_MSG(_starpu_mpi_req_list_empty(detached_requests), "List of detached requests not empty");
	STARPU_MPI_ASSERT_MSG(_starpu_mpi_req_list_empty(ready_requests), "List of ready requests not empty");
	STARPU_MPI_ASSERT_MSG(posted_requests == 0, "Number of posted request is not zero");
	_starpu_mpi_early_request_check_termination();
	_starpu_mpi_early_data_check_termination();
	_starpu_mpi_sync_data_check_termination();

	if (argc_argv->initialize_mpi)
	{
		_STARPU_MPI_DEBUG(3, "Calling MPI_Finalize()\n");
		MPI_Finalize();
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

	_starpu_mpi_sync_data_shutdown();
	_starpu_mpi_early_data_shutdown();
	_starpu_mpi_early_request_shutdown();
	_starpu_mpi_datatype_shutdown();
	free(argc_argv);

	return NULL;
}

static void _starpu_mpi_add_sync_point_in_fxt(void)
{
#ifdef STARPU_USE_FXT
	int rank;
	int worldsize;
	int ret;

	starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &worldsize);

	ret = MPI_Barrier(MPI_COMM_WORLD);
	STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Barrier returning %s", _starpu_mpi_get_mpi_error_code(ret));

	/* We generate a "unique" key so that we can make sure that different
	 * FxT traces come from the same MPI run. */
	int random_number;

	/* XXX perhaps we don't want to generate a new seed if the application
	 * specified some reproductible behaviour ? */
	if (rank == 0)
	{
		srand(time(NULL));
		random_number = rand();
	}

	ret = MPI_Bcast(&random_number, 1, MPI_INT, 0, MPI_COMM_WORLD);
	STARPU_MPI_ASSERT_MSG(ret == MPI_SUCCESS, "MPI_Bcast returning %s", _starpu_mpi_get_mpi_error_code(ret));

	_STARPU_MPI_TRACE_BARRIER(rank, worldsize, random_number);

	_STARPU_MPI_DEBUG(3, "unique key %x\n", random_number);
#endif
}

int _starpu_mpi_progress_init(struct _starpu_mpi_argc_argv *argc_argv)
{
        STARPU_PTHREAD_MUTEX_INIT(&progress_mutex, NULL);
        STARPU_PTHREAD_COND_INIT(&progress_cond, NULL);
        STARPU_PTHREAD_COND_INIT(&barrier_cond, NULL);
        ready_requests = _starpu_mpi_req_list_new();

        STARPU_PTHREAD_MUTEX_INIT(&detached_requests_mutex, NULL);
        detached_requests = _starpu_mpi_req_list_new();

        STARPU_PTHREAD_MUTEX_INIT(&mutex_posted_requests, NULL);
        _starpu_mpi_comm = starpu_getenv("STARPU_MPI_COMM") != NULL;

#ifdef STARPU_SIMGRID
	STARPU_PTHREAD_MUTEX_INIT(&wait_counter_mutex, NULL);
	STARPU_PTHREAD_COND_INIT(&wait_counter_cond, NULL);
#endif

#ifdef STARPU_SIMGRID
        _starpu_mpi_progress_thread_func(argc_argv);
        return 0;
#else
        STARPU_PTHREAD_CREATE(&progress_thread, NULL, _starpu_mpi_progress_thread_func, argc_argv);

        STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
        while (!running)
                STARPU_PTHREAD_COND_WAIT(&progress_cond, &progress_mutex);
        STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

        return 0;
#endif
}

#ifdef STARPU_SIMGRID
void _starpu_mpi_wait_for_initialization()
{
	/* Wait for MPI initialization to finish */
	STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
	while (!running)
		STARPU_PTHREAD_COND_WAIT(&progress_cond, &progress_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);
}
#endif

void _starpu_mpi_progress_shutdown(int *value)
{
        STARPU_PTHREAD_MUTEX_LOCK(&progress_mutex);
        running = 0;
        STARPU_PTHREAD_COND_BROADCAST(&progress_cond);

#ifdef STARPU_SIMGRID
	starpu_pthread_queue_signal(&dontsleep);
#endif
        STARPU_PTHREAD_MUTEX_UNLOCK(&progress_mutex);

#ifdef STARPU_SIMGRID
	/* FIXME: should rather properly wait for _starpu_mpi_progress_thread_func to finish */
	(void) value;
	MSG_process_sleep(1);
#else
	starpu_pthread_join(progress_thread, (void *)value);
#endif

        /* free the request queues */
        _starpu_mpi_req_list_delete(detached_requests);
        _starpu_mpi_req_list_delete(ready_requests);

        STARPU_PTHREAD_MUTEX_DESTROY(&mutex_posted_requests);
        STARPU_PTHREAD_MUTEX_DESTROY(&progress_mutex);
        STARPU_PTHREAD_COND_DESTROY(&barrier_cond);
}

void _starpu_mpi_data_clear(starpu_data_handle_t data_handle)
{
	_starpu_mpi_tag_data_release(data_handle);
	_starpu_mpi_cache_data_clear(data_handle);
	free(data_handle->mpi_data);
}

void starpu_mpi_data_register_comm(starpu_data_handle_t data_handle, int tag, int rank, MPI_Comm comm)
{
	struct _starpu_mpi_data *mpi_data;
	if (data_handle->mpi_data)
	{
		mpi_data = data_handle->mpi_data;
	}
	else
	{
		_STARPU_CALLOC(mpi_data, 1, sizeof(struct _starpu_mpi_data));
		mpi_data->magic = 42;
		mpi_data->node_tag.data_tag = -1;
		mpi_data->node_tag.rank = -1;
		mpi_data->node_tag.comm = MPI_COMM_WORLD;
		data_handle->mpi_data = mpi_data;
		_starpu_mpi_cache_data_init(data_handle);
		_starpu_mpi_tag_data_register(data_handle, tag);
		_starpu_data_set_unregister_hook(data_handle, _starpu_mpi_data_clear);
	}

	if (tag != -1)
	{
		mpi_data->node_tag.data_tag = tag;
	}
	if (rank != -1)
	{
		_STARPU_MPI_TRACE_DATA_SET_RANK(data_handle, rank);
		mpi_data->node_tag.rank = rank;
		mpi_data->node_tag.comm = comm;
		_starpu_mpi_comm_register(comm);
	}
}

void starpu_mpi_data_set_rank_comm(starpu_data_handle_t handle, int rank, MPI_Comm comm)
{
	starpu_mpi_data_register_comm(handle, -1, rank, comm);
}

void starpu_mpi_data_set_tag(starpu_data_handle_t handle, int tag)
{
	starpu_mpi_data_register_comm(handle, tag, -1, MPI_COMM_WORLD);
}

int starpu_mpi_data_get_rank(starpu_data_handle_t data)
{
	STARPU_ASSERT_MSG(data->mpi_data, "starpu_mpi_data_register MUST be called for data %p\n", data);
	return ((struct _starpu_mpi_data *)(data->mpi_data))->node_tag.rank;
}

int starpu_mpi_data_get_tag(starpu_data_handle_t data)
{
	STARPU_ASSERT_MSG(data->mpi_data, "starpu_mpi_data_register MUST be called for data %p\n", data);
	return ((struct _starpu_mpi_data *)(data->mpi_data))->node_tag.data_tag;
}

void starpu_mpi_get_data_on_node_detached(MPI_Comm comm, starpu_data_handle_t data_handle, int node, void (*callback)(void*), void *arg)
{
	int me, rank, tag;

	rank = starpu_mpi_data_get_rank(data_handle);
	tag = starpu_mpi_data_get_tag(data_handle);
	if (rank == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register() or starpu_mpi_data_register()\n");
	}
	if (tag == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register() or starpu_mpi_data_register()\n");
	}
	starpu_mpi_comm_rank(comm, &me);

	if (node == rank) return;

	if (me == node)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_received = _starpu_mpi_cache_received_data_set(data_handle);
		if (already_received == 0)
		{
			_STARPU_MPI_DEBUG(1, "Receiving data %p from %d\n", data_handle, rank);
			starpu_mpi_irecv_detached(data_handle, rank, tag, comm, callback, arg);
		}
	}
	else if (me == rank)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_sent = _starpu_mpi_cache_sent_data_set(data_handle, node);
		if (already_sent == 0)
		{
			_STARPU_MPI_DEBUG(1, "Sending data %p to %d\n", data_handle, node);
			starpu_mpi_isend_detached(data_handle, node, tag, comm, NULL, NULL);
		}
	}
}

void starpu_mpi_get_data_on_node(MPI_Comm comm, starpu_data_handle_t data_handle, int node)
{
	int me, rank, tag;

	rank = starpu_mpi_data_get_rank(data_handle);
	tag = starpu_mpi_data_get_tag(data_handle);
	if (rank == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI rank of this data, using starpu_mpi_data_register\n");
	}
	if (tag == -1)
	{
		_STARPU_ERROR("StarPU needs to be told the MPI tag of this data, using starpu_mpi_data_register\n");
	}
	starpu_mpi_comm_rank(comm, &me);

	if (node == rank) return;

	if (me == node)
	{
		MPI_Status status;
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_received = _starpu_mpi_cache_received_data_set(data_handle);
		if (already_received == 0)
		{
			_STARPU_MPI_DEBUG(1, "Receiving data %p from %d\n", data_handle, rank);
			starpu_mpi_recv(data_handle, rank, tag, comm, &status);
		}
	}
	else if (me == rank)
	{
		_STARPU_MPI_DEBUG(1, "Migrating data %p from %d to %d\n", data_handle, rank, node);
		int already_sent = _starpu_mpi_cache_sent_data_set(data_handle, node);
		if (already_sent == 0)
		{
			_STARPU_MPI_DEBUG(1, "Sending data %p to %d\n", data_handle, node);
			starpu_mpi_send(data_handle, node, tag, comm);
		}
	}
}

void starpu_mpi_data_migrate(MPI_Comm comm, starpu_data_handle_t data, int new_rank)
{
	int old_rank = starpu_mpi_data_get_rank(data);
	if (new_rank == old_rank)
		/* Already there */
		return;

	/* First submit data migration if it's not already on destination */
	starpu_mpi_get_data_on_node_detached(comm, data, new_rank, NULL, NULL);

	/* And note new owner */
	starpu_mpi_data_set_rank_comm(data, new_rank, comm);

	/* Flush cache in all other nodes */
	/* TODO: Ideally we'd transmit the knowledge of who owns it */
	starpu_mpi_cache_flush(comm, data);
	return;
}

int starpu_mpi_wait_for_all(MPI_Comm comm)
{
	int mpi = 1;
	int task = 1;
	while (task || mpi)
	{
		task = _starpu_task_wait_for_all_and_return_nb_waited_tasks();
		mpi = _starpu_mpi_barrier(comm);
	}
	return 0;
}
