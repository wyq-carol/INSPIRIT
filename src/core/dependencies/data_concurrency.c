/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <core/dependencies/data_concurrency.h>
#include <datawizard/coherency.h>
#include <core/sched_policy.h>
#include <common/starpu_spinlock.h>
#include <datawizard/sort_data_handles.h>

/*
 * The different types of data accesses are STARPU_R, STARPU_RW, STARPU_W,
 * STARPU_SCRATCH and STARPU_REDUX. STARPU_RW is managed as a STARPU_W access.
 * - A single STARPU_W access is allowed at a time.
 * - Concurrent STARPU_R accesses are allowed.
 * - Concurrent STARPU_SCRATCH accesses are allowed.
 * - Concurrent STARPU_REDUX accesses are allowed.
 */

/* the header lock must be taken by the caller */
static unsigned may_unlock_data_req_list_head(starpu_data_handle handle)
{
	/* if there is no one to unlock ... */
	if (starpu_data_requester_list_empty(handle->req_list))
		return 0;

	/* if there is no reference to the data anymore, we can use it */
	if (handle->refcnt == 0)
		return 1;

	if (handle->current_mode == STARPU_W)
		return 0;

	/* data->current_mode == STARPU_R, so we can process more readers */
	starpu_data_requester_t r = starpu_data_requester_list_front(handle->req_list);

	starpu_access_mode r_mode = r->mode;
	if (r_mode == STARPU_RW)
		r_mode = STARPU_W;
	
	/* If this is a STARPU_R, STARPU_SCRATCH or STARPU_REDUX type of
	 * access, we only proceed if the cuurrent mode is the same as the
	 * requested mode. */
	return (r_mode == handle->current_mode);
}

/* Try to submit a data request, in case the request can be processed
 * immediatly, return 0, if there is still a dependency that is not compatible
 * with the current mode, the request is put in the per-handle list of
 * "requesters", and this function returns 1. */
static unsigned _starpu_attempt_to_submit_data_request(unsigned request_from_codelet,
					starpu_data_handle handle, starpu_access_mode mode,
					void (*callback)(void *), void *argcb,
					starpu_job_t j, unsigned buffer_index)
{
	unsigned ret;

	if (mode == STARPU_RW)
		mode = STARPU_W;

	/* Take the lock protecting the header. We try to do some progression
	 * in case this is called from a worker, otherwise we just wait for the
	 * lock to be available. */
	if (request_from_codelet)
	{
		while (_starpu_spin_trylock(&handle->header_lock))
			_starpu_datawizard_progress(_starpu_get_local_memory_node(), 0);
	}
	else {
		_starpu_spin_lock(&handle->header_lock);
	}

	/* If there is currently nobody accessing the piece of data, or it's
	 * not another writter and if this is the same type of access as the
	 * current one, we can proceed. */
	if ((handle->refcnt == 0) || (!(mode == STARPU_W) && (handle->current_mode == mode)))
	{
		handle->refcnt++;
		handle->current_mode = mode;

		/* success */
		ret = 0;
	}
	else
	{
		/* there cannot be multiple writers or a new writer
		 * while the data is in read mode */
		
		/* enqueue the request */
		starpu_data_requester_t r = starpu_data_requester_new();
			r->mode = mode;
			r->is_requested_by_codelet = request_from_codelet;
			r->j = j;
			r->buffer_index = buffer_index;
			r->ready_data_callback = callback;
			r->argcb = argcb;

		starpu_data_requester_list_push_back(handle->req_list, r);

		/* failed */
		ret = 1;
	}

	_starpu_spin_unlock(&handle->header_lock);
	return ret;

}


unsigned _starpu_attempt_to_submit_data_request_from_apps(starpu_data_handle handle, starpu_access_mode mode,
						void (*callback)(void *), void *argcb)
{
	return _starpu_attempt_to_submit_data_request(0, handle, mode, callback, argcb, NULL, 0);
}

static unsigned attempt_to_submit_data_request_from_job(starpu_job_t j, unsigned buffer_index)
{
	/* Note that we do not access j->task->buffers, but j->ordered_buffers
	 * which is a sorted copy of it. */
	starpu_data_handle handle = j->ordered_buffers[buffer_index].handle;
	starpu_access_mode mode = j->ordered_buffers[buffer_index].mode;

	return _starpu_attempt_to_submit_data_request(1, handle, mode, NULL, NULL, j, buffer_index);

}

static unsigned _submit_job_enforce_data_deps(starpu_job_t j, unsigned start_buffer_index)
{
	unsigned buf;

	unsigned nbuffers = j->task->cl->nbuffers;
	for (buf = start_buffer_index; buf < nbuffers; buf++)
	{
                if (attempt_to_submit_data_request_from_job(j, buf)) {
                        j->task->status = STARPU_TASK_BLOCKED_ON_JOB;
			return 1;
                }
	}

	return 0;
}

/* When a new task is submitted, we make sure that there cannot be codelets
   with concurrent data-access at the same time in the scheduling engine (eg.
   there can be 2 tasks reading a piece of data, but there cannot be one
   reading and another writing) */
unsigned _starpu_submit_job_enforce_data_deps(starpu_job_t j)
{
	struct starpu_codelet_t *cl = j->task->cl;

	if ((cl == NULL) || (cl->nbuffers == 0))
		return 0;

	/* Compute an ordered list of the different pieces of data so that we
	 * grab then according to a total order, thus avoiding a deadlock
	 * condition */
	memcpy(j->ordered_buffers, j->task->buffers, cl->nbuffers*sizeof(starpu_buffer_descr));
	_starpu_sort_task_handles(j->ordered_buffers, cl->nbuffers);

	return _submit_job_enforce_data_deps(j, 0);
}

static unsigned unlock_one_requester(starpu_data_requester_t r)
{
	starpu_job_t j = r->j;
	unsigned nbuffers = j->task->cl->nbuffers;
	unsigned buffer_index = r->buffer_index;

	if (buffer_index + 1 < nbuffers)
	{
		/* not all buffers are protected yet */
		return _submit_job_enforce_data_deps(j, buffer_index + 1);
	}
	else
		return 0;
}

/* The header lock must already be taken by the caller */
void _starpu_notify_data_dependencies(starpu_data_handle handle)
{
	/* A data access has finished so we remove a reference. */
	handle->refcnt--;

	while (may_unlock_data_req_list_head(handle))
	{
		/* Grab the head of the requester list and unlock it. */
		starpu_data_requester_t r = starpu_data_requester_list_pop_front(handle->req_list);

		/* The data is now attributed to that request so we put a
		 * reference on it. */
		handle->refcnt++;
	
		/* STARPU_RW accesses are treated as STARPU_W */
		starpu_access_mode r_mode = r->mode;
		if (r_mode == STARPU_RW)
			r_mode = STARPU_W;

		handle->current_mode = r_mode;

		_starpu_spin_unlock(&handle->header_lock);

		if (r->is_requested_by_codelet)
		{
			if (!unlock_one_requester(r))
				_starpu_push_task(r->j, 0);
		}
		else
		{
			STARPU_ASSERT(r->ready_data_callback);

			/* execute the callback associated with the data requester */
			r->ready_data_callback(r->argcb);
		}

		starpu_data_requester_delete(r);
		
		_starpu_spin_lock(&handle->header_lock);
	}
}
