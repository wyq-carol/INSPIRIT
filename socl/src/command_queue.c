/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2012 University of Bordeaux
 * Copyright (C) 2012 CNRS
 * Copyright (C) 2012 Vincent Danjean <Vincent.Danjean@ens-lyon.org>
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

#include "socl.h"
#include "task.h"
#include "gc.h"

/**
 * WARNING: command queues do NOT hold references on events. Only events hold references
 * on command queues. This way, event release will automatically remove the event from
 * its command queue.
 */


void command_queue_enqueue_ex(cl_command_queue cq, cl_command cmd, cl_uint num_events, const cl_event * events) {

	/* Check if the command is a barrier */
	int is_barrier = (cmd->typ == CL_COMMAND_BARRIER || !(cq->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE));

	/* Add references to the command queue */
	gc_entity_store(&cmd->cq, cq);
	gc_entity_store(&cmd->event->cq, cq);

	/* Lock command queue */
	pthread_mutex_lock(&cq->mutex);

	/*** Number of dependencies ***/
	int ndeps = num_events;

	/* Add dependency to last barrier if applicable */
	if (cq->barrier != NULL)
		ndeps++;

	/* Add dependencies to out-of-order events (if any) */
	if (is_barrier) {
		command_list cl = cq->commands;
		while (cl != NULL) {
			ndeps++;
			cl = cl->next;
		}
	}

	/*** Dependencies ***/
	cl_event * deps = malloc(ndeps * sizeof(cl_event));

	int n = 0;

	/* Add dependency to last barrier if applicable */
	if (cq->barrier != NULL) deps[n++] = cq->barrier->event;

	/* Add dependencies to out-of-order events (if any) */
	if (is_barrier) {
		command_list cl = cq->commands;
		while (cl != NULL) {
			deps[n++] = cl->cmd->event;
			cl = cl->next;
		}
	}

	/* Add explicit dependencies */
	unsigned i;
	for (i=0; i<num_events; i++) {
		deps[n++] = events[i];
	}

	/* Make all dependencies explicit for the command */
	cmd->num_events = ndeps;
	cmd->events = deps;

	/* Increment event ref count */
	gc_entity_retain(cmd->event);

	/* Insert command in the queue */
	if (is_barrier) {
		/* Remove out-of-order commands */
		cq->commands = NULL;
		/* Register the command as the last barrier */
		cq->barrier = cmd;
	}
	else {
		/* Add command to the list of out-of-order commands */
		cq->commands = command_list_cons(cmd, cq->commands);
	}

	/* Unlock command queue */
	pthread_mutex_unlock(&cq->mutex);

	/* Submit command */
	command_submit_ex(cmd);
}
