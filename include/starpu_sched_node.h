/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Simon Archipoff
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

#ifndef __STARPU_SCHED_NODE_H__
#define __STARPU_SCHED_NODE_H__
#include <starpu.h>
#include <common/starpu_spinlock.h>
#include <starpu_bitmap.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

/* struct starpu_sched_node are scheduler modules, a scheduler is a tree-like
 * structure of them, some parts of scheduler can be shared by several contexes
 * to perform some local optimisations, so, for all nodes, a list of father is
 * defined indexed by sched_ctx_id
 *
 * they embed there specialised method in a pseudo object-style, so calls are like node->push_task(node,task)
 *
 */
struct starpu_sched_node
{
	/* node->push_task(node, task)
	 * this function is called to push a task on node subtree, this can either
	 * perform a recursive call on a child or store the task in the node, then
	 * it will be returned by a further pop_task call
	 *
	 * the caller must ensure that node is able to execute task
	 */
	int (*push_task)(struct starpu_sched_node *,
			 struct starpu_task *);
	/* this function is called by workers to get a task on them fathers
	 * this function should first return a localy stored task or perform
	 * a recursive call on father
	 *
	 * a default implementation simply do a recursive call on father
	 */
	struct starpu_task * (*pop_task)(struct starpu_sched_node *,
					 unsigned sched_ctx_id);

	/* node->push_back_task(node, task)
	 * This function can be called by room-made functions to permit
	 * the user to specify a particular push function which allows to
	 * push back the task if the push submitted by the room function fail.
	 */
	int (*push_back_task)(struct starpu_sched_node *,
			 struct starpu_task *);
	/* this function is an heuristic that compute load of subtree, basicaly
	 * it compute
	 * estimated_load(node) = sum(estimated_load(node_childs)) +
	 *          nb_local_tasks / average(relative_speedup(underlying_worker))
	 */
	double (*estimated_load)(struct starpu_sched_node * node);

	double (*estimated_end)(struct starpu_sched_node * node);
	/* the numbers of node's childs
	 */
	int nchilds;
	/* the vector of node's childs
	 */
	struct starpu_sched_node ** childs;
	/* may be shared by several contexts
	 * so we need several fathers
	 */
	struct starpu_sched_node * fathers[STARPU_NMAX_SCHED_CTXS];
	/* the set of workers in the node's subtree
	 */
	struct starpu_bitmap * workers;
	/* the workers available in context
	 * this member is set with : 
	 * node->workers UNION tree->workers UNION
	 * node->child[i]->workers_in_ctx iff exist x such as node->childs[i]->fathers[x] == node
	 */
	struct starpu_bitmap * workers_in_ctx;
	
	/* node's private data, no restriction on use
	 */
	void * data;

	void (*add_child)(struct starpu_sched_node * node, struct starpu_sched_node * child);
	void (*remove_child)(struct starpu_sched_node * node, struct starpu_sched_node * child);

	/* this function is called for each node when workers are added or removed from a context
	 */
	void (*notify_change_workers)(struct starpu_sched_node * node);

	/* this function is called by starpu_sched_node_destroy just before freeing node
	 */
	void (*deinit_data)(struct starpu_sched_node * node);
	/* is_homogeneous is 0 iff workers in the node's subtree are heterogeneous,
	 * this field is set and updated automaticaly, you shouldn't write on it
	 */
	int properties;

	/* This function is called by a node which implements a queue, allowing it to
	 * signify to its fathers that an empty slot is available in its queue.
	 * The basic implementation of this function is a recursive call to its
	 * fathers, the user have to specify a personally-made function to catch those
	 * calls.
	 */ 
	void (*room)(struct starpu_sched_node * node, unsigned sched_ctx_id);
	/* This function allow a node to wake up a worker.
	 * It is currently called by node which implements a queue, to signify to
	 * its childs that a task have been pushed in its local queue, and is
	 * available to been popped by a worker, for example.
	 * The basic implementation of this function is a recursive call to
	 * its childs, until at least one worker have been woken up.
	 */
	int (*avail)(struct starpu_sched_node * node);

#ifdef STARPU_HAVE_HWLOC
	/* in case of a hierarchical scheduler, this is set to the part of
	 * topology that is binded to this node, eg: a numa node for a ws
	 * node that would balance load between underlying sockets
	 */
	hwloc_obj_t obj;
#endif
};
enum starpu_sched_node_properties
{
	STARPU_SCHED_NODE_HOMOGENEOUS = (1<<0),
	STARPU_SCHED_NODE_SINGLE_MEMORY_NODE = (1<<1)
};

#define STARPU_SCHED_NODE_IS_HOMOGENEOUS(node) ((node)->properties & STARPU_SCHED_NODE_HOMOGENEOUS)
#define STARPU_SCHED_NODE_IS_SINGLE_MEMORY_NODE(node) ((node)->properties & STARPU_SCHED_NODE_SINGLE_MEMORY_NODE)

struct starpu_sched_tree
{
	struct starpu_sched_node * root;
	struct starpu_bitmap * workers;
	unsigned sched_ctx_id;
	/* this lock is used to protect the scheduler,
	 * it is taken in read mode pushing a task
	 * and in write mode for adding or removing workers
	 */
	starpu_pthread_mutex_t lock;
};

struct starpu_sched_node * starpu_sched_node_create(void);

void starpu_sched_node_destroy(struct starpu_sched_node * node);
void starpu_sched_node_set_father(struct starpu_sched_node *node, struct starpu_sched_node *father_node, unsigned sched_ctx_id);
void starpu_sched_node_add_child(struct starpu_sched_node * node, struct starpu_sched_node * child);
void starpu_sched_node_remove_child(struct starpu_sched_node * node, struct starpu_sched_node * child);


int starpu_sched_node_can_execute_task(struct starpu_sched_node * node, struct starpu_task * task);
int STARPU_WARN_UNUSED_RESULT starpu_sched_node_execute_preds(struct starpu_sched_node * node, struct starpu_task * task, double * length);
double starpu_sched_node_transfer_length(struct starpu_sched_node * node, struct starpu_task * task);
void starpu_sched_node_prefetch_on_node(struct starpu_sched_node * node, struct starpu_task * task);


/* no public create function for workers because we dont want to have several node_worker for a single workerid */
struct starpu_sched_node * starpu_sched_node_worker_get(int workerid);


/* this function compare the available function of the node with the standard available for worker nodes*/
int starpu_sched_node_is_worker(struct starpu_sched_node * node);
int starpu_sched_node_is_simple_worker(struct starpu_sched_node * node);
int starpu_sched_node_is_combined_worker(struct starpu_sched_node * node);
int starpu_sched_node_worker_get_workerid(struct starpu_sched_node * worker_node);

struct starpu_fifo_data
{
	unsigned ntasks_threshold;
	double exp_len_threshold;
};

struct starpu_sched_node * starpu_sched_node_fifo_create(struct starpu_fifo_data * fifo_data);
int starpu_sched_node_is_fifo(struct starpu_sched_node * node);

struct starpu_prio_data
{
	unsigned ntasks_threshold;
	double exp_len_threshold;
};

struct starpu_sched_node * starpu_sched_node_prio_create(struct starpu_prio_data * prio_data);
int starpu_sched_node_is_prio(struct starpu_sched_node * node);

struct starpu_sched_node * starpu_sched_node_work_stealing_create(void * arg STARPU_ATTRIBUTE_UNUSED);
int starpu_sched_node_is_work_stealing(struct starpu_sched_node * node);
int starpu_sched_tree_work_stealing_push_task(struct starpu_task *task);

struct starpu_sched_node * starpu_sched_node_random_create(void * arg STARPU_ATTRIBUTE_UNUSED);
int starpu_sched_node_is_random(struct starpu_sched_node *);

struct starpu_sched_node * starpu_sched_node_eager_create(void * arg STARPU_ATTRIBUTE_UNUSED);
int starpu_sched_node_is_eager(struct starpu_sched_node *);


struct starpu_mct_data
{
	double alpha;
	double beta;
	double gamma;
	double idle_power;
};

/* create a node with mct_data paremeters
   a copy the struct starpu_mct_data * given is performed during the init_data call
   the mct node doesnt do anything but pushing tasks on no_perf_model_node and calibrating_node
*/
struct starpu_sched_node * starpu_sched_node_mct_create(struct starpu_mct_data * mct_data);

int starpu_sched_node_is_mct(struct starpu_sched_node * node);

struct starpu_sched_node * starpu_sched_node_heft_create(struct starpu_mct_data * mct_data);

int starpu_sched_node_is_heft(struct starpu_sched_node * node);

/* this node select the best implementation for the first worker in context that can execute task.
 * and fill task->predicted and task->predicted_transfer
 * cannot have several childs if push_task is called
 */
struct starpu_sched_node * starpu_sched_node_best_implementation_create(void * arg STARPU_ATTRIBUTE_UNUSED);

struct starpu_perfmodel_select_data
{
	struct starpu_sched_node * calibrator_node;
	struct starpu_sched_node * no_perfmodel_node;
	struct starpu_sched_node * perfmodel_node;
};

struct starpu_sched_node * starpu_sched_node_perfmodel_select_create(struct starpu_perfmodel_select_data * perfmodel_select_data);
int starpu_sched_node_is_perfmodel_select(struct starpu_sched_node * node);
int starpu_sched_node_perfmodel_select_room(struct starpu_sched_node * node, unsigned sched_ctx_id);

/*create an empty tree
 */
struct starpu_sched_tree * starpu_sched_tree_create(unsigned sched_ctx_id);
void starpu_sched_tree_destroy(struct starpu_sched_tree * tree);

/* destroy node and all his child
 * except if they are shared between several contexts
 */
void starpu_sched_node_destroy_rec(struct starpu_sched_node * node, unsigned sched_ctx_id);

/* update all the node->workers member recursively
 */
void starpu_sched_tree_update_workers(struct starpu_sched_tree * t);
/* idem for workers_in_ctx 
 */
void starpu_sched_tree_update_workers_in_ctx(struct starpu_sched_tree * t);
/* wake up one underlaying workers of node which can execute the task
 */
void starpu_sched_node_wake_available_worker(struct starpu_sched_node * node, struct starpu_task * task );
/* wake up underlaying workers of node
 */
void starpu_sched_node_available(struct starpu_sched_node * node);

int starpu_sched_tree_push_task(struct starpu_task * task);
struct starpu_task * starpu_sched_tree_pop_task(unsigned sched_ctx_id);
void starpu_sched_tree_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);
void starpu_sched_tree_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers);
void starpu_sched_node_worker_pre_exec_hook(struct starpu_task * task);
void starpu_sched_node_worker_post_exec_hook(struct starpu_task * task);



struct starpu_sched_node_composed_recipe;

/* create empty recipe */
struct starpu_sched_node_composed_recipe * starpu_sched_node_create_recipe(void);
struct starpu_sched_node_composed_recipe * starpu_sched_node_create_recipe_singleton(struct starpu_sched_node *(*create_node)(void * arg), void * arg);

/* add a function creation node to recipe */
void starpu_sched_recipe_add_node(struct starpu_sched_node_composed_recipe * recipe, struct starpu_sched_node *(*create_node)(void * arg), void * arg);

void starpu_destroy_composed_sched_node_recipe(struct starpu_sched_node_composed_recipe *);
struct starpu_sched_node * starpu_sched_node_composed_node_create(struct starpu_sched_node_composed_recipe * recipe);


#ifdef STARPU_HAVE_HWLOC
/* null pointer mean to ignore a level L of hierarchy, then nodes of levels > L become childs of level L - 1 */
struct starpu_sched_specs
{
	/* hw_loc_machine_composed_sched_node must be set as its the root of the topology */
	struct starpu_sched_node_composed_recipe * hwloc_machine_composed_sched_node;
	struct starpu_sched_node_composed_recipe * hwloc_node_composed_sched_node;
	struct starpu_sched_node_composed_recipe * hwloc_socket_composed_sched_node;
	struct starpu_sched_node_composed_recipe * hwloc_cache_composed_sched_node;

	/* this member should return a new allocated starpu_sched_node_composed_recipe or NULL
	 * the starpu_sched_node_composed_recipe_t must not include the worker node
	 */
	struct starpu_sched_node_composed_recipe * (*worker_composed_sched_node)(enum starpu_worker_archtype archtype);
 
	/* this flag indicate if heterogenous workers should be brothers or cousins,
	 * as example, if a gpu and a cpu should share or not there numa node
	 */
	int mix_heterogeneous_workers;
};

struct starpu_sched_tree * starpu_sched_node_make_scheduler(unsigned sched_ctx_id, struct starpu_sched_specs);
#endif /* STARPU_HAVE_HWLOC */


#endif
