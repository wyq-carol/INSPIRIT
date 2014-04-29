/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2012, 2014  Université de Bordeaux 1
 * Copyright (C) 2010, 2012, 2013, 2014  Centre National de la Recherche Scientifique
 * Copyright (C) 2014  Inria
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

#ifndef __DATA_INTERFACE_H__
#define __DATA_INTERFACE_H__

#include <starpu.h>
#include <common/config.h>
#include <common/uthash.h>
#ifdef STARPU_OPENMP
#include <util/openmp_runtime_support.h>
#endif

/* Entry in the `registered_handles' hash table.  */
struct handle_entry
{
	UT_hash_handle hh;
	void *pointer;
	starpu_data_handle_t handle;
};

/* Generic type representing an interface, for now it's only used before
 * execution on message-passing devices but it can be useful in other cases.
 */
union _starpu_interface
{
	struct starpu_matrix_interface matrix;
	struct starpu_block_interface block;
	struct starpu_vector_interface vector;
	struct starpu_csr_interface csr;
	struct starpu_coo_interface coo;
	struct starpu_bcsr_interface bcsr;
	struct starpu_variable_interface variable;
	struct starpu_multiformat_interface multiformat;
	struct starpu_disk_interface disk;
};

/* Some data interfaces or filters use this interface internally */
extern struct starpu_data_interface_ops starpu_interface_matrix_ops;
extern struct starpu_data_interface_ops starpu_interface_block_ops;
extern struct starpu_data_interface_ops starpu_interface_vector_ops;
extern struct starpu_data_interface_ops starpu_interface_csr_ops;
extern struct starpu_data_interface_ops starpu_interface_bcsr_ops;
extern struct starpu_data_interface_ops starpu_interface_variable_ops;
extern struct starpu_data_interface_ops starpu_interface_void_ops;
extern struct starpu_data_interface_ops starpu_interface_multiformat_ops;

void _starpu_data_free_interfaces(starpu_data_handle_t handle)
	STARPU_ATTRIBUTE_INTERNAL;

extern
int _starpu_data_handle_init(starpu_data_handle_t handle, struct starpu_data_interface_ops *interface_ops, unsigned int mf_node);

extern void _starpu_data_interface_init(void) STARPU_ATTRIBUTE_INTERNAL;
extern int _starpu_data_check_not_busy(starpu_data_handle_t handle) STARPU_ATTRIBUTE_INTERNAL;
extern void _starpu_data_interface_shutdown(void) STARPU_ATTRIBUTE_INTERNAL;

#ifdef STARPU_OPENMP
void _starpu_omp_unregister_region_handles(struct starpu_omp_region *region);
void _starpu_omp_unregister_task_handles(struct starpu_omp_task *task);
#endif

struct starpu_data_interface_ops *_starpu_data_interface_get_ops(unsigned interface_id);

extern void _starpu_data_register_ram_pointer(starpu_data_handle_t handle,
						void *ptr)
	STARPU_ATTRIBUTE_INTERNAL;

extern void _starpu_data_unregister_ram_pointer(starpu_data_handle_t handle)
	STARPU_ATTRIBUTE_INTERNAL;

extern int _starpu_data_is_multiformat_handle(starpu_data_handle_t handle);
extern starpu_data_handle_t _starpu_data_get_data_handle_from_tag(int tag);

extern int _starpu_data_set_rank(starpu_data_handle_t handle, int rank);
extern int _starpu_data_set_tag(starpu_data_handle_t handle, int tag);

#endif // __DATA_INTERFACE_H__
