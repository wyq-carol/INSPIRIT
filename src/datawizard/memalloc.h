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

#ifndef __MEMALLOC_H__
#define __MEMALLOC_H__

#include <starpu.h>
#include <common/config.h>

#include <common/list.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/coherency.h>
#include <datawizard/copy_driver.h>

struct starpu_data_replicate_s;

LIST_TYPE(starpu_mem_chunk,
	starpu_data_handle data;
	size_t size;

	uint32_t footprint;
	
	/* The footprint of the data is not sufficient to determine whether two
	 * pieces of data have the same layout (there could be collision in the
	 * hash function ...) so we still keep a copy of the actual layout (ie.
	 * the data interface) to stay on the safe side. We make a copy of
	 * because when a data is deleted, the memory chunk remains.
	 */
	struct starpu_data_interface_ops_t *ops;
	void *interface;
	unsigned automatically_allocated;
	unsigned data_was_deleted;

	/* A buffer that is used for SCRATCH or reduction cannnot be used with
	 * filters. */
	unsigned relaxed_coherency;
	struct starpu_data_replicate_s *replicate;
)

void _starpu_init_mem_chunk_lists(void);
void _starpu_deinit_mem_chunk_lists(void);
void _starpu_request_mem_chunk_removal(starpu_data_handle handle, unsigned node);
int _starpu_allocate_memory_on_node(starpu_data_handle handle, struct starpu_data_replicate_s *replicate);
size_t _starpu_free_all_automatically_allocated_buffers(uint32_t node);
#endif
