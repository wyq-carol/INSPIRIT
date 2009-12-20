/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#include <starpu_mpi_datatype.h>

typedef int (*handle_to_datatype_func)(starpu_data_handle, MPI_Datatype *);

static int handle_to_datatype_vector(starpu_data_handle data_handle, MPI_Datatype *datatype)
{
	unsigned nx = starpu_get_vector_nx(data_handle);
	size_t elemsize = starpu_get_vector_elemsize(data_handle);

	MPI_Type_contiguous(nx*elemsize, MPI_BYTE, datatype);
	MPI_Type_commit(datatype);

	return 0;
}

static handle_to_datatype_func handle_to_datatype_funcs[STARPU_NINTERFACES_ID] = {
	[STARPU_BLAS_INTERFACE_ID]	= NULL,
	[STARPU_BLOCK_INTERFACE_ID]	= NULL,
	[STARPU_VECTOR_INTERFACE_ID]	= handle_to_datatype_vector,
	[STARPU_CSR_INTERFACE_ID]	= NULL,
	[STARPU_CSC_INTERFACE_ID]	= NULL,
	[STARPU_BCSCR_INTERFACE_ID]	= NULL
};

int starpu_mpi_handle_to_datatype(starpu_data_handle data_handle, MPI_Datatype *datatype)
{
	unsigned id = starpu_get_handle_interface_id(data_handle);

	handle_to_datatype_func func = handle_to_datatype_funcs[id];

	STARPU_ASSERT(func);

	return func(data_handle, datatype);
}
