/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2012, 2014, 2017  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012  CNRS
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

#include <starpu.h>
#include <starpu_cuda.h>
#include <common/config.h>
#include <core/workers.h>

#ifdef STARPU_USE_CUDA
#include <cublas.h>

static int cublas_initialized[STARPU_NMAXWORKERS];

static unsigned get_idx(void) {
	unsigned workerid = starpu_worker_get_id_check();
	unsigned th_per_dev = _starpu_get_machine_config()->topology.cuda_th_per_dev;
	unsigned th_per_stream = _starpu_get_machine_config()->topology.cuda_th_per_stream;

	if (th_per_dev)
		return starpu_worker_get_devid(workerid);
	else if (th_per_stream)
		return workerid;
	else
		/* same thread for all devices */
		return 0;
}

static void init_cublas_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	unsigned idx = get_idx();
	if (STARPU_ATOMIC_ADD(&cublas_initialized[idx], 1) == 1)
	{
		cublasStatus cublasst = cublasInit();
		if (STARPU_UNLIKELY(cublasst))
			STARPU_CUBLAS_REPORT_ERROR(cublasst);
	}
}

static void set_cublas_stream_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	cublasSetKernelStream(starpu_cuda_get_local_stream());
}

static void shutdown_cublas_func(void *args STARPU_ATTRIBUTE_UNUSED)
{
	unsigned idx = get_idx();
	if (STARPU_ATOMIC_ADD(&cublas_initialized[idx], -1) == 0)
		cublasShutdown();
}
#endif

void starpu_cublas_init(void)
{
#ifdef STARPU_USE_CUDA
	starpu_execute_on_each_worker(init_cublas_func, NULL, STARPU_CUDA);
	starpu_execute_on_each_worker(set_cublas_stream_func, NULL, STARPU_CUDA);
#endif
}

void starpu_cublas_shutdown(void)
{
#ifdef STARPU_USE_CUDA
	starpu_execute_on_each_worker(shutdown_cublas_func, NULL, STARPU_CUDA);
#endif
}

void starpu_cublas_set_stream(void)
{
#ifdef STARPU_USE_CUDA
	if (!_starpu_get_machine_config()->topology.cuda_th_per_dev ||
		(!_starpu_get_machine_config()->topology.cuda_th_per_stream &&
		 _starpu_get_machine_config()->topology.nworkerpercuda > 1))
		cublasSetKernelStream(starpu_cuda_get_local_stream());
#endif
}
