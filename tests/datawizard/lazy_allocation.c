/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <starpu_cuda.h>
#include <stdlib.h>
#include "../common/helper.h"

#define VECTORSIZE	1024

static starpu_data_handle v_handle;

/*
 *	Memset
 */

#ifdef STARPU_USE_CUDA
static void cuda_memset_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	cudaMemsetAsync(buf, 42, length, starpu_cuda_get_local_stream());
	cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
#endif

static void cpu_memset_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	memset(buf, 42, length);
}

static starpu_codelet memset_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = cpu_memset_codelet,
#ifdef STARPU_USE_CUDA
	.cuda_func = cuda_memset_codelet,
#endif
	.nbuffers = 1
};

/*
 *	Check content
 */

static void cpu_check_content_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	unsigned i;
	for (i = 0; i < length; i++)
	{
		if (buf[i] != 42)
		{
			FPRINTF(stderr, "buf[%u] is %c while it should be %c\n", i, buf[i], 42);
			exit(-1);
		}
	}
}

#ifdef STARPU_USE_CUDA
static void cuda_check_content_codelet(void *descr[], __attribute__ ((unused)) void *_args)
{
	char *buf = (char *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned length = STARPU_VECTOR_GET_NX(descr[0]);

	unsigned i;
	for (i = 0; i < length; i++)
	{
		char dst;
		cudaMemcpy(&dst, &buf[i], sizeof(char), cudaMemcpyDeviceToHost);
		if (dst != 42)
		{
			FPRINTF(stderr, "buf[%u] is %c while it should be %c\n", i, dst, 42);
			exit(-1);
		}
	}
}
#endif

static starpu_codelet check_content_cl = {
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = cpu_check_content_codelet,
#ifdef STARPU_USE_CUDA
	.cuda_func = cuda_check_content_codelet,
#endif
	.nbuffers = 1
};


int main(int argc, char **argv)
{
	int ret;

	ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_vector_data_register(&v_handle, (uint32_t)-1, (uintptr_t)NULL, VECTORSIZE, sizeof(char));

	ret = starpu_insert_task(&memset_cl, STARPU_W, v_handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_insert_task");

        ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	ret = starpu_insert_task(&check_content_cl, STARPU_R, v_handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_insert_task");

        ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_data_unregister(v_handle);

	starpu_shutdown();
	return 0;
}
