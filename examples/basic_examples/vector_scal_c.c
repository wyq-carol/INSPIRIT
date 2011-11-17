/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Université de Bordeaux 1
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

/*
 * This example demonstrates how to use StarPU to scale an array by a factor.
 * It shows how to manipulate data with StarPU's data management library.
 *  1- how to declare a piece of data to StarPU (starpu_vector_data_register)
 *  2- how to describe which data are accessed by a task (task->buffers[0])
 *  3- how a kernel can manipulate the data (buffers[0].vector.ptr)
 *
 * This is a variant of vector_scal.c which shows it can be integrated with fortran.
 */

#include <starpu.h>
#include <starpu_opencl.h>
#include <stdio.h>


extern void scal_cpu_func(void *buffers[], void *_args);
extern void scal_cuda_func(void *buffers[], void *_args);

static struct starpu_perfmodel vector_scal_model = {
	.type = STARPU_HISTORY_BASED,
	.symbol = "vector_scale_model"
};

static starpu_codelet cl = {
  .where = STARPU_CPU | STARPU_CUDA,
	/* CPU implementation of the codelet */
	.cpu_func = scal_cpu_func,
#ifdef STARPU_USE_CUDA
	/* CUDA implementation of the codelet */
	.cuda_func = scal_cuda_func,
#endif
	.nbuffers = 1,
	.model = &vector_scal_model
};

void compute_(int *F_NX, float *vector)
{
        int NX = *F_NX;
	
	/* Initialize StarPU with default configuration */
	starpu_init(NULL);

	/* Tell StaPU to associate the "vector" vector with the "vector_handle"
	 * identifier. When a task needs to access a piece of data, it should
	 * refer to the handle that is associated to it.
	 * In the case of the "vector" data interface:
	 *  - the first argument of the registration method is a pointer to the
	 *    handle that should describe the data
	 *  - the second argument is the memory node where the data (ie. "vector")
	 *    resides initially: 0 stands for an address in main memory, as
	 *    opposed to an adress on a GPU for instance.
	 *  - the third argument is the adress of the vector in RAM
	 *  - the fourth argument is the number of elements in the vector
	 *  - the fifth argument is the size of each element.
	 */
	starpu_data_handle vector_handle;
	starpu_vector_data_register(&vector_handle, 0, (uintptr_t)vector, NX, sizeof(vector[0]));

	float factor = 3.14;

	/* create a synchronous task: any call to starpu_task_submit will block
 	 * until it is terminated */
	struct starpu_task *task = starpu_task_create();
	task->synchronous = 1;

	task->cl = &cl;

	/* the codelet manipulates one buffer in RW mode */
	task->buffers[0].handle = vector_handle;
	task->buffers[0].mode = STARPU_RW;

	/* an argument is passed to the codelet, beware that this is a
	 * READ-ONLY buffer and that the codelet may be given a pointer to a
	 * COPY of the argument */
	task->cl_arg = &factor;
	task->cl_arg_size = sizeof(factor);

	/* execute the task on any eligible computational ressource */
	starpu_task_submit(task);

	/* StarPU does not need to manipulate the array anymore so we can stop
 	 * monitoring it */
	starpu_data_unregister(vector_handle);

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();
}
