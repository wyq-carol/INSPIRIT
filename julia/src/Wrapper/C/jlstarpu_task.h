/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018                                     Alexis Juven
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
 * jlstarpu_task.h
 *
 *  Created on: 27 juin 2018
 *      Author: ajuven
 */

#ifndef JLSTARPU_TASK_H_
#define JLSTARPU_TASK_H_


#include "jlstarpu.h"






struct jlstarpu_codelet
{
	uint32_t where;

	starpu_cpu_func_t cpu_func;
	char * cpu_func_name;

	starpu_cuda_func_t cuda_func;

	int nbuffer;
	enum starpu_data_access_mode * modes;

	struct starpu_perfmodel * model;

};



struct jlstarpu_task
{
	struct starpu_codelet * cl;
	starpu_data_handle_t * handles;
	unsigned int synchronous;

	void * cl_arg;
	size_t cl_arg_size;
};


#if 0

struct cl_args_decorator
{
	struct jlstarpu_function_launcher * launcher;
	void * cl_args;
};

#endif





#endif /* JLSTARPU_TASK_H_ */