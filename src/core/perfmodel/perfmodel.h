/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2014  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
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

#ifndef __PERFMODEL_H__
#define __PERFMODEL_H__

#include <common/config.h>
#include <starpu.h>
#include <core/task_bundle.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Performance models files are stored in a directory whose name
 * include the version of the performance model format. The version
 * number is also written in the file itself.
 * When updating the format, the variable _STARPU_PERFMODEL_VERSION
 * should be updated. It is then possible to switch easily between
 * differents versions of StarPU having different performance model
 * formats.
 */
#define _STARPU_PERFMODEL_VERSION 44

struct _starpu_perfmodel_state
{
	struct starpu_perfmodel_per_arch** per_arch; /*STARPU_MAXIMPLEMENTATIONS*/
	int** per_arch_is_set; /*STARPU_MAXIMPLEMENTATIONS*/

	starpu_pthread_rwlock_t model_rwlock;
	int *nimpls;
	int *nimpls_set;
	int ncombs;  /* The number of combinations currently used by the model */
	int ncombs_set; /* The number of combinations allocated in the array nimpls and ncombs */
	int *combs;
};

struct _starpu_perfmodel_list
{
	struct _starpu_perfmodel_list *next;
	struct starpu_perfmodel *model;
};

struct starpu_data_descr;
struct _starpu_job;
struct starpu_perfmodel_arch;

void _starpu_get_perf_model_dir(char *path, size_t maxlen);
void _starpu_get_perf_model_dir_codelets(char *path, size_t maxlen);
void _starpu_get_perf_model_dir_bus(char *path, size_t maxlen);
void _starpu_get_perf_model_dir_debug(char *path, size_t maxlen);

double _starpu_history_based_job_expected_perf(struct starpu_perfmodel *model, struct starpu_perfmodel_arch* arch, struct _starpu_job *j, unsigned nimpl);
void _starpu_load_per_arch_based_model(struct starpu_perfmodel *model);
void _starpu_load_common_based_model(struct starpu_perfmodel *model);
void _starpu_load_history_based_model(struct starpu_perfmodel *model, unsigned scan_history);
void _starpu_init_and_load_perfmodel(struct starpu_perfmodel *model);
void _starpu_initialize_registered_performance_models(void);
void _starpu_deinitialize_registered_performance_models(void);
void _starpu_deinitialize_performance_model(struct starpu_perfmodel *model);

double _starpu_regression_based_job_expected_perf(struct starpu_perfmodel *model,
					struct starpu_perfmodel_arch* arch, struct _starpu_job *j, unsigned nimpl);
double _starpu_non_linear_regression_based_job_expected_perf(struct starpu_perfmodel *model,
					struct starpu_perfmodel_arch* arch, struct _starpu_job *j, unsigned nimpl);
void _starpu_update_perfmodel_history(struct _starpu_job *j, struct starpu_perfmodel *model, struct starpu_perfmodel_arch * arch,
				unsigned cpuid, double measured, unsigned nimpl);
int _starpu_perfmodel_create_comb_if_needed(struct starpu_perfmodel_arch* arch);

void _starpu_create_sampling_directory_if_needed(void);

void _starpu_load_bus_performance_files(void);

void _starpu_set_calibrate_flag(unsigned val);
unsigned _starpu_get_calibrate_flag(void);

#if defined(STARPU_USE_CUDA)
int *_starpu_get_cuda_affinity_vector(unsigned gpuid);
#endif
#if defined(STARPU_USE_OPENCL)
int *_starpu_get_opencl_affinity_vector(unsigned gpuid);
#endif

void _starpu_save_bandwidth_and_latency_disk(double bandwidth_write, double bandwidth_read, 
					    double latency_write, double latency_read, unsigned node);

int _starpu_read_double(FILE *f, char *format, double *val);
void _starpu_simgrid_get_platform_path(char *path, size_t maxlen);

struct starpu_perfmodel_arch * _starpu_arch_comb_get(int comb);

void _starpu_perfmodel_realloc(struct starpu_perfmodel *model, int nb);

#ifdef __cplusplus
}
#endif

#endif // __PERFMODEL_H__
