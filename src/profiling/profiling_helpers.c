/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2013, 2016  Université de Bordeaux
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
#include <starpu_profiling.h>
#include <profiling/profiling.h>
#include <datawizard/memory_nodes.h>

static double convert_to_byte_units(float d, unsigned max_unit, unsigned *unit)
{
	const double divisor = 1024;

	*unit = 0;
	while (d > divisor && *unit < max_unit) {
		d /= divisor;
		(*unit)++;
	}
	return d;
}

static double convert_to_GB(float d)
{
	const double divisor = 1024;
	return d = (((d / divisor) / divisor) / divisor);
}

void _starpu_profiling_bus_helper_display_summary(FILE *stream)
{
	int long long sum_transferred = 0;
	const char *byte_units[] = { "B", "KB", "MB", "GB", "TB" };
	unsigned max_unit = sizeof(byte_units) / sizeof(byte_units[0]);

	fprintf(stream, "\n#---------------------\n");
	fprintf(stream, "Data transfer stats:\n");

	int busid;
	int bus_cnt = starpu_bus_get_count();
	for (busid = 0; busid < bus_cnt; busid++)
	{
		char src_name[128], dst_name[128];
		int src, dst;

		src = starpu_bus_get_src(busid);
		dst = starpu_bus_get_dst(busid);

		struct starpu_profiling_bus_info bus_info;
		starpu_bus_get_profiling_info(busid, &bus_info);

		int long long transferred = bus_info.transferred_bytes;
		int long long transfer_cnt =  bus_info.transfer_count;
		double elapsed_time = starpu_timing_timespec_to_us(&bus_info.total_time) / 1e6;

		/* Ne pas utiliser convert_to_byte_units */
		/* Hardcorder les unités: unit=3 "GB" */
		unsigned unit = 3;
		double d = convert_to_GB(transferred);
		/* double d = convert_to_byte_units(transferred, max_unit, &unit); */

		_starpu_memory_node_get_name(src, src_name, sizeof(src_name));
		_starpu_memory_node_get_name(dst, dst_name, sizeof(dst_name));

		fprintf(stream, "\t%s -> %s", src_name, dst_name);
		fprintf(stream, "\t%.4lf %s", d, byte_units[unit]);
		fprintf(stream, "\t%.4lf %s/s", (d * 1024) / elapsed_time, byte_units[unit - 1]); /* MB/s */
		fprintf(stream, "\t(transfers : %lld - avg %.4lf %s)\n", transfer_cnt, (d * 1024) / transfer_cnt, byte_units[unit - 1]); /* avg MB/s */

		sum_transferred += transferred;
	}
	
	/* Ne pas utiliser convert_to_byte_units */
	/* Hardcorder les unités: unit=3 "GB" */
	unsigned unit = 3;
	/* double d = convert_to_byte_units(sum_transferred, max_unit, &unit); */
	double d = convert_to_GB(sum_transferred);

	fprintf(stream, "Total transfers: %.4lf %s\n", d, byte_units[unit]);
	fprintf(stream, "#---------------------\n");
}

void starpu_profiling_bus_helper_display_summary(void)
{
	const char *stats;
	if (!((stats = starpu_getenv("STARPU_BUS_STATS")) && atoi(stats))) return;
	_starpu_profiling_bus_helper_display_summary(stderr);
}

void _starpu_profiling_worker_helper_display_summary(FILE *stream)
{
	double sum_consumed = 0.;
	int profiling = starpu_profiling_status_get();
	double overall_time = 0;
	int workerid;
	int worker_cnt = starpu_worker_get_count();

	fprintf(stream, "\n#---------------------\n");
	fprintf(stream, "Worker stats:\n");

	for (workerid = 0; workerid < worker_cnt; workerid++)
	{
		struct starpu_profiling_worker_info info;
		starpu_profiling_worker_get_info(workerid, &info);
		char name[64];

		starpu_worker_get_name(workerid, name, sizeof(name));

		fprintf(stream, "%-32s\n", name);
		fprintf(stream, "\t%d task(s)\n", info.executed_tasks);

		if (profiling)
		{
			double total_time = starpu_timing_timespec_to_us(&info.total_time) / 1000.;
			double executing_time = starpu_timing_timespec_to_us(&info.executing_time) / 1000.;
			double sleeping_time = starpu_timing_timespec_to_us(&info.sleeping_time) / 1000.;
			if (total_time > overall_time)
				overall_time = total_time;

			fprintf(stream, "\ttotal: %.2lf ms executing: %.2lf ms sleeping: %.2lf ms overhead %.2lf ms\n",
				total_time, executing_time, sleeping_time, total_time - executing_time - sleeping_time);
			if (info.used_cycles || info.stall_cycles)
				fprintf(stream, "\t%llu Mcy %llu Mcy stall\n", (unsigned long long)info.used_cycles/1000000, (unsigned long long)info.stall_cycles/1000000);
			if (info.energy_consumed)
				fprintf(stream, "\t%f J consumed\n", info.energy_consumed);
			if (info.flops)
				fprintf(stream, "\t%f GFlop/s\n\n", info.flops / total_time / 1000000);
		}

		sum_consumed += info.energy_consumed;
	}

	if (profiling)
	{
		const char *strval_idle_power = starpu_getenv("STARPU_IDLE_POWER");
		if (strval_idle_power)
		{
			double idle_power = atof(strval_idle_power); /* Watt */
			double idle_energy = idle_power * overall_time / 1000.; /* J */

			fprintf(stream, "Idle energy: %.2lf J\n", idle_energy);
			fprintf(stream, "Total energy: %.2lf J\n",
				sum_consumed + idle_energy);
		}
	}
	fprintf(stream, "#---------------------\n");
}

void starpu_profiling_worker_helper_display_summary(void)
{
	const char *stats;
	if (!((stats = starpu_getenv("STARPU_WORKER_STATS")) && atoi(stats))) return;
	_starpu_profiling_worker_helper_display_summary(stderr);
}
