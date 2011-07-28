#!/bin/bash

# StarPU --- Runtime system for heterogeneous multicore architectures.
# 
# Copyright (C) 2009, 2010  Université de Bordeaux 1
# Copyright (C) 2010  Centre National de la Recherche Scientifique
# 
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
# 
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# 
# See the GNU Lesser General Public License in COPYING.LGPL for more details.

BENCH_NAME=cholesky_and_lu

filename1=/home/ahugo/sched_ctx/tests/cholesky_and_lu/timings-sched/cholesky_and_lu
filename2=/home/ahugo/trunk2/tests/cholesky_and_lu/timings-sched/cholesky_and_lu

gnuplot > /dev/null << EOF
set term png enhanced color
set output "$BENCH_NAME_big_kernel.png"

set datafile missing 'x'

set pointsize 0.75
set title "Kernel Cholesky - 60 blocks size 61440 - in presence of another kernel cholesky - 40 blocks size 4096"
set grid y
set grid x
set xrange [0:100]
#set logscale x
#set xtics 8192,8192,65536
#set key invert box right bottom title "Scheduling policy"
#set size 0.65

set xlabel "Number of CPUs"
set ylabel "GFlop/s"

plot "$filename1" using 3:5 title 'No context' with lines lt 3 lw 2, "$filename2" using 3:5 title '2 contexts' with lines lt 2 lw 2

EOF

gnuplot > /dev/null << EOF
set term png enhanced color
set output "$BENCH_NAME_small_kernel.png"

set datafile missing 'x'

set pointsize 0.75
set title "Kernel Cholesky - 40 blocks size 4096 - in presence of another kernel cholesky - 60 blocks size 61440"
set grid y
set grid x
set xrange [0:100]
#set logscale x
#set xtics 8192,8192,65536
#set key invert box right bottom title "Scheduling policy"
#set size 0.65

set xlabel "Number of CPUs"
set ylabel "GFlop/s"

plot "$filename1" using 4:6 title 'No context' with lines lt 3 lw 2, "$filename2" using 4:6 title '2 contexts' with lines lt 2 lw 2

EOF