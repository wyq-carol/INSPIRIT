#!/bin/bash

# StarPU --- Runtime system for heterogeneous multicore architectures.
# 
# Copyright (C) 2008, 2009, 2010  Université de Bordeaux 1
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


DIR=$PWD
ROOTDIR=$DIR/../..
TIMINGDIR=$DIR/timings-sched/
mkdir -p $TIMINGDIR
BENCH_NAME=cholesky_and_lu
ns=10

filename=$TIMINGDIR/$BENCH_NAME
    
for blocks in `seq 10 2 24`
do
    size=$(($blocks*1024))
    
    echo "size : $size"
    
    OPTIONS="-pin -nblocks $blocks -size $size"

    echo "$ROOTDIR/examples/$BENCH_NAME/$BENCH_NAME $OPTIONS"

    val=`STARPU_NCUDA=2 $ROOTDIR/examples/$BENCH_NAME/$BENCH_NAME $OPTIONS`

    echo "$size $val"
    echo "$size $val" >> $filename
done





