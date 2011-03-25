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
    
    nsamples=0
    sampleList=""

    sum=0

    OPTIONS="-pin -nblocks $blocks -size $size"

    echo "$ROOTDIR/examples/$BENCH_NAME/$BENCH_NAME $OPTIONS"

    for s in `seq 1 $ns`
    do
	val=`$ROOTDIR/examples/$BENCH_NAME/$BENCH_NAME $OPTIONS`
	echo "val = $val"
	if [ "$val" != "" ];
	then
	    nsamples=$(echo "$nsamples + 1"|bc -l)
	fi

	echo "$nsamples"

	sampleList="$sampleList $val"
    done

    for val in $sampleList
    do
	sum=$(echo "$sum + $val"|bc -l)
    done
    
    if [ "nsamples" != "0" ];
    then
        avg=$(echo "$sum / $nsamples"|bc -l)
	
        orderedsampleList=$(echo "$sampleList"|tr " " "\n" |sort -n)
	
        ylow=$(echo $orderedsampleList | awk '{print $1}')
        yhigh=$(echo "$orderedsampleList"|tail -1)
		
	echo "ylow = $ylow"
	echo "yhigh = $yhigh"
	
        echo "$size $avg $ylow $yhigh" >> $filename
    fi
done





