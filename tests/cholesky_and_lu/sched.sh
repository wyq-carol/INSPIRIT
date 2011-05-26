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
#mkdir -p $TIMINGDIR
BENCH_NAME=cholesky_and_lu
nsamples=5

filename=$TIMINGDIR/$BENCH_NAME

nmaxcpus=96
nmincpus1=40
nmincpus2=30

blocks1=60
blocks2=40

size1=$(($blocks1*1024))
size2=$(($blocks2*1024))

for i in `seq $nmincpus1 2 $(($nmaxcpus-1))`
do
    if [ $i -gt $(($nmaxcpus-$nmincpus2)) ]
    then
	break
    fi

    ncpus1=$i
    ncpus2=$(($nmaxcpus-$i))    
    
    OPTIONS="-pin -nblocks $blocks1 -size $size1 -nblocks $blocks2 -size $size2 -ncpus1 $ncpus1 -ncpus2 $ncpus2"

    gflops1_avg=0
    gflops2_avg=0

    t1_avg=0
    t2_avg=0
    t_total_avg=0

    for s in `seq 1 $nsamples`
    do
	echo "$ROOTDIR/examples/$BENCH_NAME/$BENCH_NAME $OPTIONS"
	
	val=`$ROOTDIR/examples/$BENCH_NAME/$BENCH_NAME $OPTIONS`

	echo "$val"

	val=`echo $val|tr " " "\n"`
	
	i=0
	for x in $val
	do
	    if [ $i -eq 0 ]
	    then
		gflops1_avg=$(echo "$gflops1_avg + $x"|bc -l)
	    fi
	    if [ $i -eq 1 ]
	    then
		gflops2_avg=$(echo "$gflops2_avg+$x"|bc -l)
	    fi
	    if [ $i -eq 2 ]
	    then
		t1_avg=$(echo "$t1_avg+$x"|bc -l)
	    fi
	    
	    if [ $i -eq 3 ]
	    then
		t2_avg=$(echo "$t2_avg+$x"|bc -l)
	    fi

	    if [ $i -eq 4 ]
	    then
		t_total_avg=$(echo "$t_total_avg+$x"|bc -l)
	    fi
	    i=$(($i+1))
	done
	
    done

    gflops1_avg=$(echo "$gflops1_avg / $nsamples"|bc -l)
    gflops2_avg=$(echo "$gflops2_avg / $nsamples"|bc -l)
    t1_avg=$(echo "$t1_avg / $nsamples"|bc -l)
    t2_avg=$(echo "$t2_avg / $nsamples"|bc -l)
    t_total_avg=$(echo "$t_total_avg / $nsamples"|bc -l)

    echo "$ncpus1 $ncpus2 `printf '%2.2f %2.2f %2.2f %2.2f %2.2f' $gflops1_avg $gflops2_avg $t1_avg $t2_avg $t_total_avg`"
    echo "$ncpus1 $ncpus2 `printf '%2.2f %2.2f %2.2f %2.2f %2.2f' $gflops1_avg $gflops2_avg $t1_avg $t2_avg $t_total_avg`" >> $filename

done


