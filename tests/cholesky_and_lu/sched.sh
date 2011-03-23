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

ns=10
nsamples=0
sampleList_cholesky=""
sampleList_lu=""
sampleList_all=""

filename=$TIMINGDIR/sched
    
for blocks in `seq 10 2 24`
do
    size=$(($blocks*1024))
    
    echo "size : $size"
    
    OPTIONS="-pin -nblocks $blocks -size $size"

    echo "$ROOTDIR/examples/cholesky_and_lu/cholesky_and_lu $OPTIONS 2> /dev/null"

    for s in `seq 1 $ns`
    do
	val=`$ROOTDIR/examples/cholesky_and_lu/cholesky_and_lu $OPTIONS 2> /dev/null`
	echo "val = $val"
	if [ "$val" != "" ];
	then
	    nsamples=$(echo "$nsamples + 1"|bc -l)
	fi

	echo "$nsamples"

	sampleRes=(`echo $val | tr " " "\n"`)

        sampleList_cholesky="$sampleList_cholesky ${sampleRes[0]}"
	sampleList_lu="$sampleList_cholesky ${sampleRes[1]}"
	sampleList_all="$sampleList_cholesky ${sampleRes[2]}"
	
	sum_cholesky=0
	sum_lu=0
	sum_all=0
	
	for val in $sampleList_cholesky
	do
	    sum_cholesky=$(echo "$sum_cholesky + $val"|bc -l)
	done
	
	for val in $sampleList_lu
	do
	    sum_lu=$(echo "$sum_lu + $val"|bc -l)
	done
	
	for val in $sampleList_all
	do
	    sum_all=$(echo "$sum_all + $val"|bc -l)
	done
    done
    
    if [ "nsamples" != "0" ];
    then
        avg_cholesky=$(echo "$sum_cholesky / $nsamples"|bc -l)
        avg_lu=$(echo "$sum_lu / $nsamples"|bc -l)
        avg_all=$(echo "$sum_all / $nsamples"|bc -l)
	
        orderedsampleList_cholesky=$(echo "$sampleList_cholesky"|tr " " "\n" |sort -n)
        orderedsampleList_lu=$(echo "$sampleList_lu"|tr " " "\n" |sort -n)
        orderedsampleList_all=$(echo "$sampleList_all"|tr " " "\n" |sort -n)
	
        ylow_cholesky=$(echo $orderedsampleList_cholesky | awk '{print $1}')
        yhigh_cholesky=$(echo "$orderedsampleList_cholesky"|tail -1)

	echo "ylow_cholesky = $ylow_cholesky"
	echo "yhigh_cholesky = $yhigh_cholesky"

        ylow_lu=$(echo $orderedsampleList_lu | awk '{print $1}')
        yhigh_lu=$(echo "$orderedsampleList_lu"|tail -1)

	echo "ylow_lu = $ylow_lu"
	echo "yhigh_lu = $yhigh_lu"

        ylow_all=$(echo $orderedsampleList_all | awk '{print $1}')
        yhigh_all=$(echo "$orderedsampleList_all"|tail -1)
		
	echo "ylow_all = $ylow_all"
	echo "yhigh_all = $yhigh_all"
	
        echo "$size $avg_cholesky $ylow_cholesky $yhigh_cholesky" >> $filename.cholesky
        echo "$size $avg_lu $ylow_lu $yhigh_lu" >> $filename.lu
        echo "$size $avg_all $ylow_all $yhigh_all" >> $filename.all

	nsamples=0
    fi
done





