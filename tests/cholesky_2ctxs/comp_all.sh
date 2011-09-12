#!/bin/bash

compute_effic=$1
#for one matrix 20000 x 20000 and one of 10000 x 10000
ninstr=2999999987712
no_ctx_prefix=../cholesky_no_ctxs/timings-sched
ctx_prefix=timings-sched

source comp.sh $no_ctx_prefix/cholesky_no_ctxs res_cholesky_no_ctxs 0 $compute_effic $ninstr

bestval_noctx=0
while read line
do 
    results=($line)
    val=$(echo "${results[1]}"|bc -l)
    val=${val/.*}

    if [ $val -gt $bestval_noctx ]
    then
	bestval_noctx=$(echo "$val"|bc -l)
    fi
done < res_cholesky_no_ctxs

echo $bestval_noctx

source comp.sh $ctx_prefix/isole res_isole 1 $compute_effic $ninstr $bestval_noctx

#compute efficiency in a heterogeneous system
#for the homogeneous one we can compute gflops rate per PU

if [ $compute_effic -eq 1 ]
then
    source comp.sh $ctx_prefix/1gpu res_1gpu 1 $compute_effic $ninstr $bestval_noctx
    source comp.sh $ctx_prefix/2gpu res_2gpu  1 $compute_effic $ninstr $bestval_noctx
    source comp.sh $ctx_prefix/3gpu res_3gpu 1 $compute_effic $ninstr $bestval_noctx

    source gnuplot_efficiency.sh efficiency
else
    source gnuplot_gflopsrate.sh gflopsrate
fi