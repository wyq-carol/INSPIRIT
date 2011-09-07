#!/bin/bash                                                                     
#export STARPU_NCUDA=3
#export STARPU_NCPUS=9
#export STARPU_DIR=$HOME/sched_ctx/build

source sched.sh isole 0 0 3 
source sched.sh isole 0 1 2
source sched.sh isole 0 2 1
source sched.sh isole 0 3 0   

source sched.sh 1gpu 1 0 2
source sched.sh 1gpu 1 1 1
source sched.sh 1gpu 1 2 0

source sched.sh 2gpu 2 1 0
source sched.sh 2gpu 2 0 1

source sched.sh 3gpu 3 0 0
