#!/bin/bash

# Replace 'PATH_TO_SNBO_FOLDER' with the path to your SNBO project
export PYTHONPATH=PATH_TO_SNBO_FOLDER:$PYTHONPATH

# variables
method="snbo"
problem="rover"
dim=100
n_init=$((2*$dim))
max_evals=2000
n_runs=10

for i in $(seq 1 $n_runs);
do
    python algorithms/optimize.py --method $method --problem $problem --dim $dim --n_init $n_init --max_evals $max_evals --seed $i\
        --neurons 256 256 --act_funcs "GELU" "GELU"
done
