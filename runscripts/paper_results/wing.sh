#!/bin/bash

# Replace 'PATH_TO_SNBO_FOLDER' with the path to your SNBO project
export PYTHONPATH=PATH_TO_SNBO_FOLDER:$PYTHONPATH

# Set following parameters as required
method="snbo" # valid values: "bo", "ibnn", "turbo", "dycors", "snbo"

# other variables
problem="wing"
n_init=150
n_runs=5
max_evals=500
dim=119

for i in $(seq 1 $n_runs);
do
    python algorithms/optimize.py --method $method --problem $problem --dim $dim --n_init $n_init --max_evals $max_evals --seed $i\
        --neurons 128 128 128 --act_funcs "GELU" "GELU" "GELU"
done
