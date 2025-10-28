#!/bin/bash

# Replace 'PATH_TO_SNBO_FOLDER' with the path to your SNBO project
export PYTHONPATH=PATH_TO_SNBO_FOLDER:$PYTHONPATH

# Set following parameters as required
method="snbo" # valid values: "bo", "ibnn", "turbo", "dycors", "snbo"
dim=102
max_evals=2000

# other variables
n_init=$((2*$dim))
problem="halfcheetah"
n_runs=10

for i in $(seq 1 $n_runs);
do
    python algorithms/optimize.py --method $method --problem $problem --dim $dim --n_init $n_init --max_evals $max_evals --seed $i\
        --neurons 256 256 256 256 --act_funcs "GELU" "GELU" "GELU" "GELU"
done
