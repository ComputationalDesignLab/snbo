#!/bin/bash

# Replace 'PATH_TO_SNBO_FOLDER' with the path to your SNBO project
export PYTHONPATH=PATH_TO_SNBO_FOLDER:$PYTHONPATH

# variables
problem="rover"
dim=100
n_init=$((2*$dim))
max_evals=2000

python algorithms/optimize.py --method "snbo" --problem $problem --dim $dim --n_init $n_init --max_evals $max_evals\
        --neurons 256 256 --act_funcs "GELU" "GELU"

python algorithms/optimize.py --method "bo" --problem $problem --dim $dim --n_init $n_init --max_evals $max_evals

python algorithms/optimize.py --method "ibnn" --problem $problem --dim $dim --n_init $n_init --max_evals $max_evals
