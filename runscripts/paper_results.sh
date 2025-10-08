#!/bin/bash

# Replace 'PATH_TO_SNBO_FOLDER' with the path to your SNBO project
export PYTHONPATH=PATH_TO_SNBO_FOLDER:$PYTHONPATH

methods=("bo" "ibnn" "turbo" "dycors" "snbo")

############ analytical cases

dims=(10 25 50)
max_evals=(500 1000 1000)
problems=("ackley" "rastrigin" "levy")

for seed in $(seq 1 10); do
    for dim_idx in "${!dims[@]}"; do
        dim=${dims[$dim_idx]}
        n_init=$((2*$dim))
        max_eval=${max_evals[$dim_idx]}
        for prob_idx in "${!problems[@]}"; do
            for method_idx in "${!methods[@]}"; do
                python algorithms/optimize.py --method ${methods[$method_idx]} --problem ${problems[$prob_idx]} --dim $dim --n_init $n_init\
                    --max_evals $max_eval --seed $seed --neurons 256 256 --act_funcs "GELU" "GELU"
            done
        done
    done
done

############ rover problem

dim=100
max_evals=2000
problem="rover"

for seed in $(seq 1 10); do
    n_init=$((2*$dim))
    for method_idx in "${!methods[@]}"; do
        python algorithms/optimize.py --method ${methods[$method_idx]} --problem $problem --dim $dim\
            --n_init $n_init --max_evals $max_evals --seed $seed --neurons 256 256 --act_funcs "GELU" "GELU"
    done
done

############ halfcheetah problem

max_evals=2000
problem="halfcheetah"
dim=102

for seed in $(seq 1 10); do
    n_init=$((2*$dim))
    for method_idx in "${!methods[@]}"; do
        python algorithms/optimize.py --method ${methods[$method_idx]} --problem $problem --dim $dim\
            --n_init $n_init --max_evals $max_evals --seed $seed --neurons 256 256 256 256 --act_funcs "GELU" "GELU" "GELU" "GELU"
    done
done
