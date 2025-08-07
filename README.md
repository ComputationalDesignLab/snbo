# 🚀 Scalable Neural Network-based Blackbox Optimization

This repository provides implementation for SNBO (Scalable Neural Network-based Blackbox Optimization) — a novel method for efficient blackbox optimization using neural networks. It also includes code for benchmark algorithms and a suite of test problems used in the paper.

> 📝 **Note**: This work is currently under review but a preprint version is available at: https://arxiv.org/abs/2508.03827

## 📌 Features

This repository includes implementation for the following optimization algorithms:

- `SNBO`: Scalable neural network-based blackbox optimization (proposed method)
- `BO+LogEI`: [Bayesian optimization with LogEI acquisition](https://arxiv.org/abs/2310.20708)
- `IBNN`: [Bayesian optimization with IBNN kernel and LogEI acquisition](https://botorch.org/docs/tutorials/ibnn_bo/#i-bnns-for-bayesian-optimization)
- `TuRBO`: [Trust region Bayesian optimization](https://arxiv.org/abs/1910.01739)
- `DYCORS`: [Dynamic coordinate search with repsonse surface model](https://www.tandfonline.com/doi/abs/10.1080/0305215X.2012.687731)

This repository also contains following test problems:

- [Ackley function](https://www.sfu.ca/~ssurjano/ackley.html)
- [Rastrigin function](https://www.sfu.ca/~ssurjano/rastr.html)
- [Levy function](https://www.sfu.ca/~ssurjano/levy.html)
- [Rover trajectory optimization](https://github.com/zi-w/Ensemble-Bayesian-Optimization/blob/master/test_functions/rover_function.py)
- [Half-Cheetah problem](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)

## 🛠 Installation

Before running any example, you need to set up the environment and install the dependencies.

### 🐍 1. Create a conda environment

```
conda create -n "snbo" python=3.12
conda activate snbo
```

### 📦 2. Install dependecies

Install the required Python packages using ``pip``:

```
pip install numpy==2.3.1 scipy==1.16.0 matplotlib==3.10.3 torch==2.7.1
gpytorch==1.14 botorch==0.14.0 gymnasium["mujoco"]==1.2.0
```

If you want to use DYCORS, then please install it using following commands:

```
git clone https://github.com/aquirosr/DyCors
cd DyCors
pip install .
```

## ▶️ Running examples

> 📥 Important: Before running any example, download the latest released or tagged version of this repository from the [releases page](https://github.com/ComputationalDesignLab/snbo/releases).

The main file for running experiments is `optimize.py`, found in algorithms folder. This file provides following arguments:

| Argument    | Type | Description                                                                   |
| ----------- | ---- | ----------------------------------------------------------------------------- |
| `method`    | str  | Optimization method: `"bo"`, `"ibnn"`, `"turbo"`, `"dycors"`, `"snbo"`        |
| `problem`   | str  | Test problem: `"ackley"`, `"rastrigin"`, `"levy"`, `"rover"`, `"halfcheetah"` |
| `dim`       | int  | Dimensionality of the problem                                                 |
| `n_init`    | int  | Number of initial samples                                                     |
| `max_evals` | int  | Maximum number of function evaluations                                        |
| `neurons`   | list | Neurons per layer (only for `snbo`)                                           |
| `act_funcs` | list | Activation functions per layer (only for `snbo`)                              |
| `seed`      | int  | Random seed (optional)                                                        |

There are two ways to run this file:

✅ Option 1: You can directly run the file from the terminal. For example, to solve the rover problem using SNBO method, you can use following command:

```
python algorithms/optimize.py --method "snbo" --problem "rover" --dim 100 --n_init 200 --max_evals 2000 --neurons 256 256 --act_funcs "GELU" "GELU"
```

> ⚠️ ***NOTE***: Before running the file, you need to append the path of the root folder to `PYTHONPATH` variable.  This can be done by running following command in the terminal before running the python file:
>   ```
>   export PYTHONPATH=PATH_TO_SNBO_FOLDER:$PYTHONPATH
>   ```
>   Or, you can add this line to your shell configuration file (`.bashrc` or`.zshrc`) and reload the terminal.

✅ Option 2 (**Recommended**): Instead of directly running the `optimize.py` file, you can use one of the ready-to-use scripts available in runscript folder. These scripts already include the `export` statement requried for appending the `PYTHONPATH` variable at the start of the script, you just need to ensure that correct path is defined.

To solve a test problem using SNBO or any of the benchmark methods, you can use ``single_run.sh`` file under runscripts folder. To execute the file, run:

```
bash runscripts/single_run.sh
```

If you want to run a batch of optimization, you can use ``batch_run.sh`` file under runscripts folder. To execute the file, run:

```
bash runscripts/batch_run.sh
```

> ⚠️ **_NOTE:_** It is recommended to run the python or the bash file from the root folder and NOT from within the subfolder.

When you run the `optimize.py` file or any of the bash script, a folder named ``results`` will be created that consists of 
different subfolders, depending on the problem you are solving and the method you selected. A mat file will be saved within appropriate
subfoler that contains entire optimization history.

## 📊 Results from paper

To reproduce the data reported in the paper, you can use ``paper_results.sh`` script. Use the following command to run this script:

```
bash runscripts/paper_results.sh
```

For each seed value, this script loops through each problem and solves it using all the methods.

> ⏳ **Warning**: This script will take a long time to run, depending on the resources used

## 	🧾 Citation

If you use SNBO method in your research, please cite the original work (citation coming soon, paper under review).
