## 📌 Overview

This repository implements Scalable Neural Network-based Blackbox Optimization (SNBO) — a novel 
method for efficient blackbox optimization using neural networks. This repository also includes code
for various benchmark methods used for evaluating the SNBO.

This work is currently under review and will be available soon.

## 🛠 Installation

Before running SNBO or any of the benchmark methods, you need to set up the environment and install the dependencies.

### 🐍 1. Create a conda environment

```
conda create -n "snbo" python=3.12
```

### 📦 2. Install dependecies

Install the required Python packages using ``pip``:

```
pip install numpy==2.3.1 scipy==1.16.0 matplotlib==3.10.3 torch==2.7.1
gpytorch==1.14 botorch==0.14.0 gymnasium["mujoco"]==1.2.0
```

If you want to use DYCORS for running a test case, then please install it by downloading the 
[DYCORS repository](https://github.com/aquirosr/DyCors) and running following command in terminal:

```
pip install .
```

## ▶️ Running Examples

Before running any of the runscripts, you need to append the path of the
root folder to `PYTHONPATH`. The command for this is already available at the top of all the script files in the runscript folder.

Open the script file and ensure that the correct path is mentioned.

To solve a test problem using SNBO or any of the benchmark methods, you can use ``single_run.sh`` file under runscripts folder. To execute the file, run:

```
sh runscripts/single_run.sh
```

If you want to run a batch of optimization, you can use ``batch_run.sh`` file under runscripts folder. To execute the file, run:

```
sh runscripts/batch_run.sh
```

When you execute any of the above commands, it will create a folder name ``results`` and will save the entire optimization history in a mat file.

> 💡 **_NOTE:_** It is recommended to run the bash file or `optimize.py` file from the root folder and NOT from within the subfolder.

## 	🧾 Citation

If you use this code or SNBO method in your research, please cite the original work (citation coming soon, paper under review).
