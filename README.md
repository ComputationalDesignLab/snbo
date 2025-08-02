## 📌 Overview

This repository implements Scalable Neural Network-based Blackbox Optimization (SNBO) — a novel method for high-dimensional and computationally efficient blackbox optimization using neural networks.

## 🛠 Installation

Before running SNBO, you need to set up the environment and install the required packages.

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

Or install all dependencies from the requirements file:

```
pip install -r requirements.txt
```

You may also need to install DYCORS. To do that, download the [DYCORS repository](https://github.com/aquirosr/DyCors) repository and run:

```
pip install .
```

## ▶️ Running Examples

To run a test optimization using SNBO or one of the baseline methods, you can run runscript.sh

```
sh runscript.sh
```

Before running this file, include the path to the repository at the top of the runscript.sh file. This ensure that the SNBO repository is correctly included in your Python path.

## 	🧾 Citation

If you use this code or SNBO method in your research, please cite the original work (citation coming soon, paper under review).
