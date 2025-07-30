# Scalable Neural Network-based Blackbox Optimization

This repository provides code for implementing the SNBO method.

## Installation Details

Before running the SNBO method, you need to setup the environment and install various packages.
First, create a virtual environment using conda by running following command:

```
    conda create -n "snbo" python=3.12
```

Once the environment is created, next step is to install packages 
in that environment. Activate the snbo environment and run the following command:

```
    pip install numpy==2.3.1 scipy==1.16.0 matplotlib==3.10.3 torch==2.7.1
    gpytorch==1.14 botorch==0.14.0 gymnasium["mujoco"]==1.2.0
```

or, you can install packages using the requirements file by executing following command as well:

```
    pip install -r requirements.txt
```

### Running Examples

You can also add this line in your shell configuration file

To solve a test problem using SNBO (or any of the available methods), execute following 
command in the terminal from root folder:

```
    sh runscript.sh
```
