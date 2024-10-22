# Parametric Grid Convolutional Attention Networks (PGCAN)

PGCAN introduces a novel architecture for solving Partial Differential Equations (PDEs) by combining parametric grid-based encoding with convolutional and attention mechanisms to address spectral bias and enhance solution accuracy for complex PDEs.

## Overview

PGCAN leverages Parametric Grid Encoding, Convolutional Layers, and an Attention Mechanism to improve the learning of PDE solutions. This approach structures the input space, reduces overfitting, enhances feature capture, and prioritizes significant features for accurate and efficient PDE solution approximation.

![PGCAN Architecture](figures/figure1.PNG)
_Figure 1: PGCAN architecture._

![Solution Comparison](figures/sol.png)
_Figure 2: Comparative results._



## Quick Start

## 1. Installation

### Clone PGCAN repo

```
git clone https://github.com/Bostanabad-Research-Group/pgcan.git
```
```
cd pgcan
```

### Create environment

The code is tested with Python 3.8.8 PyTorch 1.13.1 with CUDA 11.6. 
You can create an anaconda environment with those requirements by running:

```
conda env create -f environment.yml
```
```
conda activate pgcan
```



## Usage

To run PGCAN, follow these steps:

## 2. Run
You can run pgcan and other models directly using the following code.Please make sure the below folders exist in the running directory:
##### Figures 
(cross plots and contours and losses will be plotted)
##### Text 
(Two type of files are saved. a .txt files to show the elapsed time and number of paramter in the model and a .csv file that shows accuracy for 10 intervals)
##### 
Saved_Models (saved models checkpoints)

[NS cavity equation]
```
python benchmark.py
```


## Citation


@article{pgcan2024,
  author = {Shishehbor, Mehdi and Hosseinmardi, Shirin and Bostanabad, Ramin},
  title = {Parametric encoding with attention and convolution mitigate spectral bias of neural partial differential equation solvers},
  journal = {Structural and Multidisciplinary Optimization},
  year = {2024},
  volume = {67},
  number = {7},
  pages = {128},
  issn={1615-1488},
  doi={10.1007/s00158-024-03834-7},
}


## Contact

For queries, please contact Raminb@uci.edu.


