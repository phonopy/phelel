# Installation

## Requirement

* phonopy
* phono3py
* spglib
* finufft
* click
* tomli
* tomli-w
* seekpath (optional)

## Installation from source code

A simplest installation using conda-forge packages:

```
% conda create -n phelel -c conda-forge
% conda activate phelel
% conda install -c conda-forge phono3py finufft click tomli tomli-w seekpath
% git clone https://github.com/phonopy/phelel.git
% cd phelel
% pip install -e .
```

PyPI and conda forge package will be made in the future.
