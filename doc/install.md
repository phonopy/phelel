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
% conda install phelel
```

From source code of phelel,
```
% conda create -n phelel -c conda-forge
% conda activate phelel
% conda install -c conda-forge phono3py finufft click tomli tomli-w seekpath
% git clone https://github.com/phonopy/phelel.git
% cd phelel
% pip install -e .
```

The pypi package is also available.
