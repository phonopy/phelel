# Installation

## Requirement

* phonopy
* phono3py
* spglib
* {ref}`reference_finufft`
* click
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
% conda install -c conda-forge phono3py finufft click tomli-w seekpath
% git clone https://github.com/phonopy/phelel.git
% cd phelel
% pip install -e .
```

The pypi package is also available.

## `phelel` and `velph` commands

Installing the phelel package makes the `phelel` and `velph` commands available.
Their usage is described in {ref}`phelel_command` and {ref}`velph_command`,
respectively. Additionally, the `velph` command supports shell completion, which
requires further configuration as described in the next section.

### Shell completion of `velph` command

Velph relies on [click](https://click.palletsprojects.com), and shell completion
is provided for popular shell implementations, see
[shell-completion](https://click.palletsprojects.com/en/stable/shell-completion/).

For example using bash (zsh) in conda environment, write the following line

```
eval "$(_VELPH_COMPLETE=bash_source velph)"  # BASH
```

```
eval "$(_VELPH_COMPLETE=zsh_source velph)"  # ZSH
```

in `~/.bashrc` (`~/.zshrc`), or in a conda environment in
`$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`.

After setting and reloading the configuration file (e.g., `~/.bashrc`),
sub-commands are listed by pushing tab key:

```bash
% velph [PUSH-TAB-KEY]
convert        -- Write velph.toml of velph 0.14 in the...
el_bands       -- Choose electronic band structure options.
el_dos         -- Choose electronic DOS options.
generate       -- Write POSCAR-unitcell and POSCAR-primitive.
init           -- Initialize an electron phonon calculation...
nac            -- Choose nac options.
ph_bands       -- Choose phonon band structure options.
ph_selfenergy  -- Choose ph_selfenergy options.
phelel         -- Choose supercell options.
phono3py       -- Choose phono3py options.
phonopy        -- Choose phonopy options.
relax          -- Choose relax options.
scheduler      -- Show the job script _job.sh of each input...
selfenergy     -- Choose selfenergy options.
transport      -- Choose transport options.
```
