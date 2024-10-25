# Phelel

A code that provides a few computations related to electron-phonon interaction
calculation in finite-displacement method reported by

Laurent Chaput, Atsushi Togo, and Isao Tanaka, Phys. Rev. B **100**, 174304
(2019).

Note that this code couples with VASP code, and the electron-phonon interaction
properties can not be computed only using this code.

Phelel user documentation is found at
http://phonopy.github.io/phelel/

## Installation

### Requirement

* phonopy
* phono3py
* spglib
* finufft
* click
* tomli
* tomli-w
* seekpath (optional)

### Installation from source code

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

## Command-line tool: velph

### Configuration of shell completion

Velph command is a convenient tool to systematically perform electron-phonon
interaction calculations with VASP code and analyze the results. Velph works in
combination of command options. The command `velph` is installed along with
the installation of phelel.

Velph relies on click, and shell completion is provided for popular shell implementations, see
https://click.palletsprojects.com/en/stable/shell-completion/.

For example using bash (zsh) in conda environment, write the following line

(for bash)
```
eval "$(_VELPH_COMPLETE=bash_source velph)"
```

(for zsh)
```
eval "$(_VELPH_COMPLETE=zsh_source velph)"
```

in `~/.bashrc` (`~/.zshrc`), or in a conda environment in
`$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`.

After setting and reloading the configuration file (e.g., `~/.bashrc`),
sub-commands are listed by pushing tab key:

```bash
% velph [PUSH-TAB-KEY]
el_bands    -- Choose electronic band structure options.
generate    -- Write POSCAR-unitcell and POSCAR-primitive.
hints       -- Show velph command hints.
init        -- Initialize an electron phonon calculation...
nac         -- Choose nac options.
ph_bands    -- Choose phonon band structure options.
phelel      -- Choose supercell options.
phono3py    -- Choose phono3py options.
relax       -- Choose relax options.
selfenergy  -- Choose selfenergy options.
transport   -- Choose transport options.
```

### `velph-hints`

This command provides a quick reference of calculation steps.

## Development

### Formatting

Formatting rules are found in `pyproject.toml`.

### pre-commit

Pre-commit (https://pre-commit.com/) is mainly used for applying the formatting
rules automatically. Therefore, it is strongly encouraged to use it at or before
git-commit. Pre-commit is set-up and used in the following way:

- Installed by `pip install pre-commit`, `conda install pre_commit` or see
  https://pre-commit.com/#install.
- pre-commit hook is installed by `pre-commit install`.
- pre-commit hook is run by `pre-commit run --all-files`.

Unless running pre-commit, pre-commit.ci may push the fix at PR by github
action. In this case, the fix should be merged by the contributor's repository.

### VSCode setting
- Not strictly, but VSCode's `settings.json` may be written like below

  ```json
  "ruff.lint.args": [
      "--config=${workspaceFolder}/pyproject.toml",
  ],
  "[python]": {
      "editor.defaultFormatter": "charliermarsh.ruff",
      "editor.codeActionsOnSave": {
          "source.organizeImports": "explicit"
      }
  },
  ```

## How to run tests

Tests are written using pytest. To run tests, pytest has to be installed. The
tests can be run by

```bash
% pytest
```

## License

BSD-3-Clause.
