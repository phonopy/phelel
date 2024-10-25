## velph command

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
