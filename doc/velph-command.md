(velph_command)=
# `velph` command

The `velph` command is a convenient tool for systematically performing
electron-phonon interaction calculations using the VASP code and for analyzing
the resulting data.

## What does the `velph` command do?

The `velph` command orchestrates the electron-phonon (el-ph) calculation
workflow {ref}`workflow <workflow>` by running a series of subcommands in a
specific sequence. These subcommands can:

- Execute `phelel`.
- Generate VASP input files for:
  - Dielectric constant calculations
  - Born effective charge calculations
  - Electronic band structure and density of states calculations
  (all of which are required for subsequent electron-phonon interaction calculations)
- Generate VASP input files for phonon and lattice thermal conductivity calculations using phonopy and phono3py.

## Listing `velph` subcommands

You can see a list of supported subcommands by simply typing:

```
% velph
Usage: velph [OPTIONS] COMMAND [ARGS]...

  Command-line utility to help VASP el-ph calculation.

Options:
  -h, --help  Show this message and exit.

Commands:
  el_bands    Choose electronic band structure options.
  generate    Write POSCAR-unitcell and POSCAR-primitive.
  hints       Show velph command hints.
  init        Initialize an electron phonon calculation project.
  nac         Choose nac options.
  ph_bands    Choose phonon band structure options.
  phelel      Choose supercell options.
  phono3py    Choose phono3py options.
  phonopy     Choose phonopy options.
  relax       Choose relax options.
  selfenergy  Choose selfenergy options.
  transport   Choose transport options.
```

## `velph hints`

This command provides a quick reference of calculation steps.

```
% velph hints
```
