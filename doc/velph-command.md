(velph_command)=
# `velph` command

The `velph` command is a convenient tool for systematically performing
electron-phonon interaction calculations using the VASP code and for analyzing
the resulting data.

The `velph` command orchestrates the electron-phonon (el-ph) calculation
workflow {ref}`workflow <workflow>` by running a series of subcommands in a
specific sequence.

## `velph` subcommands

The `velph` command includes a variety of subcommands. To initiate an
electron–phonon (el-ph) calculation, run `velph init` to generate the
`velph.toml` file. Command options for `velph init` are detailed in
{ref}`velph_init`. Once the `velph.toml` file is prepared, other `velph`
subcommands generate VASP input files, post-process VASP calculation results,
and calculate derivatives of the local potential and PAW strengths with respect
to atomic displacement, resulting in the `phelel_params.hdf5` file. Most of the
`velph` subcommands are explained in {ref}`velph_subcommands`.

A list of supported subcommands is displayed by typing:

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

(velph_hints)=
## `velph hints`

This command provides a quick reference of calculation steps.

```
% velph hints
```
