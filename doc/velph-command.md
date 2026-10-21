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

A typical workflow of velph subcommands for transport property calculation is
illustrated below.

```{mermaid}
graph LR
A[velph-init] --> B[velph-relax]
B --> C[velph-init]
C --> D[velph-nac]
C --> E[velph-el_bands]
F[velph-phelel]
D --> F
D --> H[velph-phono3py]
F --> G[velph-transport]
```

A list of supported subcommands is displayed by typing:

```bash
% velph
Usage: velph [OPTIONS] COMMAND [ARGS]...

  Command-line utility to help VASP el-ph calculation.

Options:
  -h, --help  Show this message and exit.

Commands:
  convert        Write velph.toml of velph 0.14 in the format of velph 0.15.
  el_bands       Choose electronic band structure options.
  el_dos         Choose electronic DOS options.
  generate       Write POSCAR-unitcell and POSCAR-primitive.
  init           Initialize an electron phonon calculation project.
  nac            Choose nac options.
  ph_bands       Choose phonon band structure options.
  ph_selfenergy  Choose ph_selfenergy options.
  phelel         Choose supercell options.
  phono3py       Choose phono3py options.
  phonopy        Choose phonopy options.
  relax          Choose relax options.
  scheduler      Show the job script _job.sh of each input set.
  selfenergy     Choose selfenergy options.
  transport      Choose transport options.
```

In the following example, two `velph init` command operations generate two
different directories. The first operation performs crystal structure relaxation
in the `relax` directory. The second operation, using the relaxed crystal
structure, carries out additional calculations required for electron-phonon
interactions in the `calc` directory. Details about `velph-tmpl.toml` are
described in {ref}`velph_init_template`.

```bash
% ls
POSCAR-unitcell  POTCAR  velph-tmpl.toml
% velph init --template-toml velph-tmpl.toml POSCAR-unitcell relax
...
% cd relax
% velph relax generate
...
# Run VASP relaxation calculation
...
% cd ..
% ls
POSCAR-unitcell  POTCAR  relax/  velph-tmpl.toml
% velph init --template-toml velph-tmpl.toml `ls relax/relax/iter*/CONTCAR|tail -n 1` calc
...
% cd calc
% ls
POTCAR  velph.toml
% velph nac generate
VASP input files were written in "nac".
% ls
POTCAR  nac/  velph.toml
...
# Run NAC calculation
...
% velph phelel init
NAC parameters are read from "nac/vasprun.xml".
"phelel/phelel_disp.yaml" was written.
Run "velph phelel generate" to write the VASP input files.
% velph phelel generate
VASP input files were written in "phelel/disp-000".
VASP input files were written in "phelel/disp-001".
VASP input files were written in "phelel/disp-002".
VASP input files were written in "phelel/disp-003".
VASP input files were written in "phelel/disp-004".
% ls
POTCAR  nac/  phelel/  velph.toml
...
```
