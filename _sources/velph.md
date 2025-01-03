(velph_command)=
# velph command

The `velph` command is a convenient tool to systematically perform
electron-phonon interaction calculations with VASP code and analyze the results.
Velph works in combination of command options. The `velph` command is installed
along with the installation of phelel.

## Shell completion

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

## `velph-hints`

This command provides a quick reference of calculation steps.

```
% velph hints
```

## `velph init`

`velph init` with command options will generate modified `velph.toml` from the
template.
```
% velph init [OPTIONS] CELL_FILENAME PROJECT_FOLDER
```

`velph init --help` shows short documents of the options. Two arguments have to
be specified, POSCAR-format crystal structure and directory name that is created
where `velph.toml` is stored. The available options are explained below. Some
options can be specified in `[init.options]` section of the velph-toml template
file (see {ref}`velph-init-template`).


### `--template-toml`

Using this option, `velph.toml` like file is read as the template instead of the
template hard coded in velph code.

### `--tolerance`

This is used for the symmetry check tolerance in Angstrom. Symmetry is searched
always even without `--symmetrize`.

### `--symmetrize`

By default (no-symmetrize), the input POSCAR-type structure is simply used as
the unit cell. With `--symmetrize`, input POSCAR-type structure is symmetrized
and standardized conventional unit cell and primitive cell ("unitcell" and
"primitive_cell", respectively) are written in `velph.toml`.

### `--no-find-primitive`

By this option, the input POSCAR-type structure is used as the primitive cell
even if it is not a primitive cell.

### `--kspacing` and `--kspacing-dense`

Sampling k-point meshes are calculated from these values in the similar way to
VASP `KSPACING` definition by overwriting the template.

### `--max-num-atoms`

Supercell shape is determined so that its number of atoms is equal or less than
this number respecting the crystallographic point group.

### `--cell-for-relax`

This chooses unit cell or primitive cell for structure optimization (`relax`).
The default is `unitcell`. Specify `primitive` to use primitive cell.

### `--cell-for-nac`

This chooses unit cell or primitive cell for NAC calculation (`nac`). The
default is `primitive`. Specify `primitive` to use primitive cell.

### `--primitive-cell-choice`

Primitive cell choice, "standardized" (default) or "reduced".

### `--use-grg`

Use generalized regular grid.

### `--amplitude`

Distance of displacements in Angstrom.

### `--magmom`

String corresponding to INCAR MAGMOM tag value for unit cell, e.g., "24*1" or "0
0 1". This is similar to `MAGMOM` tag in phonopy, see
https://phonopy.github.io/phonopy/setting-tags.html#magmom. In velph, the
asterisk symbol (`*`) is supported.

(velph-init-template)=
## velph-init template

`velph-init` is the command to prepare `velph.toml`. Without specifying a
velph-toml-template, the default template is used. Custom template can be
specified as follows:

```
% velph init [OPTIONS] CELL_FILENAME PROJECT_FOLDER --template-toml velph-tmpl.toml
```

The `velph-tmpl.toml` (arbitrary file name) is almost like the `velph.toml`
file. It means that the `velph-tmpl.toml` may be created modifying
`velph.toml`. The sections that exist only in the `velph-tmpl.toml` but not in
`velph.toml` is `[init.options]`. This can be used as alternatives of command
options of `velph-init`, e.g.,

```toml
[init.options]
kspacing = 0.2
kspacing_dense = 0.1
max_num_atoms = 120
```

These `[init.options]` keywords can be found along with the list of the command
options by

```
% velph init --help
```

## Settings in sections of `velph.toml`

Note that the same can be applied to `velph-tmpl.toml`.

### Default INCAR settings

The default INCAR settins are written in `[vasp.incar]`. These settings are
overwritten by `[vasp.CALC_TYPE.incar]` (`CALC_TYPE` can be `phelel`, `relax`,
`nac`, `transport`, `phono3py`, `phono3py.phonon`, etc).

### Scheduler settings

The parameters used for generating the job submission script are specified as
strings in `[scheduler]`. If `[vasp.CALC_TYPE.scheduler]` is
specified, `[scheduler]` settings are overwritten by the settings for
`[vasp.CALC_TYPE]`. The content in this section operates through [Python string
formatting](https://docs.python.org/3/library/stdtypes.html#str.format), using
replacement fields named after keyword arguments. Each parameter line in the
`[scheduler]` section is treated as a keyword argument, which is then inserted
into the string format.

In the `[scheduler]` section, the parameters `scheduler_name` and
`scheduler_template` have special roles:

- `scheduler_name`: Specifies the template type, with options `slurm`, `sge`, or
  `custom`. For `slurm` and `sge`, the template string (i.e.,
  `scheduler_template`) is
  [hard-coded](https://github.com/phonopy/phelel/blob/develop/src/phelel/velph/utils/scheduler.py).
  If `custom` is selected, the string provided by `scheduler_template` is used,
  into which the parameters are inserted.
- `scheduler_template`: The string template where parameters defined in this
  section are inserted.

An example is shown below.

```toml
[scheduler]
scheduler_name = "sge"
job_name = "PbTe"
mpirun_command = "mpirun"
vasp_binary = "/usr/local/cluster-1/bin/vasp_std"
pe = "vienna 32"
prepend_text = '''
source /opt/intel/oneapi/setvars.sh --config="/home/togo/.oneapi-config"
'''

...
[vasp.phelel.scheduler]
pe = "paris 24"

...
[vasp.phonopy.scheduler]
scheduler_template = '''#!/bin/bash
#QSUB2 core 192
#QSUB2 mpi 192
#QSUB2 smp 1
#QSUB2 wtime 48:00:00
#PBS -N {job_name}
cd $PBS_O_WORKDIR

{prepend_text}
{mpirun_command} {vasp_binary} | tee vasp_output
{append_text}
'''
job_name = "PbTe"
mpirun_command = "mpijob"
vasp_binary = "/usr/local/cluster-2/bin/vasp_std"
prepend_text = '''
. /etc/profile.d/modules.sh
module load inteloneapi22u3
'''
append_text = ""
```


## Example (NiTiSn)

`POSCAR-unitcell`, `POTCAR`, and `velph-tmpl.toml` are located in the current directory.

Initial `POSCAR-unitcell`,
```
Ti4Sn4Ni4
1.0
5.906882041566931   0.000000000000000   0.000000000000000
0.000000000000000   5.906882041566931   0.000000000000000
0.000000000000000   0.000000000000000   5.906882041566931
Ti   Sn   Ni
4   4   4
Direct
0.000000000000000   0.000000000000000   0.000000000000000
0.000000000000000   0.500000000000000   0.500000000000000
0.500000000000000   0.000000000000000   0.500000000000000
0.500000000000000   0.500000000000000   0.000000000000000
0.500000000000000   0.500000000000000   0.500000000000000
0.500000000000000   0.000000000000000   0.000000000000000
0.000000000000000   0.500000000000000   0.000000000000000
0.000000000000000   0.000000000000000   0.500000000000000
0.250000000000000   0.250000000000000   0.250000000000000
0.250000000000000   0.750000000000000   0.750000000000000
0.750000000000000   0.250000000000000   0.750000000000000
0.750000000000000   0.750000000000000   0.250000000000000
```

`velph-tmpl.toml`,
```toml
[init.options]
kspacing = 0.1
kspacing_dense = 0.2
max_num_atoms = 100

[vasp.incar]
encut = 520
ncore = 4

[vasp.phelel.incar]
kpar = 2
[vasp.phelel.scheduler]
pe = "mpi* 48"

[vasp.transport.scheduler]
pe = "mpi* 144"

[scheduler]
scheduler_name = "sge"
job_name = "TiNiSn"
mpirun_command = "mpirun"
vasp_binary = "/usr/local/vasp/bin/vasp_std"
pe = "vienna 32"
prepend_text = '''
source /opt/intel/oneapi/setvars.sh intel64 --config="/home/togo/.oneapi-config"
'''
append_text = ""
```

`POTCAR` choice,
```
  PAW_PBE Ti_sv 26Sep2005
  PAW_PBE Sn_d 06Sep2000
  PAW_PBE Ni 02Aug2007
```

```
% velph init POSCAR_Ti4Sn4Ni4 relax --template-toml velph-tmpl.toml
Crystal structure file: "POSCAR_Ti4Sn4Ni4".
Velph template file: "velph-tmpl.toml".
Read crystal structure file "POSCAR_Ti4Sn4Ni4".
Read velph template file "velph-tmpl.toml".
Following options were found in "velph-tmpl.toml":
  kspacing = "0.1"
  kspacing_dense = "0.2"
  max_num_atoms = "100"
The command options were prefered to [init.options] in "velph-tmpl.toml".
Found a primitive cell whose transformation matrix is
  [ 0.000  0.500  0.500]
  [ 0.500  0.000  0.500]
  [ 0.500  0.500  0.000]
Supercell is generated with respect to the cell below.
--------------------------------------------------------------------------------
lattice:
- [     5.906882041566931,     0.000000000000000,     0.000000000000000 ] # a
- [     0.000000000000000,     5.906882041566931,     0.000000000000000 ] # b
- [     0.000000000000000,     0.000000000000000,     5.906882041566931 ] # c
points:
- symbol: Ti # 1
  coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 47.867000
- symbol: Ti # 2
  coordinates: [  0.000000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 47.867000
- symbol: Ti # 3
  coordinates: [  0.500000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 47.867000
- symbol: Ti # 4
  coordinates: [  0.500000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 47.867000
- symbol: Sn # 5
  coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 118.710000
- symbol: Sn # 6
  coordinates: [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 118.710000
- symbol: Sn # 7
  coordinates: [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 118.710000
- symbol: Sn # 8
  coordinates: [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 118.710000
- symbol: Ni # 9
  coordinates: [  0.250000000000000,  0.250000000000000,  0.250000000000000 ]
  mass: 58.693400
- symbol: Ni # 10
  coordinates: [  0.250000000000000,  0.750000000000000,  0.750000000000000 ]
  mass: 58.693400
- symbol: Ni # 11
  coordinates: [  0.750000000000000,  0.250000000000000,  0.750000000000000 ]
  mass: 58.693400
- symbol: Ni # 12
  coordinates: [  0.750000000000000,  0.750000000000000,  0.250000000000000 ]
  mass: 58.693400
--------------------------------------------------------------------------------
[vasp.incar] (basic INCAR settings)
  ismear = 0
  sigma = 0.01
  ediff = 1e-08
  encut = 520
  prec = accurate
  lreal = False
  lwave = False
  lcharg = False
  ncore = 4
[phelel]
  supercell_dimension: [2 2 2]
[vasp.*.kpoints.mesh] (*kspacing=0.1)
  selfenergy: [18 18 18]*
  el_bands: [18 18 18]*
  phelel: [5 5 5]*
  relax: [11 11 11]*
  nac: [18 18 18]*
[vasp.*.kpoints_dense.mesh] (*kspacing_dense=0.2)
  selfenergy: [9 9 9]*
  el_bands: [9 9 9]*
Created new folder "relax".
Initial settings were written to "relax/velph.toml".
Found "POTCAR".
  PAW_PBE Ti_sv 26Sep2005
  PAW_PBE Sn_d 06Sep2000
  PAW_PBE Ni 02Aug2007
  Max ENMAX in "POTCAR" is 274.61.
"POTCAR" was copied to "relax/POTCAR".
```


A directory `relax` is created and `velph.toml` is stored in it.

`relax/velph.toml`,
```toml
[phelel]
version = "0.6.0"
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = true
plusminus = true
fft_mesh = [32, 32, 32]

# ...

[vasp.relax]
cell = "unitcell"
[vasp.relax.incar]
ediffg = -1e-06
ibrion = 2
isif = 3
nsw = 10
ismear = 0
sigma = 0.01
ediff = 1e-08
encut = 520
prec = "accurate"
lreal = false
lwave = false
lcharg = false
ncore = 4
[vasp.relax.kpoints]
mesh = [11, 11, 11]

# ...

[scheduler]
scheduler_name = "sge"
job_name = "TiNiSn"
mpirun_command = "mpirun"
vasp_binary = "/usr/local/vasp/bin/vasp_std"
pe = "vienna 32"
prepend_text = "source /opt/intel/oneapi/setvars.sh intel64 --config=\"/home/togo/.oneapi-config\"\n"
append_text = ""

[symmetry]
spacegroup_type = "F-43m"
tolerance = 1e-05
primitive_matrix = [
  [  0.000000000000000,  0.500000000000000,  0.500000000000000 ],
  [  0.500000000000000,  0.000000000000000,  0.500000000000000 ],
  [  0.500000000000000,  0.500000000000000,  0.000000000000000 ],
]

[unitcell]
lattice = [
  [     5.906882041566931,     0.000000000000000,     0.000000000000000 ], # a
  [     0.000000000000000,     5.906882041566931,     0.000000000000000 ], # b
  [     0.000000000000000,     0.000000000000000,     5.906882041566931 ], # c
]
[[unitcell.points]]  # 1
symbol = "Ti"
coordinates = [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
[[unitcell.points]]  # 2
symbol = "Ti"
coordinates = [  0.000000000000000,  0.500000000000000,  0.500000000000000 ]
[[unitcell.points]]  # 3
symbol = "Ti"
coordinates = [  0.500000000000000,  0.000000000000000,  0.500000000000000 ]
[[unitcell.points]]  # 4
symbol = "Ti"
coordinates = [  0.500000000000000,  0.500000000000000,  0.000000000000000 ]
[[unitcell.points]]  # 5
symbol = "Sn"
coordinates = [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
[[unitcell.points]]  # 6
symbol = "Sn"
coordinates = [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
[[unitcell.points]]  # 7
symbol = "Sn"
coordinates = [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
[[unitcell.points]]  # 8
symbol = "Sn"
coordinates = [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
[[unitcell.points]]  # 9
symbol = "Ni"
coordinates = [  0.250000000000000,  0.250000000000000,  0.250000000000000 ]
[[unitcell.points]]  # 10
symbol = "Ni"
coordinates = [  0.250000000000000,  0.750000000000000,  0.750000000000000 ]
[[unitcell.points]]  # 11
symbol = "Ni"
coordinates = [  0.750000000000000,  0.250000000000000,  0.750000000000000 ]
[[unitcell.points]]  # 12
symbol = "Ni"
coordinates = [  0.750000000000000,  0.750000000000000,  0.250000000000000 ]
[primitive_cell]
lattice = [
  [     0.000000000000000,     2.953441020783465,     2.953441020783465 ], # a
  [     2.953441020783465,     0.000000000000000,     2.953441020783465 ], # b
  [     2.953441020783465,     2.953441020783465,     0.000000000000000 ], # c
]
[[primitive_cell.points]]  # 1
symbol = "Ti"
coordinates = [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
[[primitive_cell.points]]  # 2
symbol = "Sn"
coordinates = [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
[[primitive_cell.points]]  # 3
symbol = "Ni"
coordinates = [  0.250000000000000,  0.250000000000000,  0.250000000000000 ]
```

`fft_mesh` is calculated from `encut` value in the `[vasp.selfenergy.incar]` section.

Change directory to `relax`.
```
% velph relax generate
% cd relax/iter1
% qsub _job.sh
```
Structure optimization is done until stress becomes less than 0.1kB (0.01GPa).
Go back to the top directory and start supercell calculation.

```
% velph init --template-toml velph-tmpl.toml `ls relax/relax/iter*/CONTCAR|tail -n 1` calc
Crystal structure file: "relax/relax/iter2/CONTCAR".
Velph template file: "velph-tmpl.toml".
Read crystal structure file "relax/relax/iter2/CONTCAR".
Read velph template file "velph-tmpl.toml".
Following options were found in "velph-tmpl.toml":
  kspacing = "0.1"
  kspacing_dense = "0.2"
  max_num_atoms = "100"
The command options were prefered to [init.options] in "velph-tmpl.toml".
Found a primitive cell whose transformation matrix is
  [ 0.000  0.500  0.500]
  [ 0.500  0.000  0.500]
  [ 0.500  0.500  0.000]
Supercell is generated with respect to the cell below.
--------------------------------------------------------------------------------
lattice:
- [     5.949975748250650,     0.000000000000000,     0.000000000000000 ] # a
- [     0.000000000000000,     5.949975748250650,    -0.000000000000000 ] # b
- [     0.000000000000000,     0.000000000000000,     5.949975748250650 ] # c
points:
- symbol: Ti # 1
  coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 47.867000
- symbol: Ti # 2
  coordinates: [  0.000000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 47.867000
- symbol: Ti # 3
  coordinates: [  0.500000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 47.867000
- symbol: Ti # 4
  coordinates: [  0.500000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 47.867000
- symbol: Sn # 5
  coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 118.710000
- symbol: Sn # 6
  coordinates: [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 118.710000
- symbol: Sn # 7
  coordinates: [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 118.710000
- symbol: Sn # 8
  coordinates: [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 118.710000
- symbol: Ni # 9
  coordinates: [  0.250000000000000,  0.250000000000000,  0.250000000000000 ]
  mass: 58.693400
- symbol: Ni # 10
  coordinates: [  0.250000000000000,  0.750000000000000,  0.750000000000000 ]
  mass: 58.693400
- symbol: Ni # 11
  coordinates: [  0.750000000000000,  0.250000000000000,  0.750000000000000 ]
  mass: 58.693400
- symbol: Ni # 12
  coordinates: [  0.750000000000000,  0.750000000000000,  0.250000000000000 ]
  mass: 58.693400
--------------------------------------------------------------------------------
[vasp.incar] (basic INCAR settings)
  ismear = 0
  sigma = 0.01
  ediff = 1e-08
  encut = 520
  prec = accurate
  lreal = False
  lwave = False
  lcharg = False
  ncore = 4
[phelel]
  supercell_dimension: [2 2 2]
[vasp.*.kpoints.mesh] (*kspacing=0.1)
  selfenergy: [18 18 18]*
  el_bands: [18 18 18]*
  phelel: [5 5 5]*
  relax: [11 11 11]*
  nac: [18 18 18]*
[vasp.*.kpoints_dense.mesh] (*kspacing_dense=0.2)
  selfenergy: [9 9 9]*
  el_bands: [9 9 9]*
Created new folder "calc".
Initial settings were written to "calc/velph.toml".
Found "POTCAR".
  PAW_PBE Ti_sv 26Sep2005
  PAW_PBE Sn_d 06Sep2000
  PAW_PBE Ni 02Aug2007
  Max ENMAX in "POTCAR" is 274.61.
"POTCAR" was copied to "calc/POTCAR".
```

Change directory to `calc`.
```
% cd calc
```

`velph.toml`,
```
[phelel]
version = "0.6.0"
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = true
plusminus = true
fft_mesh = [32, 32, 32]

[vasp.phelel.incar]
lwap = true
isym = 0
kpar = 2
ismear = 0
sigma = 0.01
ediff = 1e-08
encut = 520
prec = "accurate"
lreal = false
lwave = false
lcharg = false
ncore = 4
[vasp.phelel.kpoints]
mesh = [5, 5, 5]
[vasp.phelel.scheduler]
pe = "mpi* 48"

# ...

[vasp.nac]
cell = "primitive"
[vasp.nac.incar]
lepsilon = true
ismear = 0
sigma = 0.01
ediff = 1e-08
encut = 520
prec = "accurate"
lreal = false
lwave = false
lcharg = false
[vasp.nac.kpoints]
mesh = [18, 18, 18]

# ...
```

NAC and supercell calculations,
```
% velph nac generate
% cd nac; qsub _job.sh; cd ..
% velph phelel init
Found "nac" directory. Read NAC params.
"phelel/phelel_disp.yaml" was generated by phelel.
VASP input files will be generated by "velph phelel generate".
% velph supercell generate
VASP input files were generated in "supercell/disp-000".
VASP input files were generated in "supercell/disp-001".
VASP input files were generated in "supercell/disp-002".
VASP input files were generated in "supercell/disp-003".
VASP input files were generated in "supercell/disp-004".
VASP input files were generated in "supercell/disp-005".
VASP input files were generated in "supercell/disp-006".
% for i in {000..006};do cd phelel/disp-$i; qsub _job.sh; cd ../..;done
```

Using the VASP results of the supercell calculations, derivatives of properties
are computed using the phelel code. The results are stored in a newly created
file `phelel/phelel_params.hdf5`. This process creates only
`phelel/phelel_params.hdf5` and is invoked by the following velph command:
```
% velph phelel differentiate
Running finufft (eps=1.000e-06)...
Running finufft (eps=1.000e-06)...
Running finufft (eps=1.000e-06)...
"phelel/phelel_params.hdf5" has been made.
```

NAC parameters in `phelel/phelel_params.hdf5` comes from
`phelel/phelel_disp.yaml`, i.e.,

1. NAC parameters are stored in `phelel/phelel_disp.yaml` when
   `phelel/phelel_disp.yaml` is created by `velph phelel init`.
2. NAC data in `phelel/phelel_disp.yaml` are transfered to
   `phelel/phelel_params.hdf5` by `velph phelel differentiate`.

Therefore, `nac` calculation has to exist in the initial step `velph phelel
init`. If the `nac` calculation is performed after the supercell calculation,
`velph phelel init` and `velph phelel differentiate` have to be re-executed to
store the NAC parameters in `phelel/phelel_params.hdf5`.
