# `velph` example

## NiTiSn

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
