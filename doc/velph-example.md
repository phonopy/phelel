# `velph` example

## PbTe

`POSCAR-unitcell`, `POTCAR`, and `velph-tmpl.toml` are located in the current directory.

Initial `POSCAR-unitcell`,
```
Pb Te
   1.00000000000000
     6.4000000000000000    0.0000000000000000    0.0000000000000000
     0.0000000000000000    6.4000000000000000    0.0000000000000000
     0.0000000000000000    0.0000000000000000    6.4000000000000000
  Pb              Te
     4     4
Direct
     0.0000000000000000    0.0000000000000000    0.0000000000000000
     0.0000000000000000    0.5000000000000000    0.5000000000000000
     0.5000000000000000    0.0000000000000000    0.5000000000000000
     0.5000000000000000    0.5000000000000000    0.0000000000000000
     0.5000000000000000    0.5000000000000000    0.5000000000000000
     0.5000000000000000    0.0000000000000000    0.0000000000000000
     0.0000000000000000    0.5000000000000000    0.0000000000000000
     0.0000000000000000    0.0000000000000000    0.5000000000000000
```

`velph-tmpl.toml`,
```toml
[init.options]
kspacing = 0.2
kspacing_dense = 0.1
max_num_atoms = 100

[vasp.incar]
encut = 400
ncore = 4

[vasp.phelel.incar]
kpar = 2
[vasp.phelel.scheduler]
pe = "mpi* 48"

[vasp.transport.scheduler]
pe = "mpi* 144"

[scheduler]
scheduler_name = "sge"
job_name = "PbTe"
mpirun_command = "mpirun"
vasp_binary = "/usr/local/vasp/bin/vasp_std"
pe = "vienna 32"
prepend_text = '''
source /opt/intel/oneapi/setvars.sh
'''
append_text = ""
```

`POTCAR` choice,
```
  PAW_PBE Pb 08Apr2002
  PAW_PBE Te 08Apr2002
```

```
% velph init POSCAR-unitcell relax --template-toml velph-tmpl.toml
Read crystal structure file "POSCAR-unitcell".
Read velph template file "velph-tmpl.toml".
Following options were found in "velph-tmpl.toml":
  kspacing = "0.2"
  kspacing_dense = "0.1"
  max_num_atoms = "100"
The command options were prefered to [init.options] in "velph-tmpl.toml".
Found a primitive cell whose transformation matrix is
  [ 0.000  0.500  0.500]
  [ 0.500  0.000  0.500]
  [ 0.500  0.500  0.000]
Supercell is generated with respect to the cell below.
--------------------------------------------------------------------------------
lattice:
- [     6.400000000000000,     0.000000000000000,     0.000000000000000 ] # a
- [     0.000000000000000,     6.400000000000000,     0.000000000000000 ] # b
- [     0.000000000000000,     0.000000000000000,     6.400000000000000 ] # c
points:
- symbol: Pb # 1
  coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 207.200000
- symbol: Pb # 2
  coordinates: [  0.000000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 207.200000
- symbol: Pb # 3
  coordinates: [  0.500000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 207.200000
- symbol: Pb # 4
  coordinates: [  0.500000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 207.200000
- symbol: Te # 5
  coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 127.600000
- symbol: Te # 6
  coordinates: [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 127.600000
- symbol: Te # 7
  coordinates: [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 127.600000
- symbol: Te # 8
  coordinates: [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 127.600000
--------------------------------------------------------------------------------
[vasp.incar] (basic INCAR settings)
  ismear = 0
  sigma = 0.01
  ediff = 1e-08
  encut = 400
  prec = accurate
  lreal = False
  lwave = False
  lcharg = False
  ncore = 4
[phelel]
  supercell_dimension: [2 2 2]
[vasp.*.kpoints.mesh] (*kspacing=0.2)
  selfenergy: [8 8 8]*
  transport: [8 8 8]*
  el_bands: [8 8 8]*
  phonopy: [2 2 2]*
  phono3py: [2 2 2]*
  phelel: [2 2 2]*
  relax: [5 5 5]*
  nac: [8 8 8]*
[vasp.*.kpoints_dense.mesh] (*kspacing_dense=0.1)
  selfenergy: [17 17 17]*
  transport: [17 17 17]*
  el_bands: [17 17 17]*
Created new folder "relax".
Initial settings were written to "relax/velph.toml".
Found "POTCAR".
  PAW_PBE Pb 08Apr2002
  PAW_PBE Te 08Apr2002
  Max ENMAX in "POTCAR" is 174.982.
"POTCAR" was copied to "relax/POTCAR".
```

A directory `relax` is created and `velph.toml` is stored in it.

`relax/velph.toml`,
```toml
[phelel]
version = "0.8.2"
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = false
plusminus = true
fft_mesh = [30, 30, 30]

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
encut = 400
prec = "accurate"
lreal = false
lwave = false
lcharg = false
ncore = 4
[vasp.relax.kpoints]
mesh = [5, 5, 5]

# ...

[scheduler]
scheduler_name = "sge"
job_name = "PbTe"
mpirun_command = "mpirun"
vasp_binary = "/usr/local/vasp/bin/vasp_std"
pe = "vienna 32"
prepend_text = "source /opt/intel/oneapi/setvars.sh\n"
append_text = ""

[symmetry]
spacegroup_type = "Fm-3m"
tolerance = 1e-05
primitive_matrix = [
  [  0.000000000000000,  0.500000000000000,  0.500000000000000 ],
  [  0.500000000000000,  0.000000000000000,  0.500000000000000 ],
  [  0.500000000000000,  0.500000000000000,  0.000000000000000 ],
]

[unitcell]
lattice = [
  [     6.400000000000000,     0.000000000000000,     0.000000000000000 ], # a
  [     0.000000000000000,     6.400000000000000,     0.000000000000000 ], # b
  [     0.000000000000000,     0.000000000000000,     6.400000000000000 ], # c
]
[[unitcell.points]]  # 1
symbol = "Pb"
coordinates = [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
[[unitcell.points]]  # 2
symbol = "Pb"
coordinates = [  0.000000000000000,  0.500000000000000,  0.500000000000000 ]
[[unitcell.points]]  # 3
symbol = "Pb"
coordinates = [  0.500000000000000,  0.000000000000000,  0.500000000000000 ]
[[unitcell.points]]  # 4
symbol = "Pb"
coordinates = [  0.500000000000000,  0.500000000000000,  0.000000000000000 ]
[[unitcell.points]]  # 5
symbol = "Te"
coordinates = [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
[[unitcell.points]]  # 6
symbol = "Te"
coordinates = [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
[[unitcell.points]]  # 7
symbol = "Te"
coordinates = [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
[[unitcell.points]]  # 8
symbol = "Te"
coordinates = [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
[primitive_cell]
lattice = [
  [     0.000000000000000,     3.200000000000000,     3.200000000000000 ], # a
  [     3.200000000000000,     0.000000000000000,     3.200000000000000 ], # b
  [     3.200000000000000,     3.200000000000000,     0.000000000000000 ], # c
]
[[primitive_cell.points]]  # 1
symbol = "Pb"
coordinates = [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
[[primitive_cell.points]]  # 2
symbol = "Te"
coordinates = [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
```

Change directory to `relax`.
```
% velph relax generate
VASP input files were made in "relax/iter1".
% cd relax/iter1 && qsub _job.sh && cd ../..
% velph relax generate
"relax/iter1" exists.
"relax/iter1/CONTCAR" will be as new "POSCAR".
VASP input files were made in "relax/iter2".
```
Structure optimizations (iter1, iter2, ...) may be repeated until stress becomes
less than a target value, e.g., 0.1kB (0.01GPa). Then, go back to the top
directory and start supercell calculation.

```
% velph init --template-toml velph-tmpl.toml `ls relax/relax/iter*/CONTCAR|tail -n 1` calc
Read crystal structure file "relax/relax/iter2/CONTCAR".
Read velph template file "velph-tmpl.toml".
Following options were found in "velph-tmpl.toml":
  kspacing = "0.2"
  kspacing_dense = "0.1"
  max_num_atoms = "100"
The command options were prefered to [init.options] in "velph-tmpl.toml".
Found a primitive cell whose transformation matrix is
  [ 0.000  0.500  0.500]
  [ 0.500  0.000  0.500]
  [ 0.500  0.500  0.000]
Supercell is generated with respect to the cell below.
--------------------------------------------------------------------------------
lattice:
- [     6.440595899259139,     0.000000000000000,     0.000000000000000 ] # a
- [    -0.000000000000000,     6.440595899259139,     0.000000000000000 ] # b
- [    -0.000000000000000,    -0.000000000000000,     6.440595899259139 ] # c
points:
- symbol: Pb # 1
  coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 207.200000
- symbol: Pb # 2
  coordinates: [  0.000000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 207.200000
- symbol: Pb # 3
  coordinates: [  0.500000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 207.200000
- symbol: Pb # 4
  coordinates: [  0.500000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 207.200000
- symbol: Te # 5
  coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 127.600000
- symbol: Te # 6
  coordinates: [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 127.600000
- symbol: Te # 7
  coordinates: [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 127.600000
- symbol: Te # 8
  coordinates: [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 127.600000
--------------------------------------------------------------------------------
[vasp.incar] (basic INCAR settings)
  ismear = 0
  sigma = 0.01
  ediff = 1e-08
  encut = 400
  prec = accurate
  lreal = False
  lwave = False
  lcharg = False
  ncore = 4
[phelel]
  supercell_dimension: [2 2 2]
[vasp.*.kpoints.mesh] (*kspacing=0.2)
  selfenergy: [8 8 8]*
  transport: [8 8 8]*
  el_bands: [8 8 8]*
  phonopy: [2 2 2]*
  phono3py: [2 2 2]*
  phelel: [2 2 2]*
  relax: [5 5 5]*
  nac: [8 8 8]*
[vasp.*.kpoints_dense.mesh] (*kspacing_dense=0.1)
  selfenergy: [17 17 17]*
  transport: [17 17 17]*
  el_bands: [17 17 17]*
Created new folder "calc".
Initial settings were written to "calc/velph.toml".
Found "POTCAR".
  PAW_PBE Pb 08Apr2002
  PAW_PBE Te 08Apr2002
  Max ENMAX in "POTCAR" is 174.982.
"POTCAR" was copied to "calc/POTCAR".
```

Change directory to `calc`.
```
% cd calc
```

`velph.toml`,
```
[phelel]
version = "0.8.2"
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = false
plusminus = true
fft_mesh = [30, 30, 30]

[vasp.phelel.incar]
lwap = true
isym = 0
kpar = 2
ismear = 0
sigma = 0.01
ediff = 1e-08
encut = 400
prec = "accurate"
lreal = false
lwave = false
lcharg = false
ncore = 4
[vasp.phelel.kpoints]
mesh = [2, 2, 2]
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
encut = 400
prec = "accurate"
lreal = false
lwave = false
lcharg = false
[vasp.nac.kpoints]
mesh = [8, 8, 8]

# ...
```

NAC and phelel calculations,
```
% velph nac generate
VASP input files were made in "nac".
% cd nac && qsub _job.sh && cd ..
% velph phelel init
Found "nac" directory. Read NAC params.
"phelel/phelel_disp.yaml" was generated.
VASP input files will be generated by "velph phelel generate".
% velph phelel generate
VASP input files were generated in "phelel/disp-000".
VASP input files were generated in "phelel/disp-001".
VASP input files were generated in "phelel/disp-002".
VASP input files were generated in "phelel/disp-003".
VASP input files were generated in "phelel/disp-004".
% for i in {000..004};do cd phelel/disp-$i; qsub _job.sh; cd ../..;done
```

Using the VASP results of the phelel calculations, derivatives of properties
are computed using the phelel code. The results are stored in a newly created
file `phelel/phelel_params.hdf5`. This process creates only
`phelel/phelel_params.hdf5` and is invoked by the following velph command:
```
% velph phelel differentiate
FFT mesh: [30 30 30].
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
