# `velph` example

## PbTe

`POSCAR-unitcell`, `POTCAR`, and `velph-tmpl.toml` are located in the current
directory.

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
[init]
kspacing = 0.2
kspacing_dense = 0.1
max_num_atoms = 100
symmetrize_cell = true

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

`[init]` gives the defaults of the command-line options of `velph init` (see
{ref}`velph_init_template_init_options`). `max_num_atoms` needs
`symmetrize_cell = true`. `[vasp.incar]` gives the INCAR tags shared by the
input sets, and `velph init` writes them in the INCAR tags of each input set (see
{ref}`velph_init_template_incar`).

`POTCAR` choice,
```
  PAW_PBE Pb 08Apr2002
  PAW_PBE Te 08Apr2002
```

```
% velph init POSCAR-unitcell relax --template-toml velph-tmpl.toml
Read crystal structure file "POSCAR-unitcell".
Read velph init template "velph-tmpl.toml".
Space-group: Fm-3m
Crystal structure was standardized based on space-group-type Fm-3m.
Transformation matrix from conventional unit cell to primitive cell is
  [ 0.000  0.500  0.500]
  [ 0.500  0.000  0.500]
  [ 0.500  0.500  0.000]
Supercell is generated with respect to the cell below.
--------------------------------------------------------------------------------
lattice:
- [     6.400000000000001,     0.000000000000000,     0.000000000000000 ] # a
- [     0.000000000000000,     6.400000000000001,     0.000000000000000 ] # b
- [     0.000000000000000,     0.000000000000000,     6.400000000000001 ] # c
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
  coordinates: [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 127.600000
- symbol: Te # 6
  coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 127.600000
- symbol: Te # 7
  coordinates: [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 127.600000
- symbol: Te # 8
  coordinates: [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 127.600000
--------------------------------------------------------------------------------
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
[velph]
version = "0.15.0"

[phelel]
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = false
plusminus = true
nosym = false

# ...

[vasp.relax]
cell = "unitcell"

[vasp.relax.incar]
ismear = 0
sigma = 0.01
ediff = 1e-08
encut = 400
prec = "accurate"
lreal = false
lwave = false
lcharg = false
ncore = 4
ediffg = -1e-06
ibrion = 2
isif = 3
nsw = 10

[vasp.relax.kpoints]
mesh = [5, 5, 5]

# ...

[scheduler]
job_name = "PbTe"
mpirun_command = "mpirun"
vasp_binary = "/usr/local/vasp/bin/vasp_std"
prepend_text = """
source /opt/intel/oneapi/setvars.sh
"""
append_text = ""
nodes = 1
walltime = "96:00:00"
scheduler_template = """
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --time={walltime}
#SBATCH --output=ci_%j.log

{prepend_text}
{mpirun_command} {vasp_binary}
{append_text}
"""
scheduler_name = "sge"
pe = "vienna 32"

[symmetry]
tolerance = 1e-05
spacegroup_type = "Fm-3m"
primitive_matrix = [
  [0.0, 0.5, 0.5],
  [0.5, 0.0, 0.5],
  [0.5, 0.5, 0.0],
]

# ...
```

`scheduler_template` in `[scheduler]` is that of the default settings of
`velph init`. With `scheduler_name = "sge"`, the job script of SGE is written,
and `scheduler_template` is not used (see {ref}`velph_toml_scheduler`).

Change directory to `relax`.
```
% velph relax generate
VASP input files were generated in "relax/iter1".
% cd relax/iter1 && qsub _job.sh && cd ../..
% velph relax generate
"relax/iter1" exists.
"relax/iter1/CONTCAR" will be as new "POSCAR".
VASP input files were generated in "relax/iter2".
```
Structure optimizations (iter1, iter2, ...) may be repeated until stress becomes
less than a target value, e.g., 0.1kB (0.01GPa). Then, go back to the top
directory and start supercell calculation.

```
% velph init --template-toml velph-tmpl.toml `ls relax/relax/iter*/CONTCAR|tail -n 1` calc
Read crystal structure file "relax/relax/iter2/CONTCAR".
Read velph init template "velph-tmpl.toml".
Space-group: Fm-3m
Crystal structure was standardized based on space-group-type Fm-3m.
Transformation matrix from conventional unit cell to primitive cell is
  [ 0.000  0.500  0.500]
  [ 0.500  0.000  0.500]
  [ 0.500  0.500  0.000]
Supercell is generated with respect to the cell below.
--------------------------------------------------------------------------------
lattice:
- [     6.440595899259139,     0.000000000000000,     0.000000000000000 ] # a
- [     0.000000000000000,     6.440595899259139,     0.000000000000000 ] # b
- [     0.000000000000000,     0.000000000000000,     6.440595899259139 ] # c
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
  coordinates: [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 127.600000
- symbol: Te # 6
  coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 127.600000
- symbol: Te # 7
  coordinates: [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 127.600000
- symbol: Te # 8
  coordinates: [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 127.600000
--------------------------------------------------------------------------------
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
```toml
[vasp.nac]
cell = "primitive"

[vasp.nac.incar]
ismear = 0
sigma = 0.01
ediff = 1e-08
encut = 400
prec = "accurate"
lreal = false
lwave = false
lcharg = false
lepsilon = true

[vasp.nac.kpoints]
mesh = [9, 9, 9]

[vasp.phelel.incar]
ismear = 0
sigma = 0.01
ediff = 1e-08
encut = 400
prec = "accurate"
lreal = false
lwave = false
lcharg = false
ncore = 4
elph_prepare = true
isym = 0
kpar = 2

[vasp.phelel.kpoints]
mesh = [3, 3, 3]

[vasp.phelel.scheduler]
pe = "mpi* 48"

# ...
```

NAC and phelel calculations,
```
% velph nac generate
VASP input files were generated in "nac".
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

The FFT mesh is the FFT grid of VASP on the primitive cell for `encut` and
`prec` in the INCAR tags of the electron-phonon input sets (`selfenergy` and
`transport`), here `encut = 400` and `prec = "accurate"`. VASP stops when its FFT
grid differs from the mesh in `phelel/phelel_params.hdf5`, so these input sets
have to give the same mesh. When they give different meshes, or when one of them
does not give `encut`, `velph phelel differentiate` stops with an error.
`--encut` gives ENCUT in place of `encut` of these input sets.

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
