(velph_toml)=
# `velph.toml` file

The `velph.toml` file contains the settings required to perform electron-phonon
interaction calculations using the phelel and VASP codes. Ideally, once created,
it should remain unchanged throughout the duration of a calculation project for
the reproducibility.

There are several ways to create a `velph.toml` file:

- Using the default template (see {ref}`velph_init` command)
- Using a user-defined template (`velph init --template-toml`), as explained in
  {ref}`velph_init_template`
- Copying and modifying an existing `velph.toml` file

Regardless of how it is created, the `velph.toml` file is used in conjunction
with the `velph` command to generate VASP input files and operate other tools,
based on the information it contains. This page provides guidance on how to
interpret and effectively use the contents of `velph.toml`.

## `[phelel]`

This section contains information specific to the phelel code and must be
included to execute the {ref}`velph_phelel_subcommand` subcommand.

```toml
[phelel]
version = "0.6.5"
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = false
plusminus = true
fft_mesh = [30, 30, 30]
```

## `[vasp]`

There are sub-sections for respective specific calculations:

- `[vasp.phelel]`
- `[vasp.relax]`
- `[vasp.nac]`
- `[vasp.el_bands]`
- `[vasp.ph_bands]`
- `[vasp.phono3py]`
- `[vasp.phonopy]`

In each section, VASP calculation and job script settings can be written, e.g.,

```toml
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
gga = "PS"
[vasp.phelel.kpoints]
mesh = [4, 4, 4]
shift = [0.500000, 0.500000, 0.500000]
[vasp.phelel.scheduler]
pe = "mpi* 48"
```

When `velph.toml` is generated from a template using the `velph init
--template-toml` command, not all information needs to be explicitly specified.
Default settings for each calculation can automatically populate the relevant
sections, leveraging the VASP INCAR parameters defined in the {ref}`[vasp.incar]
<velph_init_template_incar>` section of the template.

## `[scheduler]` : Job script setting

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

...
[vasp.phono3py.scheduler]
pe = "paris 24"
```

## `[phono3py]`

This section contains information specific to the phono3py code and must be
included to execute the {ref}`velph_phono3py_subcommand` subcommand.

```toml
[phono3py]
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = false
plusminus = true
```

## `[phonopy]`

This section contains information specific to the phono3py code and must be
included to execute the {ref}`velph_phonopy_subcommand` subcommand.

```toml
[phonopy]
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = false
plusminus = true
```
