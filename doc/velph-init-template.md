(velph_init_template)=
# `velph init` template

`velph init` is the command to prepare `velph.toml`. Without specifying a
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
