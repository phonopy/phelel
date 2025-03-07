(velph_toml_scheduler)=
# `velph.toml`-`[scheduler]` section

Job scripts `_job.sh` for calculation steps are generated following the settings
of the `[scheduler]` section. Calculation-step specific settings may be written
in `[vasp.CALC_TYPE.scheduler]`.

The string lines defined by the `scheduler_template` key are used as the
template of the job submission script `_job.sh`. String values of keys in this
section are inserted through the keyword argument style of [Python format string
syntax](https://docs.python.org/3/library/string.html#formatstrings). For
example, by `velph-relax-generate`

```toml
[scheduler]
scheduler_template = '''#!/bin/bash
# key-{key}
# jobname: {job_name}
{myargument}
'''
key = "value"
job_name = "mycalc"
myargument = "sleep 10"
```

results in

```
#!/bin/bash
# key-value
# jobname: mycalc-iter1
sleep 10
```

as `_job.sh`. The `job_name` key is a special one. For supercell
calculations, the value is replaced by `{job_name}-000`, `{job_name}-001`, ...,
and for `velph-relax` calculatios `{job_name}-iter1`, `{job_name}-iter2`, ...

If `[vasp.CALC_TYPE.scheduler]` is specified, `[scheduler]` settings are
overwritten by the settings for `[vasp.CALC_TYPE]`.

An example is shown below.

```toml
[scheduler]
scheduler_template = '''#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m n
#$ -N {job_name}
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt
#$ -pe {pe}

source /opt/intel/oneapi/setvars.sh
'''
job_name = "PbTe"
pe = "vienna 24"

...
[vasp.phelel.scheduler]
scheduler_template = '''#!/bin/bash
#QSUB2 core 192
#QSUB2 mpi 192
#QSUB2 smp 1
#QSUB2 wtime 48:00:00
#PBS -N {job_name}
cd $PBS_O_WORKDIR

. /etc/profile.d/modules.sh
module load inteloneapi22u3

mpirun /usr/local/cluster-1/bin/vasp_std | tee vasp_output
'''
job_name = "PbTe"

...
[vasp.phono3py.scheduler]
pe = "paris 24"
```
