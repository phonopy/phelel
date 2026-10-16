(velph_toml_scheduler)=
# `velph.toml`-`[scheduler]` section

The generate commands, such as `velph relax generate` and `velph phelel
generate`, write a job script `_job.sh` in the directory of each calculation.
The job script is made from `[scheduler]` and `[vasp.<input set>.scheduler]` of
`velph.toml`. This page gives all the rules by which it is made, so that the
page can also be given to a large language model (LLM) to write
`scheduler_template` (see {ref}`velph_toml_scheduler_llm`).

## How `_job.sh` is written

For each calculation, a generate command writes `_job.sh` in the following
steps.

1. The settings of the calculation are the keys of `[scheduler]` and those of
   `[vasp.<input set>.scheduler]` of its input set. A key in
   `[vasp.<input set>.scheduler]` replaces the same key of `[scheduler]`.
2. The name of the calculation is appended to the value of `job_name`:

   | Calculations                                                                                        | `job_name`                                  |
   | --------------------------------------------------------------------------------------------------- | ------------------------------------------- |
   | `velph relax generate`                                                                              | `{job_name}-iter1`, `{job_name}-iter2`, ... |
   | Supercells of `velph phelel generate`, `velph phonopy generate` and `velph phono3py generate` | `{job_name}-000`, `{job_name}-001`, ...     |
   | The other generate commands                                                                         | `{job_name}`                                |

3. Each key in braces in `scheduler_template`, such as `{job_name}`, is replaced
   by the value of the key. This is done by the format method of Python strings
   (see [format string
   syntax](https://docs.python.org/3/library/string.html#formatstrings)).
4. The result is written as `_job.sh`.

When `velph.toml` has no `[scheduler]`, `_job.sh` is not written, also when
`[vasp.<input set>.scheduler]` is given.

For example, `velph relax generate` with

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

writes the following `relax/iter1/_job.sh`:

```
#!/bin/bash
# key-value
# jobname: mycalc-iter1
sleep 10
```

`velph scheduler` shows `_job.sh` of each input set without writing any
file (see {ref}`velph_subcommands_scheduler`).

## Keys

- `scheduler_template` : the template of `_job.sh`, a string. It is required.
- `job_name` : the name of the job. The name of the calculation is appended to
  it.
- Any other key can be given, with any name. Its value is a string, an integer,
  a float or a boolean. A key that `scheduler_template` does not have in braces
  is not used.

```{note}
`scheduler_name` of velph 0.14, which chose a job script of SGE or Slurm written
in velph, is removed. A `velph.toml` or a template with `scheduler_name` is an
error. See {ref}`velph_changes_0_15_scheduler` for the `scheduler_template` that
gives the same job script.
```

## Writing `scheduler_template`

Write `scheduler_template` as a multi-line literal string of TOML, between
`'''` and `'''`. The text between them is used as it is, except that the line
break right after the first `'''` is removed. Between `"""` and `"""`, a
backslash starts an escape sequence, and a backslash at the end of a line joins
the line with the next one. For example, `mpirun \` and `vasp_std` on two
lines become `mpirun vasp_std` on one line.

To write `{` or `}` in `_job.sh`, write `{{` or `}}` in `scheduler_template`.
Job scripts often have braces, for example:

| In the job script     | In `scheduler_template` |
| --------------------- | ----------------------- |
| `${SLURM_NTASKS}`     | `${{SLURM_NTASKS}}`     |
| `awk '{print $1}'`    | `awk '{{print $1}}'`    |
| `f() { echo done; }`  | `f() {{ echo done; }}`  |

`$SLURM_NTASKS` without braces needs no change.

## Errors

A generate command stops with one of the following errors before it writes any
file. `velph scheduler` shows these errors for each input set.

- `"<key>" used in the job script template is not given in [scheduler] or
  [vasp.<input set>.scheduler].` : `scheduler_template` has `{<key>}`, but the
  settings do not have `<key>`. When `<key>` is not meant as a key, for example
  `print $1` of `awk '{print $1}'`, write `{{` and `}}` in place of its braces.
- `The job script template cannot be read: ...` : `scheduler_template` has a
  single `{` or `}`, or `{}` or `{0}`. To write `{` or `}` in `_job.sh`, write
  `{{` or `}}`.
- `"scheduler_template" has to be given in [scheduler] or [vasp.<input
  set>.scheduler].` : the settings of the calculation do not have
  `scheduler_template`.

The following errors are given when `velph.toml` is read:

- `"scheduler_name" is removed.` and `"custom_template" is removed.` : these
  keys of earlier versions of velph cannot be given.
- `"scheduler_template" has to be a string` : `scheduler_template` is not a
  string.
- `"<key>" has a table as value.` : the value of a key is a table, which
  cannot be used.

## `[scheduler]` in a template of `velph init`

`velph init` writes `[scheduler]` in `velph.toml` only when its template has
`[scheduler]`, and it writes the `[scheduler]` of the template as it is.
`velph init` has no default `[scheduler]`. Without `[scheduler]` in the
template, `velph.toml` has no `[scheduler]`, and the generate commands write no
`_job.sh`. To write job scripts, give `[scheduler]` with `scheduler_template` in
the template, or add it to `velph.toml`.

(velph_toml_scheduler_llm)=
## Writing `scheduler_template` with an LLM

A job script that works on your computer can be converted to
`scheduler_template` by an LLM. Give the LLM the text of this page and the job
script, for example with the following request:

```
Write scheduler_template of the [scheduler] section of velph.toml from the job
script below, following the rules of the attached page. Replace the job name by
{job_name}, write "{{" or "}}" for "{" or "}" of the job script, and write the
template as a multi-line literal string of TOML.

(the job script)
```

Read the `scheduler_template` that the LLM writes, and check `_job.sh` with
`velph scheduler` (see {ref}`velph_subcommands_scheduler`).

## Examples

### Slurm

```toml
[scheduler]
scheduler_template = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --ntasks={ntasks}
#SBATCH --time={walltime}
#SBATCH --output=%x-%j.out

module load intel-oneapi
mpirun -np ${{SLURM_NTASKS}} /usr/local/vasp/bin/vasp_std
'''
job_name = "PbTe"
partition = "short"
ntasks = 48
walltime = "12:00:00"

...
[vasp.phelel.scheduler]
partition = "long"
ntasks = 192
walltime = "48:00:00"

...
[vasp.phono3py.scheduler]
ntasks = 96
```

`velph nac generate` writes the following `nac/_job.sh`. `${{SLURM_NTASKS}}` is
written as `${SLURM_NTASKS}`, which the shell replaces when the job runs.

```
#!/bin/bash
#SBATCH --job-name=PbTe
#SBATCH --partition=short
#SBATCH --ntasks=48
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

module load intel-oneapi
mpirun -np ${SLURM_NTASKS} /usr/local/vasp/bin/vasp_std
```

`velph phelel generate` writes the following `phelel/disp-001/_job.sh`, where
`partition`, `ntasks` and `walltime` are those of `[vasp.phelel.scheduler]`.
`[vasp.phono3py.scheduler]` replaces only `ntasks`.

```
#!/bin/bash
#SBATCH --job-name=PbTe-001
#SBATCH --partition=long
#SBATCH --ntasks=192
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out

module load intel-oneapi
mpirun -np ${SLURM_NTASKS} /usr/local/vasp/bin/vasp_std
```

### PBS

```toml
[scheduler]
scheduler_template = '''#!/bin/bash
#PBS -N {job_name}
#PBS -q {queue}
#PBS -l select={nodes}:ncpus={ncpus}:mpiprocs={ncpus}
#PBS -l walltime={walltime}

cd ${{PBS_O_WORKDIR}}
module load intel-oneapi
mpirun /usr/local/vasp/bin/vasp_std
'''
job_name = "PbTe"
queue = "default"
nodes = 1
ncpus = 48
walltime = "12:00:00"
```

`cd ${{PBS_O_WORKDIR}}` is written as `cd ${PBS_O_WORKDIR}` in `_job.sh`.

### SGE

```toml
[scheduler]
scheduler_template = '''#!/bin/bash
#$ -N {job_name}
#$ -cwd
#$ -pe {pe} {slots}
#$ -l h_rt={walltime}

module load intel-oneapi
mpirun -np ${{NSLOTS}} /usr/local/vasp/bin/vasp_std
'''
job_name = "PbTe"
pe = "mpi"
slots = 48
walltime = "12:00:00"
```

`${{NSLOTS}}` is written as `${NSLOTS}` in `_job.sh`.
