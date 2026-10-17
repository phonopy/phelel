(velph_changes_0_15)=
# Changes in velph 0.15

velph 0.15 changes the format of `velph.toml` and of the `velph init` template,
and some subcommands. This page lists the changes from velph 0.14.

```{note}
To write a `velph.toml` of velph 0.14 in the format of velph 0.15, run
`velph convert`. See {ref}`velph_changes_0_15_convert`.
```

(velph_changes_0_15_project_folder)=
## Project folders made by velph 0.14

A project folder made by `velph init` of velph 0.14 has `velph.toml` and the
directories of the calculations. This section describes how velph 0.15 uses
such a folder.

(velph_changes_0_15_convert)=
### Converting `velph.toml` with `velph convert`

`velph convert` writes a `velph.toml` of velph 0.14 in the format of velph 0.15.
It does not change the original file, and it does not overwrite an existing
file. It prints the settings that were removed and the k-point blocks whose
`kspacing` was replaced by `mesh`:

```
% velph convert velph.toml velph-new.toml
fft_mesh of [phelel] was removed.
"velph-new.toml" has been written from "velph.toml".
```

To use the new file, rename `velph-new.toml` to `velph.toml`.

The velph subcommands also read a `velph.toml` of velph 0.14 without
`velph convert`. They convert the settings to those of velph 0.15 when they
read the file, and they do not change the file. In both cases, `kspacing` in a
k-point block is converted to `mesh` in the same way as velph 0.14 does.

### Settings that velph 0.15 does not use

The following settings of velph 0.14 have no place in velph 0.15.
`velph convert` removes them, and the subcommands do not use them:

- `fft_mesh` of `[phelel]`. `velph phelel differentiate` computes the FFT mesh
  (see {ref}`velph_changes_0_15_subcommands`).
- `phonon_supercell_dimension` and `phonon_supercell_matrix` of `[phelel]` and
  `[phono3py]`, and `[vasp.phelel.phonon]` and `[vasp.phono3py.phonon]`.
  `velph phelel init` writes `phelel/phelel_disp.yaml` without the phonon
  supercell and prints no message about it. To use a phonon supercell, see
  {ref}`velph_changes_0_15_phonon_supercell`.
- `[vasp.incar]`.

A key that velph 0.15 does not know is an error, also in a `velph.toml` of
velph 0.14. velph 0.14 did not use such a key and printed no message.

`scheduler_name` in `[scheduler]` or in `[vasp.<input set>.scheduler]` is an
error, also in a `velph.toml` of velph 0.14, and `velph convert` writes no file
then. Replace it with `scheduler_template` (see
{ref}`velph_changes_0_15_scheduler`).

### Directories with new names

The results of velph 0.14 in the following directories are not read by velph
0.15, because the directories have new names (see
{ref}`velph_changes_0_15_subcommands`):

- `el_bands/bands` and `el_bands/dos`, which are now `el_bands` and `el_dos`.
- `ph_bands/bands`, which is now `ph_bands`.
- `phelel/ph-disp-*`, the calculations of the phonon supercell. When
  `phelel/phelel_disp.yaml` has the phonon supercell, `velph phelel
  differentiate` reads these calculations from `phonopy/disp-*`, and it stops
  when `phonopy/phonopy_disp.yaml` does not exist.

## `velph.toml`

See {ref}`velph_toml` for the format of velph 0.15.

- `[velph] version` replaces `version` of `[phelel]`. A `velph.toml` without
  `[velph] version` is read as a file of velph 0.14.
- `[vasp.el_bands.bands]` and `[vasp.el_bands.dos]` are now `[vasp.el_bands]`
  and `[vasp.el_dos]`.
- `fft_mesh` of `[phelel]` is removed.
- `phonon_supercell_dimension` and `phonon_supercell_matrix` of `[phelel]` and
  `[phono3py]`, and `[vasp.phelel.phonon]` and `[vasp.phono3py.phonon]`, are
  removed.
- `[vasp.incar]` and `incar_unset` are settings of a template only. Each input
  set of `velph.toml` has all of its INCAR tags, and `velph.toml` with
  `[vasp.incar]` or `incar_unset` is an error.
- `kspacing` in a k-point block is an error. `velph init` converts it to `mesh`.
- `velph init` writes `nosym` in `[phelel]`, `[phonopy]` and `[phono3py]`.
- `velph phono3py init` uses `number_of_snapshots` of `[phono3py]` when `--rd`
  is not given.
- `scheduler_name` of `[scheduler]` and `[vasp.<input set>.scheduler]` is
  removed. The job script is written from `scheduler_template` only (see
  {ref}`velph_changes_0_15_scheduler`).
- A key that velph does not know is an error.

## `velph init` template

See {ref}`velph_init_template` for the template of velph 0.15. `velph init`
stops with an error for a template of velph 0.14, and the error lists the
changes that concern the file.

- `[init.options]` is now `[init]`. The keys of `[init]` are those of the
  command-line options of `velph init` (see
  {ref}`velph_init_template_init_options`). `amplitude`, `diagonal`,
  `plusminus`, `nosym` and the supercell keys in `[init]` are used for all the
  codes in `codes`.
- `phelel_nosym` of `[init.options]` is now `nosym` of `[init]`.
- An INCAR tag set to an empty table `{}` is not used. To exclude a tag of
  `[vasp.incar]` from an input set, list it in `incar_unset` of the input set
  (see {ref}`velph_init_template_incar`).
- `kspacing` can be given in every `kpoints` and `kpoints_dense` block,
  including `[vasp.ph_bands.kpoints]`.
- `scheduler_name` is removed, as in `velph.toml`.

(velph_changes_0_15_scheduler)=
## Job scripts of `scheduler_name`

With `scheduler_name = "sge"` or `"slurm"`, velph 0.14 wrote a job script of SGE
or Slurm that was written in velph, and it did not use `scheduler_template`.
velph 0.15 writes the job script from `scheduler_template` only. To write the
job script that velph 0.14 wrote, remove `scheduler_name` and give the
following `scheduler_template`.

For `scheduler_name = "sge"`,

```toml
scheduler_template = '''#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m n
#$ -N {job_name}
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt
#$ -pe {pe}
#$ -l h_rt={walltime}

{prepend_text}

{mpirun_command} {vasp_binary}

{append_text}
'''
```

velph 0.14 wrote the line of `-pe` only when `pe` was given, and the line of
`-l h_rt` only when `walltime` was given. Remove the line of a key that is not
given. `velph init` of velph 0.14 wrote `walltime = "96:00:00"` by default.
When `pe` is given only in some `[vasp.<input set>.scheduler]`, give
`scheduler_template` with the line of `-pe` in those sections.

For `scheduler_name = "slurm"`,

```toml
scheduler_template = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --time=96:00:00      # Time limit hrs:min:sec
#SBATCH --output=ci_%j.log   # Standard output and error log

{prepend_text}

{mpirun_command} {vasp_binary}

{append_text}
'''
```

velph 0.14 wrote `--time=96:00:00` and did not use `walltime`. To use
`walltime`, write `--time={walltime}`.

(velph_changes_0_15_subcommands)=
## Subcommands

See {ref}`velph_init` and {ref}`velph_subcommands` for the subcommands of velph
0.15.

### `velph convert`

`velph convert` is a new subcommand, which writes a `velph.toml` of velph 0.14
in the format of velph 0.15 (see {ref}`velph_changes_0_15_convert`).

### `velph init`

- `--code` chooses the codes whose sections are written. Repeat it for more
  than one code.
- `--nosym` replaces `--phelel-nosym`, and it is used for all the chosen codes.
- When `velph init` stops with an error, the exit code is 1. velph 0.14 gave 0.
- With `--symmetrize-cell`, the k-point meshes of the input sets on the unit
  cell are computed with the symmetry of the unit cell. velph 0.14 used the
  symmetry of the input cell. For example, for a rhombohedral input cell, the
  mesh on the hexagonal unit cell can differ along the c axis.
- With `--max-num-atoms`, the supercell has at most that number of atoms, also
  with `--no-find-primitive`. With a primitive cell as input, velph 0.14 could
  give a larger supercell.
- When a supercell has fewer rotations than the unit cell, `velph init` prints
  a warning.
- `velph init` writes `[scheduler]` only when its template has `[scheduler]`,
  and it writes the `[scheduler]` of the template as it is. velph 0.14 wrote a
  default `[scheduler]` with a job script of Slurm, which `[scheduler]` of a
  template updated key by key. See {ref}`velph_toml_scheduler`.

### `velph el_bands`, `velph el_dos` and `velph ph_bands`

- `velph el_bands generate` writes the input files in `el_bands` only, in place
  of `el_bands/bands` and `el_bands/dos`.
- `velph el_dos generate` is a new subcommand, which writes the input files of
  the DOS calculation in `el_dos`.
- `velph el_dos plot` is a new subcommand, which plots the DOS.
- `velph el_bands plot_eigenvalues` is now `velph el_dos plot_eigenvalues`.
- `velph ph_bands generate` writes the input files in `ph_bands`, in place of
  `ph_bands/bands`.
- The band paths that seekpath gives are written in the basis of
  `[primitive_cell]`. velph 0.14 wrote the coordinates in the basis that
  seekpath chose, which put the points in wrong places when the two bases
  differ, for example for a body-centred tetragonal cell with
  `--primitive-cell-choice reduced`.

(velph_changes_0_15_phonon_supercell)=
### Phonon supercells

The phonon supercells of phelel and phono3py are taken from phonopy:

1. `velph phonopy init` and `velph phonopy generate` make the phonon
   supercells in `phonopy`.
2. `velph phelel init --phonon-from-phonopy` and `velph phono3py init
   --phonon-from-phonopy` take the phonon supercells from
   `phonopy/phonopy_disp.yaml`.
3. `velph phelel differentiate` reads the calculations of the phonon
   supercells from `phonopy/disp-*`.

`velph phelel generate` and `velph phono3py generate` no longer write
`ph-disp-*`. See {ref}`velph_phelel_init_subcommand`.

### `velph phelel differentiate`

The FFT mesh written in `phelel/phelel_params.hdf5` is computed from `encut` and
`prec` in the INCAR tags of the electron-phonon input sets, or from `--encut`
(see {ref}`velph_phelel_differentiate_subcommand`). velph 0.14 read `fft_mesh`
of `[phelel]`. A missing `prec` is taken as Normal, the default of VASP.

### Other subcommands

- `check-fft` of `velph selfenergy` and `velph transport` is removed.
- When `CONTCAR` of the previous step does not exist, `velph relax generate`
  stops with an error and does not make the next directory. velph 0.14 made
  the next directory empty.
