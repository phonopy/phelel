(changelog)=

# Change Log

## Unreleased

- velph is redesigned, and the format of `velph.toml` and of the `velph init`
  template changes. See {ref}`velph_changes_0_15` for the changes from velph
  0.14.
- `velph convert` writes a `velph.toml` of velph 0.14 in the format of velph
  0.15.
- `scheduler_name` of the scheduler settings is removed, and the job script is
  written from `scheduler_template` only. See
  {ref}`velph_changes_0_15_scheduler` for the `scheduler_template` that gives
  the job script of velph 0.14.
- A `scheduler_template` with a single `{` or `}` stops the generate commands
  with an error on braces, in place of a traceback.
- `velph init` writes `[scheduler]` only when its template has `[scheduler]`.
  Without it, no job script `_job.sh` is written.
- When `scheduler_template` cannot be filled, the generate commands stop before
  they write any file. Before, the directory was left without `_job.sh`.
- `velph scheduler` shows the job script `_job.sh` of each input set without
  writing any file.
- `velph relax generate` without `velph.toml` stops with an error. Before, the
  exit code was 0.
- The plot commands without `vaspout.h5` stop with an error. Before, the exit
  code was 0, or `velph ph_bands plot` and `velph ph_selfenergy plot
  selfenergy` stopped with a traceback.
- Fix the force constants of the derivatives with a phonon supercell. They were
  computed from the displacements of the phelel supercells and the forces of
  the phonon supercells, and they are now computed from the displacements of
  the phonon supercells.
- A phonon supercell with random displacements of all atoms can be used for the
  derivatives. Its force constants are computed by symfc.
- The forces in `vaspout.h5` are read by phonopy, which takes those of the last
  ionic step. Before, those of the first ionic step were read.
- `velph phelel phonopy` reads `vaspout.h5` as well as `vasprun.xml`.

## Oct-9-2026: Version 0.14.0

- Python 3.10 is no longer supported. Python 3.11 or later is required.
- phonopy 4.8.2 or later and phono3py 4.8.0 or later are required.
- Major refactoring of velph.
- velph computes a k-point mesh from `kspacing` by rounding up, as VASP
  `KSPACING` does, so the mesh can be larger than before.
- `velph init` writes `amplitude = 0.01` in `[phonopy]` instead of 0.03 by
  default.

## May-13-2026: Version 0.13.3

- Fix import errors.

## May-13-2026: Version 0.13.2

- Update following the change of phonopy and phono3py.

## Mar-8-2026: Version 0.13.1

- Rename `cmd_root.py` to `velph_cmd_root.py` for safer module import when
  calling velph command.

## Mar-6-2026: Version 0.13.0

- Maintenance release

## Feb-16-2026: Version 0.12.0

- Maintenance release

## Oct-24-2025: Version 0.11.0

- Default INCAR tags are removed by setting empty table {} in velph template
  toml.
- Use generalized regular grid when k-point grid symmetry is broken.

## Jun-27-2025: Version 0.10.0

- Update following the change of phonopy.

## May-1-2025: Version 0.9.1

- Update following the change of phonopy.

## Mar-23-2025: Version 0.9.0

- Numpy arraies of `dtype='long'` were changed to those of `dtype='int64'`.
- Collection of updates in velph commands.

## Mar-1-2025: Version 0.8.3

- `--save` option was implemented for `velph-el_bands-plot` and
  `velph-ph_bands-plot`. Without this option, the plot is shown on display.

## Jan-10-2025: Version 0.8.2

- Collection of small updates of velph command

## Jan-8-2025: Version 0.8.1

- Fix minor bugs in velph command

## Jan-5-2025: Version 0.8.0

- Refactoring of `Phelel` class
- Major update of documentation

## Dec-31-2024: Version 0.7.0

- Maintenance release to follow the change of phonopy.
- `elph_selfen_band_stop` is estimated from el-DOS instead of `elph_nbands` for
  transport mode in velph.

## Dec-9-2024: Version 0.6.6

- Collection of minor updates of velph command.
