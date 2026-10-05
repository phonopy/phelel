(changelog)=

# Change Log

## Unreleased

- `velph init`: command-line options and `[init.options]` now take
  precedence over `[phelel]`, `[phonopy]`, and `[phono3py]` of the template
  for `amplitude`, `diagonal`, and `plusminus`, as for the supercell matrix.
- `velph init`: giving two or more of `max_num_atoms`,
  `supercell_dimension`, and `supercell_matrix` in one place is an error.
  A supercell option on the command line is no longer overridden by
  `max_num_atoms` in `[init.options]`.
- `velph init`: INCAR tag names in `[vasp.incar]` and
  `[vasp.CALC_TYPE.incar]` of the template are case-insensitive, as
  documented. Upper-case tags were ignored or raised an error.
- `velph init`: `kspacing` in a k-point block of the template is copied
  to `velph.toml`, and the mesh is computed by the generate commands.
- velph always writes the k-point files. The INCAR tags `kspacing` and
  `elph_kspacing` are an error in `velph init` templates and in
  `velph.toml`.
- velph computes a k-point mesh from `kspacing` by rounding up, as the VASP
  INCAR tag `KSPACING` does. It was rounded to the nearest integer, so the
  mesh can be larger than before. phonopy 4.8.1 or later is required for
  this.
- `velph phelel init`, `velph phonopy init`, and `velph phono3py init` use
  `plusminus = true` and `diagonal = false` when these keys are missing in
  `velph.toml`, as `velph init` writes and as documented. They used the
  defaults of phonopy (`"auto"` and `true`).
- `velph init` writes `amplitude = 0.01` in `[phonopy]` instead of 0.03 when
  the amplitude is not given. `[phelel]` and `[phono3py]` keep 0.03. These are
  the default displacement distances of phonopy and phono3py for VASP, which
  the generate commands also use when `amplitude` is missing.

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
