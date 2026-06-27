(velph_toml_vasp)=
# `velph.toml`-`[vasp]` section

The `[vasp]` section is the largest part of `velph.toml`. It stores the VASP
`INCAR` tags, k-point sampling, and per-step job-script settings used to generate
the input files for each calculation step of the workflow.

Each calculation step has its own sub-section `[vasp.CALC_TYPE]`. The recognized
calculation types are:

| `CALC_TYPE`        | Purpose                                              | Subcommand                          |
| ------------------ | ---------------------------------------------------- | ----------------------------------- |
| `relax`            | Crystal structure relaxation                         | {ref}`velph relax <velph_subcommands>` |
| `nac`              | Dielectric constant and Born effective charges       | {ref}`velph nac <velph_nac_subcommand>` |
| `phelel`           | Supercells for derivatives of potentials/PAW strengths | {ref}`velph phelel <velph_phelel_subcommand>` |
| `phonopy`          | Supercells for phonon (phonopy)                      | {ref}`velph phonopy <velph_phonopy_subcommand>` |
| `phono3py`         | Supercells for anharmonic phonon (phono3py)          | {ref}`velph phono3py <velph_phono3py_subcommand>` |
| `selfenergy`       | Fan self-energy (electron-phonon)                    | `velph selfenergy`                  |
| `transport`        | Transport properties (electron-phonon)               | `velph transport`                   |
| `el_bands`         | Electronic band structure and DOS                    | `velph el_bands`                    |
| `ph_bands`         | Phonon band structure                                | `velph ph_bands`                    |
<!-- Hidden until ph_selfenergy is public in VASP:
| `ph_selfenergy`    | Phonon self-energy (superconductivity)               | `velph ph_selfenergy`               |
-->


Two calculation types are split into sub-types that are calculated separately and
therefore have their own sub-sections:

- `[vasp.el_bands.bands]` and `[vasp.el_bands.dos]`
- `[vasp.phono3py.phonon]` (the larger phonon supercell used together with the
  smaller third-order supercell)

## Common sub-block structure

With a few exceptions noted below, every `[vasp.CALC_TYPE]` is built from the
same four kinds of sub-blocks:

```toml
[vasp.CALC_TYPE.incar]      # INCAR tags
[vasp.CALC_TYPE.kpoints]    # k-point sampling mesh
[vasp.CALC_TYPE.kpoints_dense]  # dense mesh (only some calc types)
[vasp.CALC_TYPE.scheduler]  # per-step job-script overrides
```

- `incar` is written for all calculation types.
- `kpoints` is written for all calculation types. For band-structure steps it
  holds the regular mesh and is accompanied by a separate line-mode block
  (`kpoints_opt` or `qpoints`, see below).
- `kpoints_dense` is written only for the electron-phonon steps (`selfenergy`,
  `transport`<!-- , `ph_selfenergy` -->) and for `el_bands.dos`. It provides the dense
  charge-density mesh used for the "sandwich" calculation.
- `scheduler` is optional. When present, it overrides the top-level
  {ref}`[scheduler] <velph_toml_scheduler>` settings for this step only.

The `relax` and `nac` sections additionally carry a `cell` key:

```toml
[vasp.relax]
cell = "unitcell"
[vasp.nac]
cell = "primitive"
```

`cell` selects whether the unit cell or the primitive cell is used for that step.
The defaults are `unitcell` for `relax` and `primitive` for `nac`, and they can
be set with the `velph init` options `--cell-for-relax` and `--cell-for-nac`
(see {ref}`velph_init`).

(velph_toml_vasp_incar)=
## INCAR settings

### `[vasp.incar]` : base settings

`[vasp.incar]` holds the base INCAR tags shared by all calculation types. It does
not correspond to a calculation step on its own; instead its tags are merged into
every `[vasp.CALC_TYPE.incar]` (see the layering rule below). Settings that are
common to the whole project, such as the plane-wave cutoff (`encut`) and the
parallelization tags (`ncore`, `kpar`, ...), belong here.

```toml
[vasp.incar]
ismear = 0
sigma = 0.01
ediff = 1e-8
encut = 500
prec = "accurate"
lreal = false
lwave = false
lcharg = false
```

The values above are the defaults written by `velph init`.

### `[vasp.CALC_TYPE.incar]` and layering

INCAR parameters are propagated in two stages. `velph init` merges the base
settings into each calculation type and writes the fully merged result into
`velph.toml`. Later, `velph CALC_TYPE generate` writes that merged block verbatim
to the `INCAR` file.

```{mermaid}
flowchart LR
  subgraph INIT["velph init"]
    direction TB
    DEF["defaults:<br/>base vasp.incar<br/>+ per-calc-type"]
    TMPL["template toml:<br/>vasp.incar /<br/>vasp.CALC_TYPE.incar"]
    MERGE{{"merge base into<br/>each calc type;<br/>drop empty tags"}}
    DEF --> MERGE
    TMPL --> MERGE
  end
  MERGE --> MERGED["velph.toml:<br/>vasp.CALC_TYPE.incar<br/>(fully merged)"]
  MERGED --> GEN["velph CALC_TYPE<br/>generate"]
  GEN --> INCAR["INCAR file"]
```

```{note}
The base-to-calc-type merge happens **only** at `velph init`. In a generated
`velph.toml`, the base `[vasp.incar]` section is informational: `generate` reads
only `[vasp.CALC_TYPE.incar]`. To change a tag after initialization, edit the
relevant `[vasp.CALC_TYPE.incar]`, not `[vasp.incar]`.
```

INCAR tags are written as plain key-value pairs. Tag names are case-insensitive
and are normalized to lower case:

```toml
[vasp.phelel.incar]
elph_prepare = true
isym = 0
encut = 500
ismear = 0
sigma = 0.01
ediff = 1e-08
prec = "accurate"
lreal = false
lwave = false
lcharg = false
```

The base INCAR settings are defined once in the `[vasp.incar]` section and are
merged into every `[vasp.CALC_TYPE.incar]` for any tag that the calculation type
does not define itself. See {ref}`velph_init_template_incar` for how this base
section is used by the template. The effective INCAR for a step is therefore:

1. the tags written in `[vasp.CALC_TYPE.incar]`, overriding
2. the base tags from `[vasp.incar]`.

To **suppress** an inherited base tag for a particular calculation type, set it to
an empty table `{}`. The tag is then removed instead of being inherited. This is
how, e.g., `ph_bands` drops the smearing and I/O tags that the base section
provides:

```toml
[vasp.ph_bands.incar]
ibrion = -1
nsw = 0
elph_run = true
ismear = {}
sigma = {}
ediff = {}
lreal = {}
lwave = {}
lcharg = {}
```

```{note}
The empty-table suppression only matters in the *template* (`velph-tmpl.toml`)
and in the default settings, because that is where the base-to-calc-type merge
happens. In a generated `velph.toml`, each `[vasp.CALC_TYPE.incar]` already
contains the fully merged result.
```

### Default INCAR tags by calculation type

In addition to the base `[vasp.incar]` tags, each calculation type contributes
the following defaults:

| `CALC_TYPE`        | Default INCAR tags                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| `relax`            | `ediffg = -1e-6`, `ibrion = 2`, `isif = 3`, `nsw = 10`                                               |
| `nac`              | `lepsilon = true` (and `npar`/`ncore`/`kpar` suppressed with `{}`)                                   |
| `phelel`           | `elph_prepare = true`, `isym = 0`                                                                    |
| `phonopy`          | `addgrid = true`, `isym = 0`                                                                          |
| `phono3py`         | `addgrid = true`, `isym = 0`                                                                          |
| `selfenergy`       | `elph_run = true`, `elph_selfen_fan = true`, `elph_selfen_dw = true`, `elph_selfen_delta = 0.01`, `elph_selfen_temps = [0, 300]`, `elph_ismear = -24` |
| `transport`        | `elph_fermi_nedos = 501`, `elph_ismear = -24`, `elph_mode = "transport"`, `elph_selfen_carrier_den = 0.0`, `elph_scattering_approx = ["serta", "mrta_lambda"]`, `elph_selfen_temps = [...]`, `elph_transport_nedos = 501` |
| `el_bands.dos`     | `ibrion = -1`, `nsw = 0`, `lorbit = 11`, `nedos = 5001`, `ismear = -5`                               |
| `el_bands.bands`   | `ibrion = -1`, `nsw = 0`                                                                             |
| `ph_bands`         | `ibrion = -1`, `nsw = 0`, `elph_run = true` (base smearing/I/O tags suppressed)                      |
<!-- Hidden until ph_selfenergy is public in VASP:
| `ph_selfenergy`    | `elph_fermi_nedos = 501`, `elph_run = true`, `elph_selfen_temps = [0, 300]`, `elph_ismear = -24`, `elph_driver = "ph"`, `elph_mode = "superconductivity"`, `elph_selfen_carrier_den = 0.0` |
-->

These calculation-type tags are merged on top of the base `[vasp.incar]` settings
shown above.

```{note}
These tags are starting points. Review the generated `INCAR` files and adjust
them for the target material, in particular `encut`, the smearing tags, and the
parallelization tags (`ncore`, `kpar`, ...).
```

(velph_toml_vasp_kpoints)=
## `[vasp.CALC_TYPE.kpoints]` : sampling mesh

A k-point block describes a regular sampling mesh. It accepts the following keys:

```toml
[vasp.phelel.kpoints]
mesh = [4, 4, 4]
shift = [0.5, 0.5, 0.5]
```

- `mesh` : either three integers for a Gamma-centered grid, or a 3x3 integer
  matrix for a generalized regular grid:

  ```toml
  [vasp.phelel.kpoints]
  mesh = [
    [-4, 4, 4],
    [4, -4, 4],
    [4, 4, -4],
  ]
  ```

- `shift` : optional shift of the mesh relative to neighboring grid points, in
  units of the grid spacing. `0.5` is a half-grid shift.

- `kspacing` : optional. Instead of an explicit `mesh`, a target spacing (in
  1/Angstrom) can be given and the mesh is computed from it at the time the input
  files are generated. This mirrors VASP's `KSPACING` with `KGAMMA = .TRUE.`:

  ```toml
  [vasp.relax.kpoints]
  kspacing = 0.2
  ```

The `kspacing` and `kspacing_dense` values used by `velph init` to populate the
`kpoints` and `kpoints_dense` blocks come from the `velph init` options of the
same name (see {ref}`velph_init`).

### `[vasp.CALC_TYPE.kpoints_dense]`

The electron-phonon calculation steps (`selfenergy`,
`transport`<!-- , `ph_selfenergy` -->) and `el_bands.dos` use a second, denser mesh for the
charge-density part of the sandwich calculation. It has the same syntax as
`kpoints`:

```toml
[vasp.transport.kpoints]
mesh = [12, 12, 12]
[vasp.transport.kpoints_dense]
mesh = [24, 24, 24]
```

(velph_toml_vasp_bands)=
## Band and q-point paths (`kpoints_opt`, `qpoints`)

Band-structure steps add a line-mode block on top of the regular `kpoints` mesh:

- `[vasp.el_bands.bands.kpoints_opt]` for the electronic band structure.
- `[vasp.ph_bands.qpoints]` for the phonon band structure.

Both use the same line-mode syntax:

```toml
[vasp.ph_bands.qpoints]
line = 51
path = [
  [ [0.000000, 0.000000, 0.000000], [0.500000, 0.000000, 0.500000] ],
  [ [0.500000, 0.000000, 0.500000], [0.500000, 0.250000, 0.750000] ],
]
label = [
  [ "GAMMA", "X" ],
  [ "X", "W" ],
]
```

- `line` : number of sampling points per path segment (default `51`).
- `path` : a list of segments, each segment being a pair of endpoints in
  fractional reciprocal coordinates.
- `label` : optional labels for the endpoints of each segment, used when
  plotting.

For `el_bands`, the regular `[vasp.el_bands.bands.kpoints]` mesh is still
required (VASP reads it as the self-consistent mesh), while the band path lives in
the `kpoints_opt` block. For `ph_bands`, `[vasp.ph_bands.kpoints]` is the
Gamma-only SCF mesh and the q-point path lives in `qpoints`.

(velph_toml_vasp_scheduler)=
## `[vasp.CALC_TYPE.scheduler]`

A per-step `scheduler` block overrides the top-level `[scheduler]` settings for
that calculation type only. Any key written here replaces the corresponding key
of `[scheduler]`; keys that are not written are inherited. See
{ref}`velph_toml_scheduler` for the full description of the scheduler mechanism
and the `scheduler_template` format.

```toml
[scheduler]
job_name = "PbTe"
pe = "vienna 24"
...
[vasp.phono3py.scheduler]
pe = "paris 24"
```

## Section ordering

`velph init` writes the `[vasp.*]` sub-sections in this order: the supercell
types (`phelel`, `phonopy`, `phono3py`), the electron-phonon types
(`selfenergy`, `transport`<!-- , `ph_selfenergy` -->), `relax`, `nac`,
`el_bands.bands`, `el_bands.dos`, and `ph_bands`. Only the sub-sections relevant
to the chosen project are written. The ordering has no semantic meaning; TOML
sections can appear in any order.
