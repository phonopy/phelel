(velph_toml_vasp)=
# `velph.toml`-`[vasp]` section

The `[vasp]` section is the largest part of `velph.toml`. It stores the VASP
`INCAR` tags, k-point sampling, and job-script settings used to generate the
input files of each calculation of the workflow.

The settings of one kind of calculation are called an input set, and each input
set is a sub-section `[vasp.<input set>]`. The generate command of an input set,
for example `velph relax generate` for `[vasp.relax]`, writes the VASP input
files in the directory of the same name. The input sets are:

| Input set          | Purpose                                                | Subcommand                                    | Written by `velph init`       |
| ------------------ | ------------------------------------------------------ | --------------------------------------------- | ----------------------------- |
| `relax`            | Crystal structure relaxation                           | {ref}`velph relax <velph_subcommands>`        | always                        |
| `nac`              | Dielectric constant and Born effective charges         | {ref}`velph nac <velph_nac_subcommand>`       | always                        |
| `phelel`           | Supercells for derivatives of potentials/PAW strengths | {ref}`velph phelel <velph_phelel_subcommand>` | with `[phelel]`               |
| `phonopy`          | Supercells for phonon (phonopy)                        | {ref}`velph phonopy <velph_phonopy_subcommand>` | with `[phonopy]`            |
| `phono3py`         | Supercells for anharmonic phonon (phono3py)            | {ref}`velph phono3py <velph_phono3py_subcommand>` | with `[phono3py]`         |
| `selfenergy`       | Fan self-energy (electron-phonon)                      | `velph selfenergy`                            | with `[phelel]`               |
| `transport`        | Transport properties (electron-phonon)                 | `velph transport`                             | with `[phelel]`               |
| `el_bands`         | Electronic band structure                              | `velph el_bands`                              | always                        |
| `el_dos`           | Electronic density of states                           | `velph el_dos`                                | always                        |
| `ph_bands`         | Phonon band structure                                  | `velph ph_bands`                              | with `[phelel]`               |
<!-- Hidden until ph_selfenergy is public in VASP:
| `ph_selfenergy`    | Phonon self-energy (superconductivity)                 | `velph ph_selfenergy`                         | with `[phelel]`               |
-->

The VASP input files of `relax` are written in `relax/iter1`, `relax/iter2`, ....
Those of the supercells of `phelel` are written in `phelel/disp-000`,
`phelel/disp-001`, ..., and those of `phonopy` and `phono3py` in the same way.

## Common sub-block structure

Every input set is built from the following kinds of sub-blocks:

```toml
[vasp.<input set>.incar]          # INCAR tags
[vasp.<input set>.kpoints]        # k-point sampling mesh
[vasp.<input set>.kpoints_dense]  # dense mesh (only some input sets)
[vasp.<input set>.kpoints_opt]    # band path (el_bands only)
[vasp.<input set>.qpoints]        # q-point path (ph_bands only)
[vasp.<input set>.scheduler]      # job-script settings of this input set
```

Each k-point block is written to one k-point file of VASP:

| Input set                                     | `KPOINTS`  | Second k-point file              |
| --------------------------------------------- | ---------- | -------------------------------- |
| `relax`, `nac`, `phelel`, `phonopy`, `phono3py` | `kpoints` | —                                |
| `selfenergy`, `transport`<!-- , `ph_selfenergy` --> | `kpoints` | `KPOINTS_ELPH` from `kpoints_dense` |
| `el_bands`                                    | `kpoints`  | `KPOINTS_OPT` from `kpoints_opt` |
| `el_dos`                                      | `kpoints`  | `KPOINTS_OPT` from `kpoints_dense` |
| `ph_bands`                                    | `kpoints`  | `QPOINTS` from `qpoints`         |

In `velph.toml`, the k-point blocks of this table are required. `incar` and
`scheduler` are optional:

- `incar` is written by `velph init` for all input sets.
- `scheduler` is optional. When present, it overrides the top-level
  {ref}`[scheduler] <velph_toml_scheduler>` settings for this input set only.

The `relax` and `nac` sections also have a `cell` key, which is required in
`velph.toml`:

```toml
[vasp.relax]
cell = "unitcell"
[vasp.nac]
cell = "primitive"
```

`cell` selects whether the unit cell or the primitive cell is used for that input
set. The defaults of `velph init` are `unitcell` for `relax` and `primitive` for
`nac`, and they can be set with the `velph init` options `--cell-for-relax` and
`--cell-for-nac` (see {ref}`velph_init`). The value is `"unitcell"` or
`"primitive"`, compared in lower case. Any other value, such as
`"primitive_cell"`, is an error in `velph init` and in the generate commands.

The other input sets use a fixed cell: the supercells with displacements for
`phelel`, `phonopy` and `phono3py`, and `[primitive_cell]` for the others.

(velph_toml_vasp_incar)=
## INCAR settings

Each input set holds all of its INCAR tags in `[vasp.<input set>.incar]`. The
generate command of the input set writes them to the `INCAR` file as they are.

```toml
[vasp.phelel.incar]
ismear = 0
sigma = 0.01
ediff = 1e-08
encut = 500
prec = "accurate"
lreal = false
lwave = false
lcharg = false
elph_prepare = true
isym = 0
```

INCAR tags are written as plain key-value pairs. Tag names are case-insensitive
and are normalized to lower case. A tag given twice in different cases, such as
`encut` and `ENCUT`, is an error.

velph always writes the k-point files from the k-point blocks described below.
For this reason, the VASP tags `kspacing` and `elph_kspacing`, which make VASP
generate the k-points without these files, cannot be written in an INCAR
section. If a template has one of them, `velph init` stops with an error, and if
a `velph.toml` has one of them, the generate commands stop with an error. To give
the k-point mesh by a spacing, write `kspacing` in a k-point block of the
template instead (see {ref}`velph_toml_vasp_kpoints`).

### INCAR tags shared by the input sets in a template

A template of `velph init` can give the INCAR tags shared by the input sets in
`[vasp.incar]`. `velph init` adds them to the INCAR tags of each input set that
does not give them. A tag listed in `incar_unset` of an input set is not added
to that input set. For example, with the template

```toml
[vasp.incar]
encut = 520
ncore = 4

[vasp.relax]
incar_unset = ["ncore"]
```

`velph init` writes `encut = 520` in the INCAR tags of all input sets, and
`ncore = 4` in those of all input sets except `relax` and `nac`. `nac` does not
have `ncore`, because `incar_unset` of `nac` in the default settings lists it
(see the table below). The tags of the template are added to the default tags, and
`incar_unset` of an input set in the template replaces that of the default
settings.

`[vasp.incar]` and `incar_unset` are used only by `velph init`, so `velph.toml`
cannot have them. To change a tag after `velph init`, edit it in each
`[vasp.<input set>.incar]` that uses it. See {ref}`velph_init_template_incar` for
how these settings are used in a template.

```{mermaid}
flowchart LR
  subgraph INIT["velph init"]
    direction TB
    DEF["defaults:<br/>vasp.incar<br/>+ vasp.INPUT_SET.incar"]
    TMPL["template toml:<br/>vasp.incar /<br/>vasp.INPUT_SET.incar"]
    MERGE{{"add vasp.incar to<br/>each input set,<br/>except incar_unset"}}
    DEF --> MERGE
    TMPL --> MERGE
  end
  MERGE --> MERGED["velph.toml:<br/>vasp.INPUT_SET.incar<br/>(all tags)"]
  MERGED --> GEN["velph INPUT_SET<br/>generate"]
  GEN --> INCAR["INCAR file"]
```

### Default INCAR tags by input set

The default settings of `velph init` give the following tags in `[vasp.incar]`:

```toml
[vasp.incar]
ismear = 0
sigma = 0.01
ediff = 1e-08
encut = 500
prec = "accurate"
lreal = false
lwave = false
lcharg = false
```

and the following tags in each input set:

| Input set          | Default INCAR tags                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| `relax`            | `ediffg = -1e-6`, `ibrion = 2`, `isif = 3`, `nsw = 10`                                               |
| `nac`              | `lepsilon = true` (and `incar_unset = ["npar", "ncore"]`)                                           |
| `phelel`           | `elph_prepare = true`, `isym = 0`                                                                    |
| `phonopy`          | `addgrid = true`, `isym = 0`                                                                          |
| `phono3py`         | `addgrid = true`, `isym = 0`                                                                          |
| `selfenergy`       | `elph_run = true`, `elph_selfen_fan = true`, `elph_selfen_dw = true`, `elph_selfen_delta = 0.01`, `elph_selfen_temps = [0, 300]`, `elph_ismear = -24` |
| `transport`        | `elph_fermi_nedos = 501`, `elph_ismear = -24`, `elph_mode = "transport"`, `elph_selfen_carrier_den = 0.0`, `elph_scattering_approx = ["serta", "mrta_lambda"]`, `elph_selfen_temps = [...]`, `elph_transport_nedos = 501` |
| `el_bands`         | `ibrion = -1`, `nsw = 0`                                                                             |
| `el_dos`           | `ibrion = -1`, `nsw = 0`, `lorbit = 11`, `nedos = 5001`, `ismear = -5`                               |
| `ph_bands`         | `ibrion = -1`, `nsw = 0`, `elph_run = true` (and `incar_unset = ["ismear", "sigma", "ediff", "lreal", "lwave", "lcharg"]`) |
<!-- Hidden until ph_selfenergy is public in VASP:
| `ph_selfenergy`    | `elph_fermi_nedos = 501`, `elph_run = true`, `elph_selfen_temps = [0, 300]`, `elph_ismear = -24`, `elph_driver = "ph"`, `elph_mode = "superconductivity"`, `elph_selfen_carrier_den = 0.0` |
-->

A tag of an input set has priority over the same tag of `[vasp.incar]`, as
`ismear = -5` of `el_dos` does.

When the tags of the `phelel` input set have `ispin = 2`,
`lnoncollinear = true` or `lsorbit = true`, and no `lwap`, `velph init` adds
`lwap = true` to them with a message. VASP 6.6.1 and older do not write the
local potential correctly in `vaspout.h5` for these calculations, and phelel
reads `LOCAL-POTENTIAL.bin` written with `LWAP = .TRUE.` instead.

```{note}
These tags are starting points. Review the generated `INCAR` files and adjust
them for the target material, in particular `encut`, the smearing tags, and the
parallelization tags (`ncore`, `kpar`, ...).
```

(velph_toml_vasp_kpoints)=
## `[vasp.<input set>.kpoints]` : sampling mesh

A k-point block of `velph.toml` gives a regular sampling mesh with the following
keys:

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

In a template of `velph init`, a k-point block can give `kspacing`, a spacing in
1/Angstrom, instead of `mesh`:

```toml
[vasp.relax.kpoints]
kspacing = 0.2
```

`velph init` computes the mesh from it on the cell of the input set and writes
the mesh as `mesh` in `velph.toml`. The number of divisions along each
reciprocal basis vector b_i is max(1, ceiling(|b_i| 2π / `kspacing`)), the same
rule as the VASP INCAR tag `KSPACING`. Give either `mesh` or `kspacing` in one
block. A block that gives neither takes the `kspacing` or `kspacing_dense` option
of `velph init` (see {ref}`velph_init`).

### `[vasp.<input set>.kpoints_dense]`

The electron-phonon input sets (`selfenergy`,
`transport`<!-- , `ph_selfenergy` -->) use a second, denser mesh for the
charge-density part of the sandwich calculation, and `el_dos` uses it as the mesh
on which VASP computes the density of states. It has the same syntax as
`kpoints`:

```toml
[vasp.transport.kpoints]
mesh = [12, 12, 12]
[vasp.transport.kpoints_dense]
mesh = [24, 24, 24]
```

(velph_toml_vasp_bands)=
## Band and q-point paths (`kpoints_opt`, `qpoints`)

Band-structure input sets add a line-mode block on top of the regular `kpoints`
mesh:

- `[vasp.el_bands.kpoints_opt]` for the electronic band structure.
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

- `line` : number of sampling points per path segment. It is required, and
  `velph init` writes `line = 51`.
- `path` : a list of segments, each segment being a pair of endpoints in
  fractional coordinates of the reciprocal basis of `[primitive_cell]`.
- `label` : optional labels for the endpoints of each segment, used when
  plotting.

`velph init` writes `line` only. When `path` is not given, the generate command
writes the band path that seekpath gives, so seekpath has to be installed.

For `el_bands`, the regular `[vasp.el_bands.kpoints]` mesh is still required
(VASP reads it as the self-consistent mesh), and the band path is given in the
`kpoints_opt` block. For `ph_bands`, `[vasp.ph_bands.kpoints]` is the
Gamma-only SCF mesh, and the q-point path is given in `qpoints`.

(velph_toml_vasp_scheduler)=
## `[vasp.<input set>.scheduler]`

A `scheduler` block of an input set overrides the top-level `[scheduler]`
settings for that input set only. Any key written here replaces the
corresponding key of `[scheduler]`; keys that are not written are inherited.
Without the top-level `[scheduler]`, `_job.sh` is not written. See
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

`velph init` writes the input sets in the order of the table at the top of this
page<!-- , with `ph_selfenergy` after `transport` -->. Only the input sets
relevant to the chosen codes are written. The ordering has no semantic meaning;
TOML sections can appear in any order.
