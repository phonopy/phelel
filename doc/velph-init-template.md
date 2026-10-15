(velph_init_template)=
# `velph init` template

The `velph init` command is used to prepare the `velph.toml` file. If no custom
template is specified, the default settings of `velph init` are used. To specify
a custom template, use the following command:

```bash
% velph init CELL_FILENAME PROJECT_FOLDER --template-toml velph-tmpl.toml [OPTIONS]
```

The `velph-tmpl.toml` file (which can have any file name) is written in the same
format as `velph.toml`. A template can have the sections `[init]`, `[phelel]`,
`[phonopy]`, `[phono3py]`, `[vasp]` and `[scheduler]`, and every key in them is
optional. `velph init` uses the settings of the template instead of its default
settings, key by key, so a template needs only the settings that differ from the
default settings.

A template can also have settings that `velph.toml` cannot have, such as the
`[init]` section, which gives the defaults of the command-line options of
`velph init`. See {ref}`velph_toml_template_only` for the list.

## Making a template from an existing `velph.toml`

`velph init` writes `[velph]`, `[symmetry]`, `[unitcell]` and `[primitive_cell]`
in `velph.toml`, and a template cannot have them. To use an existing
`velph.toml` as a template, remove these four sections. With the other sections
left as they are, `velph init` writes the same settings for the same crystal
structure.

The k-point blocks of such a template give `mesh`, and its `[phelel]`,
`[phonopy]` and `[phono3py]` give the supercell. `velph init` uses them as they
are, also for another crystal structure. To have `velph init` compute the meshes
for the new structure, remove the k-point blocks or write `kspacing` in them
instead of `mesh`. To change the supercell, give it on the command line, for
example with `--dim`.

(velph_init_template_init_options)=
## `[init]`

```toml
[init]
codes = ["phelel", "phonopy"]
kspacing = 0.2
kspacing_dense = 0.1
max_num_atoms = 120
```

The keys of `[init]` correspond to command-line options of `velph init`. A value
given on the command line overrides the one in `[init]`. To view the available
options and their defaults, use:

```bash
% velph init --help
```

The help of each option shows its key and type in parentheses, for example
`(kspacing: float, default=0.1)`. The keys are listed below.

| Key                     | Type                            | Default                            | `velph init` option        |
| ----------------------- | ------------------------------- | ---------------------------------- | -------------------------- |
| `codes`                 | list[str]                       | `["phelel", "phonopy", "phono3py"]` | `--code`                  |
| `amplitude`             | float                           | (see below)                        | `--amplitude`              |
| `cell_for_nac`          | str                             | `"primitive"`                      | `--cell-for-nac`           |
| `cell_for_relax`        | str                             | `"unitcell"`                       | `--cell-for-relax`         |
| `diagonal`              | bool                            | `false`                            | `--diagonal`               |
| `find_primitive`        | bool                            | `true`                             | `--no-find-primitive`      |
| `kspacing`              | float                           | `0.1`                              | `--kspacing`               |
| `kspacing_dense`        | float                           | `0.05`                             | `--kspacing-dense`         |
| `magmom`                | str                             | (none)                             | `--magmom`                 |
| `max_num_atoms`         | int                             | (none)                             | `--max-num-atoms`          |
| `nosym`                 | bool                            | `false`                            | `--nosym`                  |
| `plusminus`             | bool or `"auto"`                | `true`                             | `--plusminus` / `--auto`   |
| `primitive_cell_choice` | str                             | `"standardized"`                   | `--primitive-cell-choice`  |
| `supercell_dimension`   | list[int] (3)                   | (none)                             | `--dim`                    |
| `supercell_matrix`      | list[int] (9) or list[list[int]] (3x3) | (none)                      | `--supercell-matrix`       |
| `symmetrize_cell`       | bool                            | `false`                            | `--symmetrize-cell`        |
| `tolerance`             | float                           | `1e-5`                             | `--tolerance`              |
| `use_grg`               | bool                            | `false`                            | `--use-grg`                |
<!-- Hidden until site mixture is public:
| `site_mixture`          | str                             | (none)                             | `--site-mixture`           |
| `split_site_mixture`    | bool                            | `false`                            | `--split-site-mixture`     |
-->

Notes:

- `codes` chooses the codes whose sections are written in `velph.toml`. A section
  of the template for a code that is not in `codes` is not used, and
  `velph init` prints, for example,
  `[phonopy] of the template is ignored: "phonopy" is not in codes.`
- Without `amplitude`, `velph init` writes 0.03 in `[phelel]` and `[phono3py]`
  and 0.01 in `[phonopy]`. These are the default displacement distances of
  phono3py and phonopy for VASP.
- `amplitude`, `diagonal`, `plusminus`, `nosym` and the supercell keys in
  `[init]` are used for all the codes in `codes`.
- `cell_for_nac` and `cell_for_relax` accept `"primitive"` or `"unitcell"`.
- `primitive_cell_choice` accepts `"standardized"` or `"reduced"`.
- These values are compared in lower case, so `"Primitive"` is accepted. Any
  other value, such as `"primitive_cell"`, is an error.
- `max_num_atoms` determines the supercell dimension and must be used together
  with `symmetrize_cell`.
- `max_num_atoms`, `supercell_dimension` (three integers), and `supercell_matrix`
  (nine integers or a 3x3 matrix) are three ways to give the supercell. Give only
  one of them in `[init]`. When two or more of them are given, `velph init` stops
  with an error.
- When the command line gives the supercell, for example with `--dim`, the
  supercell keys in `[init]` are not used. See
  {ref}`velph_init_template_precedence`.
<!-- Hidden until site mixture is public:
- `site_mixture` and `split_site_mixture` are experimental, and `site_mixture`
  cannot be combined with `magmom`. As in phonopy, co-located atoms are merged
  into mixed-species sites by default, and `split_site_mixture` keeps them as
  separate weighted species. phono3py supports neither, so
  `velph phono3py init` stops with an error for a site-mixture cell.
-->
- The file-handling options of `velph init` (`--force`, `--template-toml`,
  `--toml-filename`) are command-line only and have no `[init]` key.

(velph_init_template_precedence)=
## Order of precedence

The same setting can be given in more than one place. For example, the
displacement amplitude can be given by `--amplitude` on the command line, by
`amplitude` in `[init]`, and by `amplitude` in `[phelel]` of the template.
`velph init` takes the value from the first of the following places that gives
it:

1. a command-line option of `velph init`
2. `[init]` of the template
3. the other sections of the template, such as `[phelel]`
4. the default value

This order applies to the settings in the table below.

| Setting   | Command line                                  | `[init]`                                              | Template section                                                     |
| --------- | --------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| amplitude | `--amplitude`                                 | `amplitude`                                           | `amplitude` in `[phelel]`, `[phonopy]`, `[phono3py]`                 |
| diagonal  | `--diagonal`                                  | `diagonal`                                            | `diagonal` in `[phelel]`, `[phonopy]`, `[phono3py]`                  |
| plusminus | `--plusminus` / `--auto`                      | `plusminus`                                           | `plusminus` in `[phelel]`, `[phonopy]`, `[phono3py]`                 |
| nosym     | `--nosym`                                     | `nosym`                                               | `nosym` in `[phelel]`, `[phonopy]`, `[phono3py]`                     |
| supercell | `--max-num-atoms`, `--dim`, `--supercell-matrix` | `max_num_atoms`, `supercell_dimension`, `supercell_matrix` | `max_num_atoms`, `supercell_dimension`, `supercell_matrix` in `[phelel]`, `[phonopy]`, `[phono3py]` |
| cell      | `--cell-for-relax`, `--cell-for-nac`          | `cell_for_relax`, `cell_for_nac`                      | `cell` in `[vasp.relax]`, `[vasp.nac]`                               |

A value from the command line or from `[init]` is used for all of `[phelel]`,
`[phonopy]`, and `[phono3py]`. A value in a template section is used only for
that section. For example, `amplitude = 0.02` in `[phonopy]` of the template
sets the amplitude of `[phonopy]`, and `[phelel]` gets the default value 0.03.
With `amplitude = 0.05` in `[init]` of the same template, all three sections get
0.05.

A template made from an existing `velph.toml` has `amplitude`, `diagonal`,
`plusminus`, `nosym`, and the supercell in `[phelel]`. These values are used when
the command line and `[init]` do not give them. To change one of them for a new
project, give it on the command line, for example `--amplitude 0.02`, and the
template can be used as it is.

The supercell is one setting that can be given in three ways: by the number of
atoms (`max_num_atoms`), by three integers (`supercell_dimension`), or by nine
integers (`supercell_matrix`). In each place, give only one of them. When two or
more of them are given in the same place, `velph init` stops with an error. When
the command line gives one of them, none of the three in `[init]` is used. For
example, with `max_num_atoms = 120` in `[init]` and `--dim 3 3 3` on the command
line, the supercell is 3x3x3.

The k-point meshes follow another order. A k-point block of the template that
gives `mesh` or `kspacing` takes priority over `--kspacing` and
`--kspacing-dense`, and over `kspacing` and `kspacing_dense` in `[init]`. These
options and keys are used for the k-point blocks that the template does not
give. See {ref}`velph_init` for an example.

INCAR tags follow different rules, which are described in the next section.

(velph_init_template_incar)=
## `[vasp.incar]` and `incar_unset`

In a template, the `[vasp.incar]` section holds the INCAR tags shared by the
input sets, such as the plane-wave cutoff and the parallelization tags:

```toml
[vasp.incar]
encut = 400
ncore = 4
gga = "PS"
```

`velph init` adds these tags to the INCAR tags of each input set that does not
give them, and writes the result in `[vasp.<input set>.incar]` of `velph.toml`.
A tag given in `[vasp.<input set>.incar]`, by the template or by the default
settings, takes priority over the same tag of `[vasp.incar]`.

A tag listed in `incar_unset` of an input set is not added to that input set:

```toml
[vasp.nac]
incar_unset = ["npar", "ncore", "kpar"]
```

The default settings of `velph init` list tags in `incar_unset` of `nac` and
`ph_bands` (see {ref}`velph_toml_vasp_incar`). `incar_unset` of an input set in
the template replaces that of the default settings. For example,
`incar_unset = ["ismear", "sigma"]` in `[vasp.ph_bands]` of the template makes
`velph init` add `ediff`, `lreal`, `lwave` and `lcharg` of `[vasp.incar]` to
`ph_bands`, which the default `incar_unset` of `ph_bands` lists.

`[vasp.incar]` and `incar_unset` are used only by `velph init`, so `velph.toml`
cannot have them. See {ref}`velph_toml_vasp_incar` for the INCAR tags of each
input set in `velph.toml`.

## Templates of velph 0.14

In a template of velph 0.14, the defaults of the command-line options were given
in `[init.options]`, and an INCAR tag set to an empty table `{}` was not added to
the input set. `velph init` stops with an error for such a template, and the
error lists the changes that concern the file. To use the template:

- Rename `[init.options]` to `[init]`, and `phelel_nosym` in it to `nosym`.
  `nosym`, `amplitude`, `diagonal`, `plusminus` and the supercell keys in
  `[init]` are used for all the codes in `codes`.
- Remove the tags set to `{}`, and list them in `incar_unset` of the input set
  instead.
