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

See {ref}`velph_toml_scheduler`.

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
