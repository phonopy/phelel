(velph_subcommands)=
# `velph` subcommands

The `velph` subcommands, except for {ref}`velph_init`, are explained on this
page.

The following subcommands must be executed in the directory containing the
`velph.toml` file. This ensures that the directories and files generated are
properly organized relative to the current directory where `velph.toml` is
located.

## `velph generate`

This subcommand writes `[unitcell]` and `[primitive_cell]` of `velph.toml` to
`POSCAR-unitcell` and `POSCAR-primitive`. An existing file is not overwritten.
The prefix `POSCAR` can be changed with `--prefix`.

```bash
% velph generate
"POSCAR-unitcell" was written.
"POSCAR-primitive" was written.
```

## `velph convert`

This subcommand writes a `velph.toml` of velph 0.14 in the format of velph 0.15.
The first argument is the `velph.toml` of velph 0.14, which is not changed. The
second argument is the file to be written, which must not exist. The subcommand
prints the settings that were removed and the k-point blocks whose `kspacing`
was replaced by `mesh`.

```bash
% velph convert velph.toml velph-new.toml
fft_mesh of [phelel] was removed.
"velph-new.toml" has been written from "velph.toml".
```

The other subcommands also read a `velph.toml` of velph 0.14 as it is. See
{ref}`velph_changes_0_15_project_folder`.

(velph_subcommands_scheduler)=
## `velph scheduler`

This subcommand shows the job script `_job.sh` of each input set of
`velph.toml`, as the generate commands write it, and it writes no file. It is
used to check `[scheduler]` and `[vasp.<input set>.scheduler]` (see
{ref}`velph_toml_scheduler`). The input sets given as arguments are shown in
the order given. Without arguments, all the input sets in `[vasp]` are shown.
`job_name` is shown without the name of the calculation that the generate
commands append to it, such as `-iter1` of `velph relax generate`. When
`velph.toml` has no `[scheduler]`, the subcommand prints that no `_job.sh` is
written.

When `_job.sh` of an input set cannot be made, the error is shown in place of
`_job.sh`, the remaining input sets are shown, and the exit code is 1. In the
following example, `scheduler_template` of `[vasp.nac.scheduler]` has
`{qos}`, which is not given.

```bash
% velph scheduler relax nac
[vasp.relax]
#!/bin/bash
#SBATCH --job-name=NaCl
#SBATCH --partition=short
#SBATCH --ntasks=48
#SBATCH --time=12:00:00

srun vasp_std

[vasp.nac]
Error: "qos" used in the job script template is not given in [scheduler] or [vasp.<input set>.scheduler].  To write "{" or "}" in _job.sh, write "{{" or "}}" in scheduler_template.
```

## `velph relax`

It is recommended to create a project directory specific for relaxation
calculation, e.g., as

```bash
% ls
POSCAR-unitcell  POTCAR  velph-tmpl.toml
% velph init --template-toml velph-tmpl.toml POSCAR-unitcell relax
...
% cd relax
% ls
POTCAR  velph.toml
% velph relax generate
...
```
### `velph relax generate`

This subcommand generates VASP input files and a job script for crystal
structure relaxation. Initially, these files are created in the `iter1`
directory. The process supports multiple relaxation steps, where the `CONTCAR`
file from the previous iteration is used as the `POSCAR` for the next step once
the preceding VASP calculation is complete. Each subsequent set of input files
is generated in directories named `iter2`, `iter3`, and so on.

```bash
% ls
POTCAR  velph.toml
% velph relax generate
VASP input files were written in "relax/iter1".
...
# Run VASP calculation in relax/iter1
...
% ls
POTCAR  relax/  velph.toml
% velph relax generate
"relax/iter1" exists.
"relax/iter1/CONTCAR" was copied to "relax/iter2/POSCAR".
VASP input files were written in "relax/iter2".
...
```

(velph_nac_subcommand)=
## `velph nac`

This subcommand is used to prepare VASP input files for calculating the
dielectric constant and Born effective charges. These quantities are required
for the non-analytical term correction in phonon calculations and for handling
the long-range terms in electron-phonon interaction calculations.

### `velph nac generate`

The VASP input files are written in `nac`, on the cell given by `cell` of
`[vasp.nac]`, the primitive cell by default. After the VASP calculation is
complete, the dielectric constant and Born
effective charges are extracted from the `vasprun.xml` file during the execution
of `velph phelel init`, `velph phono3py init`, or `velph phonopy init`. The
extracted values are then written to `phelel_disp.yaml`, `phono3py_disp.yaml`,
or `phonopy_disp.yaml`, respectively.

```bash
% velph nac generate
VASP input files were written in "nac".
```

## `velph el_bands`

This subcommand is used to prepare VASP input files for electronic band
structure calculation, and to plot the band structure together with the density
of states calculated with {ref}`velph el_dos <velph_el_dos_subcommand>`.

### `velph el_bands generate`

The VASP input files are written in `el_bands`, so that the calculation results
will be stored in `vaspout.h5`. `KPOINTS_OPT` gives the band path of
`[vasp.el_bands.kpoints_opt]`. When this block does not give `path`, the band
path that seekpath gives is written (see {ref}`velph_toml_vasp_bands`).

```bash
% velph el_bands generate
VASP input files were written in "el_bands".
```

(velph_el_bands_plot)=
### `velph el_bands plot`

The electronic band structure and density of states are plotted from
`el_bands/vaspout.h5` and `el_dos/vaspout.h5`. The Fermi level is taken from the
DOS calculation, so run `velph el_dos generate` and VASP in `el_dos` before this
command. The energy range must be specified using the `--window` option, with
the first and second arguments representing the range below and above the Fermi
energy in eV, respectively. With `--save` option, the plot is saved in the
`el_bands/el_bands.pdf` file.

```bash
% velph el_bands plot --window -5 9 --save
Electronic band structure plot was saved in "el_bands/el_bands.pdf".
```

(velph_el_dos_subcommand)=
## `velph el_dos`

This subcommand is used to prepare VASP input files for electronic density of
states calculation, and to plot the density of states.

### `velph el_dos generate`

The VASP input files are written in `el_dos`. `KPOINTS_OPT` gives the dense mesh
of `[vasp.el_dos.kpoints_dense]`, on which VASP computes the density of states
and the Fermi level.

```bash
% velph el_dos generate
VASP input files were written in "el_dos".
```

### `velph el_dos plot`

The electronic density of states is plotted from `el_dos/vaspout.h5`, with the
energy on the horizontal axis. The Fermi level of the DOS calculation is shown
by a red line. The energy range must be specified using the `--window` option
in the same way as {ref}`velph el_bands plot <velph_el_bands_plot>`. Another
`vaspout.h5` can be given with `--vaspout-filename`. With `--save` option, the
plot is saved in `el_dos.pdf` in the directory of `vaspout.h5`.

```bash
% velph el_dos plot --window -5 9 --save
Electronic DOS plot was saved in "el_dos/el_dos.pdf".
```

### `velph el_dos plot_eigenvalues`

The eigenvalues in `el_dos/vaspout.h5` are shown in the Brillouin zone in the
same way as
{ref}`velph transport plot eigenvalues <velph_transport_plot_eigenvalues>`.

(velph_phelel_subcommand)=
## `velph phelel`

This subcommand calculates the derivatives of local potentials and PAW strengths
with respect to displacement, performing similar operations as steps 1 and 3
carried out by the `phelel` command in the {ref}`workflow <workflow_minimal>`.

The subcommands below should be executed in the following order:

1. {ref}`velph_phelel_init_subcommand`
2. {ref}`velph_phelel_generate_subcommand`
3. {ref}`velph_phelel_differentiate_subcommand`
4. {ref}`velph_phelel_phonopy_subcommand` (optional)

Certain configurations for phelel can be specified directly in the
`velph.toml` file:

```toml
[phelel]
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = false
plusminus = true
```

(velph_phelel_init_subcommand)=
### `velph phelel init`

This subcommand generates the `phelel_disp.yaml` file, which contains
information about the supercell structure and displacements. It is recommended
to generate `phelel_disp.yaml` after performing the NAC calculation (see
{ref}`velph_nac_subcommand`). At this stage, the dielectric constant and Born
effective charges are written to `phelel_disp.yaml`, and these values are
ultimately stored in `phelel_params.hdf5` through the `phelel_disp.yaml` file.

```bash
% velph phelel init
Displacement distance: 0.03
Number of displacements: 4
NAC parameters are read from "nac/vasprun.xml".
"phelel/phelel_disp.yaml" was written.
Run "velph phelel generate" to write the VASP input files.
```

The phonon part of phelel can be computed with the supercell of `[phonopy]`,
which can be larger than the supercell of `[phelel]`. Run `velph phonopy init`
first, and then `velph phelel init --phonon-from-phonopy`, which takes the
supercells with displacements from `phonopy/phonopy_disp.yaml`. The forces of
these supercells are then read from `phonopy/disp-*` by
`velph phelel differentiate`.

(velph_phelel_generate_subcommand)=
### `velph phelel generate`

VASP input files and job scripts are generated.

```bash
% velph phelel generate
VASP input files were written in "phelel/disp-000".
VASP input files were written in "phelel/disp-001".
VASP input files were written in "phelel/disp-002".
VASP input files were written in "phelel/disp-003".
VASP input files were written in "phelel/disp-004".
```

(velph_phelel_differentiate_subcommand)=
### `velph phelel differentiate`

The derivatives of local potentials and PAW strengths with respect to
displacement are calculated and stored in `phelel_params.hdf5`.

The FFT mesh of the local potentials is the FFT grid of VASP on the primitive
cell for `encut` and `prec` in the INCAR tags of the electron-phonon input sets
(`selfenergy` and `transport`). These input sets have to give the same mesh,
because VASP stops when its FFT grid differs from the mesh in
`phelel_params.hdf5`. `--encut` gives ENCUT in place of `encut` of these input
sets.

```bash
% velph phelel differentiate
FFT mesh: [30 30 30].
Found "phelel/disp-000/vaspout.h5".
Found "phelel/disp-001/vaspout.h5".
Found "phelel/disp-002/vaspout.h5".
Found "phelel/disp-003/vaspout.h5".
Found "phelel/disp-004/vaspout.h5".
"phelel/phelel_params.hdf5" was written.
```

(velph_phelel_phonopy_subcommand)=
### `velph phelel phonopy`

```{note}
This subcommand is similar to `velph ph_bands`, but the phonon band structure is
calculated using the phonopy code. While the phonon calculation routines of VASP
and phonopy are expected to yield similar results, they are not entirely
equivalent. However, for electron-phonon calculations, the VASP routine is
utilized.
```

`phonopy_params.yaml` that contains information necessary for phonon calculation
by the phonopy code is generated from `phelel_disp.yaml` and the results of the
VASP inputs generated by `velph phelel generate`.

```bash
% velph phelel phonopy
"phelel/phonopy_params.yaml" was written with the forces of the phelel supercells.
% phonopy-load phelel/phonopy_params.yaml --band auto -p
```

`phelel/phonopy_params.yaml` is always made from the supercells of `[phelel]`.
When `velph.toml` has `[vasp.phonopy]`, the command also says that the phonopy
calculation with the `[phonopy]` supercell is in `phonopy`.

## `velph selfenergy`

This subcommand is used to prepare VASP input files for calculating the Fan
self-energy based on electron-phonon interactions.

### `velph selfenergy generate`

The VASP input files are written in `selfenergy`, and
`phelel/phelel_params.hdf5` made by `velph phelel differentiate` is copied
there. `KPOINTS_ELPH` gives the dense mesh of `[vasp.selfenergy.kpoints_dense]`.

```bash
% velph selfenergy generate
VASP input files were written in "selfenergy".
```

With `--dry-run`, `nelm = 0` and `elph_run = false` are added to the INCAR, so
that VASP stops before the electron-phonon calculation. `phelel_params.hdf5` is
not needed and is not copied.

## `velph transport`

This subcommand is used to prepare VASP input files for calculating transport
properties based on electron-phonon interactions.

### `velph transport generate`

The VASP input files are prepared for transport property calculations in
`transport`, in the same way as `velph selfenergy generate`, including
`--dry-run`. Users should review the generated `INCAR` tags and modify them as
necessary for the target material.

```bash
% velph transport generate
VASP input files were written in "transport".
```

### `velph transport plot`

#### `velph transport plot transport`

The following command opens a graphical window that contains plots of transport
properties.  By default, it uses the `transport/vaspout.h5` file unless a
different file is specified as the first argument.

```bash
% velph transport plot transport
```

(velph_transport_plot_eigenvalues)=
#### `velph transport plot eigenvalues`

The following command opens a graphical window to display eigenvalues in the
Brillouin zone whose occupations are from `--cutoff-occupancy` to 1 minus it,
0.01 to 0.99 by default. By default, these eigenvalues are extracted from the
`transport/vaspout.h5` file unless a different file is specified as the first
argument. Their occupations are
determined based on the temperature and Fermi level set through the command-line
options.

```bash
% velph transport plot eigenvalues
mu: 4.837002 eV
temperature: 300.000000 K
No eigenvalues to plot.
% velph transport plot eigenvalues --mu 4
```

This subcommand requires pre-calculated eigenvalues, Fermi levels, and input
parameters such as temperatures. However, electron-phonon calculation results
are not necessary. If the goal is to visualize Fermi surface pockets on a
k-point grid, it is recommended to run the electron-phonon calculation using
`ELPH_SCATTERING_APPROX=CRTA` to generate the required `vaspout.h5` file
efficiently.

#### `velph transport plot selfenergy`

The following command opens a graphical window to display Fan self-energies. By
default, it uses the `transport/vaspout.h5` file unless a different file is
specified as the first argument.

```bash
% velph transport plot selfenergy
```

## `velph ph_bands`

This subcommand is used to prepare VASP input files for phonon band structure
calculation using the supercell force constants stored in `phelel_params.hdf5`,
and to plot the resulting data.

### `velph ph_bands generate`

The VASP input files including `QPOINTS` are written in `ph_bands`, so that the
calculation results will be stored in `vaspout.h5`. `phelel/phelel_params.hdf5`
made by `velph phelel differentiate` is copied there. When
`[vasp.ph_bands.qpoints]` does not give `path`, the band path that seekpath gives
is written (see {ref}`velph_toml_vasp_bands`).

```bash
% velph ph_bands generate
VASP input files were written in "ph_bands".
```

### `velph ph_bands plot`

The phonon band structure in `ph_bands/vaspout.h5` is plotted. The frequencies
are shown as angular frequencies by default, and as ordinary frequencies with
`--ordinary-frequency`. With `--save` option, the plot is saved in the
`ph_bands/ph_bands.pdf` file.

```bash
% velph ph_bands plot --save
Phonon band structure plot was saved in "ph_bands/ph_bands.pdf".
```

(velph_phono3py_subcommand)=
## `velph phono3py`

This subcommand facilitates anharmonic phonon calculation including lattice
thermal conductivity calculation using the phono3py code. Certain configurations
for supercell and displacements can be specified directly in the `velph.toml`
file:

```toml
[phono3py]
supercell_dimension = [2, 2, 2]
amplitude = 0.03
diagonal = false
plusminus = true
```

### `velph phono3py init`

The `phono3py_disp.yaml` file, which contains displacement information, is
generated by this subcommand. It is recommended to use the `--rd` option with a
sufficiently large argument to create supercells with random directional
displacements. Note that it is not necessary to calculate forces for all
supercells to ultimately generate the `phono3py_params.yaml` file.

```bash
% velph phono3py init --rd 2000
Displacement distance: 0.03
Number of displacements: 2000
NAC parameters are read from "nac/vasprun.xml".
"phono3py/phono3py_disp.yaml" was written.
Run "velph phono3py generate" to write the VASP input files.
```

The second-order (fc2) part can be computed with the supercell of `[phonopy]`,
which can be larger than the supercell of `[phono3py]`. Run
`velph phonopy init` first, and then `velph phono3py init --phonon-from-phonopy`,
which takes the supercells with displacements from `phonopy/phonopy_disp.yaml`.

### `velph phono3py generate`

VASP input files for supercell force calculations are generated, with each
directory corresponding to a specific supercell displacement configuration. The
perfect supercell (i.e., without displacements) is labeled as `disp-0000`. The
number of digits grows with the number of supercells, and here 2000 supercells
need four digits.

For example:

```bash
% velph phono3py generate
VASP input files were written in "phono3py/disp-0000".
VASP input files were written in "phono3py/disp-0001".
...
```

After completing supercell force calculations using the VASP code, the
`phono3py_params.yaml` file can be created with the following command:

```bash
% cd phono3py
% phono3py --cfz disp-0000/vasprun.xml --cf3 disp-{0001..0120}/vasprun.xml --sp
```

In this step, forces from 120 supercells are collected, and the corresponding
data are stored in the `phono3py_params.yaml` file.

(velph_phonopy_subcommand)=
## `velph phonopy`

This subcommand facilitates phonon calculation using the phonopy code. Certain
configurations for supercell and displacements can be specified directly in the
`velph.toml` file:

```toml
[phonopy]
supercell_dimension = [2, 2, 2]
amplitude = 0.01
diagonal = false
plusminus = true
```

### `velph phonopy init`

```bash
% velph phonopy init
Displacement distance: 0.01
Number of displacements: 4
NAC parameters are read from "nac/vasprun.xml".
"phonopy/phonopy_disp.yaml" was written.
Run "velph phonopy generate" to write the VASP input files.
```

### `velph phonopy generate`

```bash
% velph phonopy generate
VASP input files were written in "phonopy/disp-000".
VASP input files were written in "phonopy/disp-001".
VASP input files were written in "phonopy/disp-002".
VASP input files were written in "phonopy/disp-003".
VASP input files were written in "phonopy/disp-004".
```

After completing supercell force calculations using the VASP code, the
`phonopy_params.yaml` file can be created with the following command:

```bash
% cd phonopy
% phonopy --fz disp-{000..004}/vasprun.xml --sp
```
