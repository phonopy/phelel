# Phelel

A code that provides a few computations related to electron-phonon interaction
calculation in finite-displacement method reported by

- Laurent Chaput, Atsushi Togo, and Isao Tanaka, Phys. Rev. B **100**, 174304
(2019).
- Manuel Engel, Henrique Miranda, Laurent Chaput, Atsushi Togo, Carla Verdi,
  Martijn Marsman, and Georg Kresse, Phys. Rev. B **106**, 094316 (2022)

This code couples with VASP code, and the electron-phonon interaction properties
cannot be computed only using this code.

Properties calculated using phelel and VASP codes:

- Thermoelectric properties
- Bandgap renormalization

(phelel_workflow)=
## Workflow of electron-phonon interaction calculation

1. (Phelel) Generate supercell structures with atomic displacements from a given unit cell
2. (VASP) Run VASP calculations for the supercell structures to obtain local
  potentials and PAW strengths under atomic displacements
3. (Phelel) Collect VASP results and computer derivatives of local potentials and PAW
  strengths with respect to atomic displacements
4. (VASP) Run VASP with derivatives of local potentials and PAW strengths to
   calculate electron-phonon interaction strengths and related properties, e.g.,
   - Electrical conductivity
   - Seebeck coefficient
   - Electrical thermal conductivity

## Command-line tools

### `phelel` and `phelel-load`

These command-line tools are used to perform the steps 1 and 3 in
{ref}`phelel_workflow`. Usage of `phelel` and `phelel-load` commands are similar
to `phonopy` and `phonopy-load` commands in
[phonopy](https://phonopy.github.io/phonopy/) code for phonon calculation.

### `velph`

(see {ref}`velph_command`)

The `velph` command organizes the {ref}`phelel_workflow` by running its
sub-commands in a specific sequence. These `velph` sub-commands can perform the
following operations:

- Execute `phelel`.
- Generate VASP input files for the following calculations (required for the
  subsequent electron-phonon interaction calculation):
  - Dielectric constant
  - Born effective charges
  - Electronic band structure and density of states
- Generate VASP input files for phonon and lattice thermal conductivity
  calculations using phonopy and phono3py.
## License

BSD-3-Clause.

## Contributors

- {user}`Atsushi Togo <atztogo>` (National Institute for Materials Science)

```{toctree}
:hidden:
install
velph
changelog
```
