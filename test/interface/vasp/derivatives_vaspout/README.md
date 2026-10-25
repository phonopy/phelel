# Derivatives from vaspout.h5

Each directory holds the VASP results of the supercells of one system and the
derivatives that phelel computed from them:

- `phelel_disp.yaml`: the supercell and the displacements, as written by
  `velph phelel init`,
- `disp-000/vaspout.h5`, ...: the perfect supercell and the supercells with
  displacements, reduced as described below,
- `reference.hdf5`: dVdu and dDijdu computed by phelel from these files, with
  `fft_mesh` as an attribute.

| directory | crystal | supercell | displacements | POTCAR |
|---|---|---|---|---|
| `C` | diamond, conventional cell, 8 atoms | 1x1x1 | atom 1 along x, 0.01 A | PAW_PBE C 08Apr2002 |
| `Mg` | hcp Mg, 2 atoms | 1x1x1 | atom 1 along x and along z, 0.03 A | PAW_PBE Mg 13Apr2007 |

The calculations were done with VASP 6.6.0, ENCUT = 300, PREC = Accurate,
EDIFF = 1e-8, ISYM = 0, LREAL = .FALSE. and the Davidson algorithm, without
spin polarization (ncdij = 1).  C uses ISMEAR = 0, SIGMA = 0.01 and a
Gamma-centered 6x6x6 k-point mesh.  Mg uses ISMEAR = 1, SIGMA = 0.2 and a
Gamma-centered 12x12x7 k-point mesh.  All calculations reached EDIFF.

## Reduced vaspout.h5

To keep the files small, `vaspout.h5` has only the groups `input/poscar`,
`intermediate/ion_dynamics`, `results/paw` and `version`, the dataset
`results/potential/total`, and the datasets `ispin`, `nb_tot`, `kpoints`,
`nions` and `kpoint_coords` of `results/electron_eigenvalues`.  The datasets
with more than 1000 elements are compressed with gzip in HDF5, so h5py reads
them without decompressing the file.  dVdu, dDijdu, dqijdu and the force
constants computed from the reduced files are equal to those computed from the
full files, element by element.

## Reference values

`reference.hdf5` was made at commit fde8157 with

```python
phe = phelel.load("phelel_disp.yaml", fft_mesh=fft_mesh, is_nac=False)
create_derivatives(phe, ["disp-000", "disp-001", ...])
```

and `phe.dVdu.dVdu` and `phe.dDijdu.dDijdu` stored as complex arrays, with
`fft_mesh = [14, 14, 14]` for C and `[12, 12, 20]` for Mg.  The values pin the
results of phelel, and the test does not check them against an independent
calculation.  When a change of phelel is meant to change these results, the
reference values are made again in the same way.

dqijdu is not stored.  qij of all supercells of these data are equal, so
dqijdu is zero, and the test checks that it is.
