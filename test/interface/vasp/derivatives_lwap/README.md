# Derivatives from the files written with LWAP = .TRUE.

Each directory holds the VASP results of the supercells of one magnetic system
and the derivatives that phelel computed from them:

- `phelel_disp.yaml`: the supercell, the magnetic moments and the
  displacements, as written by `velph phelel init`,
- `disp-000/`, `disp-001/`: the perfect supercell and the supercell with the
  displacement, with
  - `inwap.yaml`, `PAW-STRENGTH.bin`, `PAW-OVERLAP.bin`: as written by VASP,
  - `LOCAL-POTENTIAL.bin.xz`: `LOCAL-POTENTIAL.bin` written by VASP,
    compressed with xz,
  - `vaspout.h5`: the groups `input/poscar`, `intermediate/ion_dynamics` and
    `version` of `vaspout.h5`, from which the forces are read,
- `reference.hdf5`: dVdu and dDijdu computed by phelel from these files, with
  `fft_mesh` as an attribute.

| directory | crystal | ncdij | MAGMOM | POTCAR |
|---|---|---|---|---|
| `Cr` | bcc Cr, conventional cell, 2 atoms | 2 | 2.0 -2.0 | PAW_PBE Cr 06Sep2000 |
| `Fe-ncl` | bcc Fe, conventional cell, 2 atoms | 4 | 6*3.0 | PAW Fe 03Mar1998 (LDA) |

In both, the supercell is 1x1x1, atom 1 is displaced along x by 0.03 A, and
some of the symmetry operations used in the symmetrization are combined with
time reversal.

The calculations were done with VASP 6.6.0, LWAP = .TRUE., ENCUT = 300,
PREC = Accurate, EDIFF = 1e-8, ISYM = 0, LREAL = .FALSE., ISMEAR = 1 and
SIGMA = 0.2.  All calculations reached EDIFF.
- Cr uses ISPIN = 2, the Davidson algorithm and a Gamma-centered 11x11x11
  k-point mesh.  It is antiferromagnetic.
- Fe-ncl uses LNONCOLLINEAR = .TRUE. without LSORBIT, LDA, IALGO = 48 and a
  Gamma-centered 12x12x12 k-point mesh.  The moments are along (1, 1, 1).
  `disp-000` is the calculation of `test/base/Dij_perfect/Fe-ncl.npz` and
  `Dij_qij_two_formats/Fe-ncl`.

VASP 6.6.1 and older do not write the local potential correctly in
`vaspout.h5` for ISPIN = 2 or LNONCOLLINEAR = .TRUE., so the local potential,
Dij and qij are read from the files of the old format.  When a VASP that writes
it correctly is released and the readers of the old format are removed, these
data are replaced by the results of that VASP in `vaspout.h5`.

dVdu, dDijdu, dqijdu and the force constants computed from these files are
equal, element by element, to those computed from the full output of the same
calculations.

## Reference values

`reference.hdf5` was made at commit 961ea4d with

```python
phe = phelel.load("phelel_disp.yaml", fft_mesh=[16, 16, 16], is_nac=False)
create_derivatives(phe, ["disp-000", "disp-001"])
```

and `phe.dVdu.dVdu` and `phe.dDijdu.dDijdu` stored as complex arrays.  The
values pin the results of phelel, and the test does not check them against an
independent calculation.  When a change of phelel is meant to change these
results, the reference values are made again in the same way.

The test fails when time reversal is left out of the symmetrization: dVdu
then differs from the reference by 5e-2 and dDijdu by 5e-2 (Cr) and 1.3e-1
(Fe-ncl) of their largest values.

dqijdu is not stored.  qij of the two supercells are equal, so dqijdu is zero,
and the test checks that it is.
