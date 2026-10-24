# PAW strengths Dij of perfect supercells

Each file holds Dij of the perfect supercell of one VASP calculation, with what
is needed to apply the symmetry operations to it:

- `dij`: `/results/paw/dij` of `vaspout.h5` as complex numbers, shape
  (ncdij, natom, lmdim, lmdim),
- `lm_channels`: the lm channels of the atoms as a JSON string, as returned by
  `read_inwap_vaspouth5` in `lm_orbitals`,
- `lattice`, `scaled_positions`, `symbols`: the supercell,
- `magnetic_moments`: MAGMOM of the VASP calculation.

| file | crystal | ncdij | MAGMOM | VASP | POTCAR |
|---|---|---|---|---|---|
| `Fe.npz` | bcc Fe, conventional cell, 2 atoms | 2 | 2*3.0 | 6.6.0 | PAW_PBE Fe 06Sep2000 |
| `Fe-ncl.npz` | bcc Fe, conventional cell, 2 atoms | 4 | 6*3.0 | 6.6.0 | PAW_PBE Fe 06Sep2000 |
| `C-ncl.npz` | diamond, conventional cell, 8 atoms | 4 | 24*0.0 | 6.6.1 | PAW_PBE C 08Apr2002 |

All calculations use ENCUT = 300, PREC = Accurate, EDIFF = 1e-8 and ISYM = 0,
so VASP does not symmetrize Dij.  Fe and Fe-ncl use ISMEAR = 1 and SIGMA = 0.2.
Fe uses ISPIN = 2.  Fe-ncl and C-ncl use LNONCOLLINEAR = .TRUE. without
LSORBIT.  In Fe-ncl, the converged magnetization is about 0.04 degrees off the
direction (1, 1, 1) of MAGMOM.

Only Dij is stored.
