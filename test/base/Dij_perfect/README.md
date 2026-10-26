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
| `Cr.npz` | bcc Cr, conventional cell, 2 atoms | 2 | 2.0 -2.0 | 6.6.0 | PAW_PBE Cr 06Sep2000 |
| `Fe-ncl.npz` | bcc Fe, conventional cell, 2 atoms | 4 | 6*3.0 | 6.6.0 | PAW Fe 03Mar1998 (LDA) |
| `C-ncl.npz` | diamond, conventional cell, 8 atoms | 4 | 24*0.0 | 6.6.1 | PAW_PBE C 08Apr2002 |

All calculations use ENCUT = 300, PREC = Accurate, EDIFF = 1e-8 and ISYM = 0,
so VASP does not symmetrize Dij.  Fe, Cr and Fe-ncl use ISMEAR = 1 and
SIGMA = 0.2.  Fe and Cr use ISPIN = 2.  Fe-ncl and C-ncl use
LNONCOLLINEAR = .TRUE. without LSORBIT.  Cr is antiferromagnetic: the
converged moments of the two atoms are +1.243 and -1.243 mu_B, read from
"magnetization (x)" of OUTCAR of the same calculation with LORBIT = 11.

Fe-ncl uses LDA and IALGO = 48, and its electronic self-consistency reached
EDIFF.  With IALGO = 58, Dij was not converged when the energy reached EDIFF.
Its magnetization in OSZICAR, (2.4389, 2.4388, 2.4387) mu_B, is along the
direction (1, 1, 1) of MAGMOM to the four digits written.  With GGA (PAW_PBE Fe
06Sep2000), the magnetic part of the local potential of the non-collinear
calculation differed from that of the same calculation with ISPIN = 2 by more
than 20 eV at points between the atoms where the magnetization is small, also
with GGA_COMPAT = .FALSE. and with ENCUT = 500 (up to 50 eV).  With LDA, the
two agree to 4e-4 eV at all points.

Only Dij is stored.
