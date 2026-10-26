# PAW strengths and overlaps in the two output formats of VASP

Each directory holds Dij and qij of one VASP calculation with LWAP = .TRUE.,
which writes them both in `vaspout.h5` and in the old format:

- `inwap.yaml`, `PAW-STRENGTH.bin`, `PAW-OVERLAP.bin`: as written by VASP,
- `vaspout-paw.h5`: `/results/paw/dij` and `/results/paw/qij` copied from
  `vaspout.h5` of the same calculation.

| directory | crystal | ncdij | MAGMOM | VASP | POTCAR |
|---|---|---|---|---|---|
| `Cr` | bcc Cr, conventional cell, 2 atoms | 2 | 2.0 -2.0 | 6.6.1 | PAW_PBE Cr 06Sep2000 |
| `Fe-ncl` | bcc Fe, conventional cell, 2 atoms | 4 | 6*3.0 | 6.6.0 | PAW Fe 03Mar1998 (LDA) |
| `Fe-lsorbit` | bcc Fe, conventional cell, 2 atoms | 4 | 6*3.0 | 6.6.0 | PAW Fe 03Mar1998 (LDA) |

All are perfect supercells with ENCUT = 300, PREC = Accurate, EDIFF = 1e-8,
ISYM = 0, ISMEAR = 1 and SIGMA = 0.2, and reached EDIFF.  Cr uses ISPIN = 2
and is antiferromagnetic.  Fe-ncl uses LNONCOLLINEAR = .TRUE. without
LSORBIT, LDA and IALGO = 48, and is the calculation of
`test/base/Dij_perfect/Fe-ncl.npz`.  Fe-lsorbit is the calculation of Fe-ncl
with LSORBIT = .TRUE. added; the INCAR, POSCAR and KPOINTS differ only in this
tag.  Only the agreement of the two formats of the same calculation is tested
with these data.  These data and the test that uses them are for the readers
of the old format, and are removed with them.

In Cr, the two spin components and the two atoms of Dij differ, and in Fe-ncl
and Fe-lsorbit, Dij has imaginary parts and its four spin components differ.

Without spin-orbit coupling, Dij and qij of each spin component and atom are
symmetric in the two lm indices: the largest |D - D^T|, with ^T exchanging the
two lm indices, is 0 in Cr and 4.5e-16 eV in Fe-ncl.  In Fe-lsorbit, Dij is
not symmetric in them: the largest |D - D^T| is 0.28 eV in each of the four
spin components, in the imaginary part of the components 11 and 22 and in
both parts of 12 and 21, while the largest |Dij| is 22 eV.  Dij of Fe-lsorbit
is Hermitian in the index combining spin and lm, to 8e-14 eV.  qij is
symmetric in all three, with |q - q^T| = 0.  These values were read from
`vaspout-paw.h5`.
