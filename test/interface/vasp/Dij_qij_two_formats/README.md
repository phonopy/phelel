# PAW strengths and overlaps in the two output formats of VASP

Each directory holds Dij and qij of one VASP calculation with LWAP = .TRUE.,
which writes them both in `vaspout.h5` and in the old format:

- `inwap.yaml`, `PAW-STRENGTH.bin`, `PAW-OVERLAP.bin`: as written by VASP,
- `vaspout-paw.h5`: `/results/paw/dij` and `/results/paw/qij` copied from
  `vaspout.h5` of the same calculation.

| directory | crystal | ncdij | MAGMOM | VASP | POTCAR |
|---|---|---|---|---|---|
| `Cr` | bcc Cr, conventional cell, 2 atoms | 2 | 2.0 -2.0 | 6.6.1 | PAW_PBE Cr 06Sep2000 |
| `Fe-ncl` | bcc Fe, conventional cell, 2 atoms | 4 | 6*3.0 | 6.6.0 | PAW_PBE Fe 06Sep2000 |

Both are perfect supercells with ENCUT = 300, PREC = Accurate, ISYM = 0,
ISMEAR = 1 and SIGMA = 0.2.  Cr uses ISPIN = 2 and EDIFF = 1e-8, and is
antiferromagnetic.  Fe-ncl uses LNONCOLLINEAR = .TRUE. without LSORBIT,
EDIFF = 1e-6 and NELM = 200, and its electronic self-consistency did not
reach EDIFF.  Only the agreement of the two formats of the same calculation is
tested with these data.

In Cr, the two spin components and the two atoms of Dij differ, and in
Fe-ncl, Dij has imaginary parts and its four spin components differ.  In both,
Dij and qij of each spin component and atom are symmetric in the two lm
indices.
