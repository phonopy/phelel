(workflow)=
## Workflow of el-ph calculation

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
