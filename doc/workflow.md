(workflow)=
# Workflow of el-ph calculation

```{note}
To perform the workflow for electron-phonon related properties (including
thermoelectric properties), {ref}`velph_command` can be particularly helpful in
managing the required calculations. For instance, it can generate VASP inputs
for supercells with displacements as well as the parameters needed to
incorporate the long-range dipole contribution in electron-phonon interactions.
Additionally, it enables monitoring of both electronic and phonon band
structures.
```

## Minimal calculation steps

1. **(Phelel)** Generate supercell structures with atomic displacements based on the initial unit cell.
2. **(VASP)** Perform calculations on these supercell structures to obtain local potentials and PAW strengths under the applied atomic displacements.
3. **(Phelel)** Gather VASP results and compute derivatives of the local potentials and PAW strengths with respect to the atomic displacements.
4. **(VASP)** Use these derived quantities to calculate electron–phonon interaction strengths and related properties, such as:
   - Electrical conductivity
   - Seebeck coefficient
   - Electrical thermal conductivity

## Additional steps (Optional)

**Before step 4**

- Electronic band structure and density of states calculations
- Dielectric constant and Born effective charge calculations
- Phonon band structure calculations

**After step 4**

- Lattice thermal conductivity calculations needed to compute the thermoelectric
  figure of merit
