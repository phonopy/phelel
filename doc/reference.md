(references)=
# References

This page presents the methodologies and software packages that the phelel and
velph codes rely on.

(reference_vasp)=
## VASP

The electronic-structure calculations and the electron-phonon interaction
properties are computed with the VASP code
([https://www.vasp.at/](https://www.vasp.at/)). phelel prepares the inputs for
VASP and post-processes its outputs; it cannot compute electron-phonon
properties on its own.

The references recommended for citing VASP are:

- G. Kresse and J. Hafner, Phys. Rev. B **47**, 558 (1993); **49**, 14251 (1994).
- G. Kresse and J. Furthmüller, Comput. Mater. Sci. **6**, 15 (1996).
- G. Kresse and J. Furthmüller, Phys. Rev. B **54**, 11169 (1996).
- G. Kresse and J. Hafner, J. Phys.: Condens. Matter **6**, 8245 (1994) (ultrasoft pseudopotentials).
- G. Kresse and D. Joubert, Phys. Rev. B **59**, 1758 (1999) (PAW method).

When using a specific VASP feature, such as the electron-phonon routines, also
cite the papers listed in the corresponding page on the
[VASP wiki](https://www.vasp.at/wiki/).

(reference_phonopy)=
## phonopy

Harmonic phonon properties and the underlying crystal-structure and displacement
handling build on phonopy
([https://phonopy.github.io/phonopy/](https://phonopy.github.io/phonopy/)). See
also the
[phonopy citation page](https://phonopy.github.io/phonopy/citation.html).

- A. Togo, L. Chaput, T. Tadano, and I. Tanaka, "Implementation strategies in
  phonopy and phono3py", J. Phys. Condens. Matter **35**, 353001 (2023).
- A. Togo, "First-principles Phonon Calculations with Phonopy and Phono3py",
  J. Phys. Soc. Jpn. **92**, 012001 (2023).

(reference_phono3py)=
## phono3py

Anharmonic phonon properties, including lattice thermal conductivity, are
computed with phono3py
([https://phonopy.github.io/phono3py/](https://phonopy.github.io/phono3py/)). See
also the
[phono3py citation page](https://phonopy.github.io/phono3py/citation.html).

- A. Togo, L. Chaput, and I. Tanaka, "Distributions of phonon lifetimes in
  Brillouin zones", Phys. Rev. B **91**, 094306 (2015).
- A. Togo, L. Chaput, T. Tadano, and I. Tanaka, "Implementation strategies in
  phonopy and phono3py", J. Phys. Condens. Matter **35**, 353001 (2023).

(reference_spglib)=
## spglib

Crystal symmetry, including space-group determination and primitive-cell
reduction, is handled by spglib
([https://spglib.readthedocs.io/](https://spglib.readthedocs.io/)).

- A. Togo, K. Shinohara, and I. Tanaka, "Spglib: a software library for crystal
  symmetry search", Sci. Technol. Adv. Mater. Methods **4**, 2384822 (2024).
- K. Shinohara, A. Togo, and I. Tanaka, "Algorithms for magnetic symmetry
  operation search and identification of magnetic space group from magnetic
  crystal structure", Acta Cryst. A **79**, 390-398 (2023) (magnetic space
  groups).

(reference_finufft)=
## finufft

The local potentials from VASP calculation results are interpolated using
[finufft](https://finufft.readthedocs.io/). For more information and references
on finufft, visit
[https://finufft.readthedocs.io/en/latest/refs.html](https://finufft.readthedocs.io/en/latest/refs.html).
