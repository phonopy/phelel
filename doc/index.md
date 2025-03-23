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


## License

BSD-3-Clause.

## Contributors

- {user}`Atsushi Togo <atztogo>` (National Institute for Materials Science)

```{toctree}
:hidden:
install
workflow
phelel-command
velph-command
velph-toml
velph-toml-scheduler
velph-init
velph-init-template
velph-subcommands
velph-example
reference
changelog
```
