# Phelel

This code provides computations related to electron-phonon interactions using
the finite-displacement method with a supercell approach, as reported in:

- Laurent Chaput, Atsushi Togo, and Isao Tanaka, Phys. Rev. B **100**, 174304
  (2019).
- Manuel Engel, Henrique Miranda, Laurent Chaput, Atsushi Togo, Carla Verdi,
  Martijn Marsman, and Georg Kresse, Phys. Rev. B **106**, 094316 (2022)
- Laurent Chaput, Henrique Miranda, Atsushi Togo, Manuel Engel, Martin Schlipf,
  Martijn Marsman, Georg Kresse, Phys. Rev. B **113**, 014313 (2026)

This code is designed to be used together with the VASP code; electron-phonon
interaction properties cannot be computed using this code alone.

```{note}
To start an electron-phonon calculation with this code, it is recommended to use
the {ref}`velph <velph_command>` command-line tool, which systematically
orchestrates the phelel and VASP workflows.
```

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
