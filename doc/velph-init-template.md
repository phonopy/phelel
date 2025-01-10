(velph_init_template)=
# `velph init` template

The `velph init` command is used to prepare the `velph.toml` file. If no custom
template is specified, a default template is applied. To specify a custom
template, use the following command:

```bash
% velph init CELL_FILENAME PROJECT_FOLDER --template-toml velph-tmpl.toml [OPTIONS]
```

The `velph-tmpl.toml` file (which can have any file name) is similar to the
`velph.toml` file. It can be created by modifying an existing `velph.toml`.
However, a key difference is that `velph-tmpl.toml` may include an
`[init.options]` section that is not present in `velph.toml`. This section
allows you to define default values as substitutes for command-line options used
with `velph init`, as described in the next section.

(velph_init_template_init_options)=
## `[init.options]`

```toml
[init.options]
kspacing = 0.2
kspacing_dense = 0.1
max_num_atoms = 120
```

The `[init.options]` keywords correspond to command-line options for `velph
init`. To view a complete list of available options, use:

```bash
% velph init --help
```

(velph_init_template_incar)=
## `[vasp.incar]`

The base INCAR settings are defined in the `[vasp.incar]` section. These
settings can be overridden by the `[vasp.CALC_TYPE.incar]` section if existed,
where `CALC_TYPE` could be `phelel`, `relax`, `nac`, `transport`, `phono3py`,
`phono3py.phonon`, etc. For example, consider the following configuration:

```toml
[vasp.incar]
encut = 400
ncore = 4
gga = "PS"
```

In this case, the `[vasp.phelel.incar]` section in `velph.toml` is initially
populated with these base settings. And then, the default parameters specific to
the `phelel` calculation, as well as any settings defined in the
`[vasp.phelel.incar]` section of the template, will override these
configurations as needed.
