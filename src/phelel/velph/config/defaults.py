"""Default settings of velph init.

The defaults are written as a velph init template in TOML, and the same
loader reads them as a template given by the user.  The template of the user
and the command line override them.

[init] here holds only values that velph init uses itself, such as codes and
kspacing.  A default that belongs to a section is written in that section,
for example cell in [vasp.relax] and amplitude in [phelel].

"""

from __future__ import annotations

import tomli

from phelel.velph.config.io import parse_template
from phelel.velph.config.schema import VelphConfig

DEFAULT_TEMPLATE_TOML = r'''
[init]
codes = ["phelel", "phonopy", "phono3py"]
tolerance = 1e-05
symmetrize_cell = false
find_primitive = true
primitive_cell_choice = "standardized"
use_grg = false
kspacing = 0.1
kspacing_dense = 0.05
split_site_mixture = false

[phelel]
amplitude = 0.03
diagonal = false
plusminus = true
nosym = false

[phonopy]
amplitude = 0.01
diagonal = false
plusminus = true
nosym = false

[phono3py]
amplitude = 0.03
diagonal = false
plusminus = true
nosym = false

[vasp.incar]
ismear = 0
sigma = 0.01
ediff = 1e-08
encut = 500
prec = "accurate"
lreal = false
lwave = false
lcharg = false

[vasp.relax]
cell = "unitcell"
[vasp.relax.incar]
ediffg = -1e-06
ibrion = 2
isif = 3
nsw = 10

[vasp.nac]
cell = "primitive"
incar_unset = ["npar", "ncore", "kpar"]
[vasp.nac.incar]
lepsilon = true

[vasp.phelel.incar]
elph_prepare = true
isym = 0

[vasp.phonopy.incar]
addgrid = true
isym = 0

[vasp.phono3py.incar]
addgrid = true
isym = 0

[vasp.selfenergy.incar]
elph_run = true
elph_selfen_fan = true
elph_selfen_dw = true
elph_selfen_delta = 0.01
elph_selfen_temps = [0, 300]
elph_ismear = -24

[vasp.transport.incar]
elph_fermi_nedos = 501
elph_ismear = -24
elph_mode = "transport"
elph_selfen_carrier_den = 0.0
elph_scattering_approx = ["serta", "mrta_lambda"]
elph_selfen_temps = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
  110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,
  260, 270, 280, 290, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
  1300, 1400,
]
elph_transport_nedos = 501

[vasp.ph_selfenergy.incar]
elph_fermi_nedos = 501
elph_run = true
elph_selfen_temps = [0, 300]
elph_ismear = -24
elph_driver = "ph"
elph_mode = "superconductivity"
elph_selfen_carrier_den = 0.0

[vasp.el_bands.incar]
ibrion = -1
nsw = 0
[vasp.el_bands.kpoints_opt]
line = 51

[vasp.el_dos.incar]
ibrion = -1
nsw = 0
lorbit = 11
nedos = 5001
ismear = -5

[vasp.ph_bands]
incar_unset = ["ismear", "sigma", "ediff", "lreal", "lwave", "lcharg"]
[vasp.ph_bands.incar]
ibrion = -1
nsw = 0
elph_run = true
[vasp.ph_bands.kpoints]
mesh = [1, 1, 1]
[vasp.ph_bands.qpoints]
line = 51

[scheduler]
job_name = "vasp-elph"
mpirun_command = "mpirun"
vasp_binary = "vasp_std"
prepend_text = ""
append_text = ""
nodes = 1
walltime = "96:00:00"
scheduler_template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --time={walltime}
#SBATCH --output=ci_%j.log

{prepend_text}
{mpirun_command} {vasp_binary}
{append_text}
"""
'''


def get_default_config() -> VelphConfig:
    """Return the default settings of velph init."""
    return parse_template(tomli.loads(DEFAULT_TEMPLATE_TOML), name="defaults")
