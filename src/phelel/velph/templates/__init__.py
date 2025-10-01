"""Templates of vasp inputs written in velph.toml.

``template_dict`` is used to generate ``velph.toml``, and these have a similar
dictionary structure.

"phelel":
    General parameters to run phelel. "supercell_dimension":
        list[list[int, int, int]], list[int, int, int], or int Supercell
        dimension with respect to unit cell. Typically given by three integers,
        but nine integers can be used. If one integer value is given, supercell
        dimension is estimated by taking this value as the maximum number of
        atoms in the supercell.
    "fft_mesh":
        FFT mesh used in primitive cell for sandwich.
"vasp":
    Types of VASP calculations, e.g., "phelel", "relax", "nac", "selfenergy",
    "transport", and default set of INCAR settins, "incar".

    The default incar settings in the "incar" section are copied to the
    "incar" sections of all VASP calculation types if those default INCAR
    settings are not available in each section. To prevent from writing
    default INCAR settings, i.e., avoiding unnecessary INCAR settings in a
    specific VASP calculation type, INCAR tag with empty dictionary `{}` has to
    be explicitly described (see "ph_bands" section.)

    In each VASP calculation following settings are specified.
        "incar":
            VASP INCAR setting by a dictionary.
        "kpoints":
            Sampling mesh of k-points. "mesh": list[list[int, int, int]],
            list[int, int, int]
                Typically three integers. With nine integers, generalized
                regular grid is used.
            "shift": list[float, float, float]
                Shift of mesh with respect to neighboring grid points. 0.5 means
                half-grid shift.
            "kspacing": float
                This value is treated as same as KSPACING with KGAMMA=.TRUE. in
                VASP INCAR setting.
        "kpoints_dense":
            This is used for sandwich. Charge density calculation is performed
            using the configuration of "kpoints".
"scheduler":
    Schedular configuration. Values of the kyes are inserted to specific
    schedular templeate found in ``velph/template/scheduler``. This section is
    replaced by ``.config/velph/scheduler.toml`` if this path exists.

"""

from __future__ import annotations

from typing import Any

default_template_dict: dict[str, Any] = {
    "phelel": {},
    "phonopy": {},
    "phono3py": {},
    "vasp": {
        "incar": {
            "ismear": 0,
            "sigma": 0.01,
            "ediff": 1e-8,
            "encut": 500,
            "prec": "accurate",
            "lreal": False,
            "lwave": False,
            "lcharg": False,
        },
        "selfenergy": {
            "incar": {
                "elph_run": True,
                "elph_selfen_fan": True,
                "elph_selfen_dw": True,
                "elph_selfen_delta": 0.01,
                "elph_selfen_temps": [300],
                "elph_nbands": 100,
                "elph_nbands_sum": [50, 100],
                "elph_selfen_gaps": True,
                "elph_ismear": -24,
                "elph_fermi_nedos": 21,
            },
        },
        "transport": {
            "incar": {
                "elph_fermi_nedos": 501,
                "elph_ismear": -24,
                "elph_mode": "transport",
                "elph_selfen_carrier_den": 0.0,
                "elph_scattering_approx": ["serta", "mrta_lambda"],
                "elph_selfen_temps": [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    20,
                    30,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                    100,
                    110,
                    120,
                    130,
                    140,
                    150,
                    160,
                    170,
                    180,
                    190,
                    200,
                    210,
                    220,
                    230,
                    240,
                    250,
                    260,
                    270,
                    280,
                    290,
                    300,
                    400,
                    500,
                    600,
                    700,
                    800,
                    900,
                    1000,
                    1100,
                    1200,
                    1300,
                    1400,
                ],
                "elph_transport_nedos": 501,
            },
        },
        "el_bands.dos": {
            "incar": {
                "ibrion": -1,
                "nsw": 0,
                "lorbit": 11,
                "nedos": 5001,
                "ismear": -5,
            }
        },
        "el_bands.bands": {
            "incar": {
                "ibrion": -1,
                "nsw": 0,
            },
            "kpoints_opt": {
                "line": 51,
            },
        },
        "ph_bands": {
            "incar": {
                "ibrion": -1,
                "nsw": 0,
                "elph_run": True,
                "ismear": {},
                "sigma": {},
                "ediff": {},
                "lreal": {},
                "lwave": {},
                "lcharg": {},
            },
            "qpoints": {
                "line": 51,
            },
        },
        "phonopy": {
            "incar": {
                "addgrid": True,
                "isym": 0,
            },
        },
        "phono3py": {
            "incar": {
                "addgrid": True,
                "isym": 0,
            },
        },
        "phelel": {
            "incar": {
                "elph_prepare": True,
                "isym": 0,
            },
        },
        "relax": {
            "incar": {
                "ediffg": -1e-6,
                "ibrion": 2,
                "isif": 3,
                "nsw": 10,
            },
        },
        "nac": {
            "incar": {
                "lepsilon": True,
                "npar": {},
                "ncore": {},
                "kpar": {},
            },
        },
    },
    "scheduler": {
        "job_name": "vasp-elph",
        "mpirun_command": "mpirun",
        "vasp_binary": "vasp_std",
        "prepend_text": "",
        "append_text": "",
        "nodes": 1,
        "walltime": "96:00:00",
        "scheduler_template": """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --time={walltime}
#SBATCH --output=ci_%j.log

{prepend_text}
{mpirun_command} {vasp_binary}
{append_text}
""",
    },
}
