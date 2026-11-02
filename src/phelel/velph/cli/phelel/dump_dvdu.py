"""Implementation of velph-phelel-dump-dvdu."""

from __future__ import annotations

from collections.abc import Sequence

import click
import numpy as np
from phonopy.structure.cells import get_primitive

from phelel.file_IO import read_phelel_params_hdf5
from phelel.velph.jobs.project_directory import ProjectDirectory


def dump_dvdu(
    atom_numbers: Sequence[int] = (), project_dir: ProjectDirectory | None = None
) -> None:
    """Write dV/du of phelel/phelel_params.hdf5 in LOCPOT-like files in phelel/.

    The files of DLocalPotential.visualize are written for the atoms given by
    their numbers in the supercell counted from 1.  dV/du is given for the atoms
    of the primitive cell in the supercell, and all of them are written by
    default.  The directories are those of project_dir, in the current
    directory by default.  click.ClickException is raised when
    phelel_params.hdf5 does not exist or an atom number is not one of the
    atoms of dV/du.

    """
    if project_dir is None:
        project_dir = ProjectDirectory()
    hdf5_filename = project_dir.phelel_params
    if not hdf5_filename.exists():
        raise click.ClickException(f'"{hdf5_filename}" does not exist.')

    dVdu, _, _, _ = read_phelel_params_hdf5(hdf5_filename)
    numbers = [int(i) + 1 for i in dVdu.atom_indices]
    for n in atom_numbers:
        if n not in numbers:
            raise click.ClickException(
                f"Atom {n} is not one of the atoms of dV/du "
                f"({', '.join(str(i) for i in numbers)})."
            )

    pcell = get_primitive(dVdu.supercell, np.linalg.inv(dVdu.p2s_matrix))
    for i_atom, n in enumerate(numbers):
        if atom_numbers and n not in atom_numbers:
            continue
        filenames = dVdu.visualize(pcell, i_atom, directory=hdf5_filename.parent)
        click.echo(
            f"dV/du of atom {n} was written in {len(filenames)} files "
            f'"{hdf5_filename.parent / f"locpot_viz-{n:03d}-*.dat"}".'
        )
