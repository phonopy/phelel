"""Directories and files of a velph project.

The velph commands write and read the files of a project in fixed places,
relative to one directory, the root of the project.  The root is the current
directory for the commands.  ProjectDirectory gives all these places.

- The VASP inputs of an input set are written in the directory named after the
  input set, for example "el_bands" for [vasp.el_bands].  relax is written in
  relax/iter1, relax/iter2, ..., and the supercells of a code in
  <code>/disp-000, <code>/disp-001, ..., where disp-000 is the perfect
  supercell.
- The init command of a code writes <code>/<code>_disp.yaml.
- velph phelel differentiate writes phelel/phelel_params.hdf5, and velph phelel
  phonopy writes phelel/phonopy_params.yaml.
- The NAC parameters are read from nac/vasprun.xml.
- The plot commands read vaspout.h5 in the directory of an input set, and
  velph ph_selfenergy dump_phono3py writes ph_selfenergy/gamma_elph{mesh}.hdf5.

The other modules of velph get these places from ProjectDirectory and do not
write them themselves.  test/velph/jobs/test_project_directory.py fails when a
string of another module writes one of them, such as "el_dos/vaspout.h5".

"""

from __future__ import annotations

import dataclasses
import pathlib

from phelel.velph.config.schema import CodeName, InputSetName

# Minimum number of digits of the number of a supercell, as in disp-001.
_MIN_DISP_DIGITS = 3


@dataclasses.dataclass(frozen=True)
class ProjectDirectory:
    """Directories and files of a velph project, relative to its root.

    The names of the files and directories of a project are written only
    here, so that a name is changed in one place.
    test/velph/jobs/test_project_directory.py fails when another module of
    velph writes one of them as a string.  root is a parameter only so that
    the tests can use a directory other than the current directory.

    Attributes
    ----------
    root : pathlib.Path
        Directory of the project.  Default is pathlib.Path(""), the current
        directory.

    """

    root: pathlib.Path = pathlib.Path("")

    def input_set_directory(self, name: InputSetName) -> pathlib.Path:
        """Return the directory named after an input set, such as "el_bands".

        The VASP inputs of an input set are written in this directory.  Those
        of each relax and of the supercells of phelel, phonopy and phono3py are
        written in the directories under it, given by relax_directory and
        disp_directories.

        """
        return self.root / name

    def vaspout(self, name: InputSetName) -> pathlib.Path:
        """Return vaspout.h5 of the VASP calculation of an input set."""
        return self.input_set_directory(name) / "vaspout.h5"

    def plot_pdf(self, name: InputSetName) -> pathlib.Path:
        """Return <name>/<name>.pdf, the plot of el_bands or ph_bands."""
        return self.input_set_directory(name) / f"{name}.pdf"

    def relax_directory(self, i: int) -> pathlib.Path:
        """Return relax/iter<i>, the directory of the i-th relax, counting from 1."""
        return self.root / "relax" / f"iter{i}"

    def disp_directories(self, code: CodeName, n: int) -> list[pathlib.Path]:
        """Return <code>/disp-* of the perfect supercell and n supercells.

        The number of a directory has as many digits as n needs, and at least
        three, for example disp-000, disp-001, ..., disp-012 for n = 12.

        """
        nd = max(len(str(n)), _MIN_DISP_DIGITS)
        return [self.root / code / f"disp-{i:0{nd}d}" for i in range(n + 1)]

    @staticmethod
    def disp_id(directory: pathlib.Path) -> str:
        """Return the number of a directory of disp_directories, such as "001".

        The number is appended to job_name of the scheduler settings.

        """
        return directory.name.removeprefix("disp-")

    def disp_yaml(self, code: CodeName) -> pathlib.Path:
        """Return <code>/<code>_disp.yaml, written by the init command of code."""
        return self.root / code / f"{code}_disp.yaml"

    @property
    def phelel_params(self) -> pathlib.Path:
        """Return phelel/phelel_params.hdf5, written by velph phelel differentiate."""
        return self.root / "phelel" / "phelel_params.hdf5"

    @property
    def phonopy_params(self) -> pathlib.Path:
        """Return phelel/phonopy_params.yaml, written by velph phelel phonopy."""
        return self.root / "phelel" / "phonopy_params.yaml"

    @property
    def gamma_elph_hdf5(self) -> pathlib.Path:
        """Return ph_selfenergy/gamma_elph{mesh}.hdf5 of dump_phono3py.

        {mesh} is replaced by the mesh, such as "-m202020", when the file is
        written.

        """
        return self.input_set_directory("ph_selfenergy") / "gamma_elph{mesh}.hdf5"

    @property
    def nac_directory(self) -> pathlib.Path:
        """Return the directory of the NAC calculation, nac."""
        return self.root / "nac"
