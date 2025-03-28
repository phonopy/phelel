"""Utilities to handle VASP files."""

from __future__ import annotations

import io
import os
from collections import defaultdict
from typing import Optional, Union

import h5py
import numpy as np
from phonopy.structure.cells import PhonopyAtoms


class VaspKpoints:
    """KPOINTS file writer.

    At least following three formats must be supported.

    ---------------------------------------------
    # Generated by velph
    0
    Gamma
            18            18            18
    0.000000000   0.000000000   0.000000000
    ---------------------------------------------

    ---------------------------------------------
    # Generated by velph
    0
    Reciprocal
    0.000000000   0.045454545   0.045454545
    0.045454545   0.000000000   0.045454545
    0.045454545   0.045454545   0.000000000
    0.000000000   0.000000000   0.000000000
    ---------------------------------------------

    ---------------------------------------------
    k points along high symmetry lines by velph
    51
    line mode
    fractional
    0.00000000 0.00000000 0.00000000 GAMMA
    0.50000000 0.00000000 0.50000000 X

    0.50000000 0.00000000 0.50000000 X
    0.62500000 0.25000000 0.62500000 U

    0.62500000 0.25000000 0.62500000 U
    0.37500000 0.37500000 0.75000000 K

    0.37500000 0.37500000 0.75000000 K
    0.00000000 0.00000000 0.00000000 GAMMA
    ---------------------------------------------

    """

    @classmethod
    def write_line_mode(
        cls,
        fp: Union[str, bytes, os.PathLike, io.IOBase],
        cell: PhonopyAtoms,
        toml_dict: dict,
    ) -> None:
        """Write KPOINTS with line mode."""
        lines = cls._get_line_mode(toml_dict, cell)
        cls._write_lines(fp, lines)

    @staticmethod
    def _write_lines(fp, lines):
        if isinstance(fp, io.IOBase):
            fp.write(("\n".join(lines)).encode("utf-8"))
        else:
            with open(fp, "w") as w:
                print("\n".join(lines), file=w)

    @staticmethod
    def _get_line_mode(toml_dict: dict, cell: PhonopyAtoms) -> list:
        """Return KPOINTS lines.

        toml_dict['line'] is necessary.

        """
        if "line" not in toml_dict:
            raise RuntimeError("line in toml_dict is not found.")
        lines = []
        lines.append("k points along high symmetry lines by velph")
        lines.append(str(toml_dict["line"]) + "")
        lines.append("line mode")
        lines.append("fractional")

        if "path" in toml_dict:
            for p in list(toml_dict["path"]):
                for i in range(2):
                    coord = " ".join([f"{toml_dict[p[i]][j]:.8f}" for j in range(3)])
                    lines.append(f"{coord} {p[i]}")
                lines.append("")
        else:
            try:
                import seekpath
            except ImportError as exc:
                raise ModuleNotFoundError("You need to install seekpath.") from exc

            band_path = seekpath.get_path(cell.totuple())
            for p in list(band_path["path"]):
                for i in range(2):
                    coord = " ".join(
                        [f"{band_path['point_coords'][p[i]][j]:.8f}" for j in range(3)]
                    )
                    lines.append(f"{coord} {p[i]}")
                lines.append("")
        return lines

    @classmethod
    def write_mesh_mode(
        cls,
        fp: Union[str, bytes, os.PathLike, io.IOBase],
        toml_dict: dict,
    ):
        """Write KPOINTS with mesh mode."""
        lines = cls._get_mesh_mode(toml_dict)
        cls._write_lines(fp, lines)

    @staticmethod
    def _get_mesh_mode(toml_dict: dict) -> list:
        if "mesh" not in toml_dict:
            raise RuntimeError('"mesh" not found in dict.')

        mesh = toml_dict["mesh"]
        if "shift" in toml_dict:
            shift = toml_dict["shift"]
        else:
            shift = [0, 0, 0]

        lines = ["# Generated by velph"]
        lines.append("0")
        if isinstance(mesh[0], int):
            lines.append("Gamma")
            lines.append(" ".join([f"{v:13d}" for v in mesh]))
            lines.append(" ".join([f"{v:20.16f}" for v in shift]))
        else:
            lines.append("Reciprocal")
            mesh_inv_t = np.linalg.inv(mesh).T
            mesh_inv_t = np.where(np.abs(mesh_inv_t) < 1e-15, 0, mesh_inv_t)
            for vec in mesh_inv_t:
                lines.append(" ".join([f"{v:20.16f}" for v in vec]))
            lines.append(" ".join([f"{v:20.16f}" for v in shift]))
        return lines


class VaspIncar:
    """Write INCAR file and expand INCAR value.

    Supported data types
    --------------------
    int
    float
    str
    bool

    INCAR writer
    ------------
    Tag name is converted to uppercase.
    True is represented by ".TRUE.".
    False is represented by ".FALSE.".
    Tags are sorted by alphabetical order.

    INCAR value expander
    --------------------
    Asterisk (*) is considered to expand INCAR value.

    """

    specific_tags = []

    @classmethod
    def write(cls, fp: Union[str, bytes, os.PathLike, io.IOBase], toml_dict: dict):
        """Write INCAR."""
        lines = cls._get_incar(toml_dict)
        cls._write_lines(fp, lines)

    @staticmethod
    def expand(line: str) -> list[float]:
        """Expand right hand side of INCAR tag line.

        Parameters
        ----------
        line: str
            "3*1 0 0 0 3*2" -> [1., 1., 1., 0., 0., 0., 2., 2., 2.]

        Returns
        -------
        list[float]
            Expanded INCAR value.

        """
        vals = []
        for chunk in line.split():
            if "*" in chunk:
                n, v = chunk.split("*")
                vals += [float(v)] * int(n)
            else:
                vals.append(float(chunk))
        return vals

    @classmethod
    def _get_incar(cls, toml_dict: dict) -> list:
        sorted_list = sorted(toml_dict.items(), key=lambda pair: pair[0].lower())
        lines = []
        for key, value in sorted_list:
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    lines.append(f"{key.upper()} = .TRUE.")
                else:
                    lines.append(f"{key.upper()} = .FALSE.")
            elif isinstance(value, list):
                flattened_list = cls._flatten_list(value)
                if key == "magmom":
                    list_str = cls._compress_magmom_values(flattened_list)
                else:
                    list_str = " ".join([f"{v}" for v in flattened_list])
                lines.append(f"{key.upper()} = {list_str}")
            elif isinstance(value, str):
                lines.append(f"{key.upper()} = {value}")
            elif isinstance(value, (float, int)):
                lines.append(f"{key.upper()} = {value}")
            else:
                raise NotImplementedError(
                    f"Handler for value type {type(value)} not implemented."
                )
        return lines

    @staticmethod
    def _write_lines(fp: Union[str, bytes, os.PathLike, io.IOBase], lines: list[str]):
        if isinstance(fp, io.IOBase):
            fp.write(("\n".join(lines)).encode("utf-8"))
        else:
            with open(fp, "w") as w:
                print("\n".join(lines), file=w)

    @staticmethod
    def _compress_magmom_values(values, tolerance=1e-15):
        uniq_vals = [[values[0]]]
        for i, v in enumerate(values[1:]):
            if abs(v - values[i]) < tolerance:
                uniq_vals[-1].append(v)
            else:
                uniq_vals.append([v])
        return " ".join([f"{len(vals)}*{vals[0]}" for vals in uniq_vals])

    @classmethod
    def _flatten_list(cls, values):
        val_list = []
        for v in values:
            if isinstance(v, list):
                val_list += cls._flatten_list(v)
            else:
                val_list.append(v)
        return val_list


class VaspPotcar:
    """POTCAR reader.

    Attributes
    ----------
    enmax : list[float]
        ENMAX of atoms.

    """

    value_types = {"ENMAX": float, "TITEL": str}

    def __init__(self, fp: Union[str, bytes, os.PathLike, io.IOBase]):
        """Init method."""
        self._property_dict = defaultdict(list)
        self._read(fp)

    @property
    def enmax(self) -> list[float]:
        """Return values of ENMAX in POTCAR as a list."""
        return self._property_dict["ENMAX"]

    @property
    def titel(self) -> list[float]:
        """Return values of TITEL in POTCAR as a list."""
        return self._property_dict["TITEL"]

    def _read(self, fp: Union[str, bytes, os.PathLike, io.IOBase]):
        if isinstance(fp, io.IOBase):
            self._parse(fp)
        else:
            with open(fp) as _fp:
                self._parse(_fp)

    def _parse(self, fp: io.IOBase):
        """Parse lines of POTCAR.

        This method will be more complicated to support more properties.

        """
        for line in fp:
            for key in self.value_types:
                if key in line:
                    if key == "ENMAX":
                        ary = line.split()
                        index = ary.index(key)
                        valstr = ary[index + 2].strip(" ;")
                        value = self.value_types[key](valstr)
                        self._property_dict[key].append(value)
                    elif key == "TITEL":
                        ary = line.split("=")
                        self._property_dict[key].append(ary[1].strip())


class CutoffToFFTMesh:
    """Determin NGX, NGY, NGX."""

    @classmethod
    def get_FFTMesh(
        cls, cutoff_eV: float, lattice: np.ndarray, incar_prec: Optional[str] = None
    ) -> np.ndarray:
        """Return FFT mesh corresponding to (NGX, NGY, NGZ).

        The return value has to be dividable by 2 and factorized by 2, 3, 5, and 7.

        Parameters
        ----------
        lattice : np.ndarray
            Basis vectors in row vectors.
        cutoff_eV : float
            Plane wave cutoff energy in eV.
        incar_prec : str or None
            Default is None, corresponding to "normal".
            cutoff_factor is 4 if 'high', 'accurate', or 'single'. Otherwise 3.
            Note that this is VASP6 definition.

        """
        lengths = np.linalg.norm(lattice, axis=1) / 0.529177249
        cutoff = np.sqrt(cutoff_eV / 13.605826) / (2 * np.pi / lengths)
        fft_mesh = np.rint(cutoff * cls._get_cutoff_factor(incar_prec)).astype(int)
        for i in range(3):
            while not cls._is_factorized_by_2357(fft_mesh[i]):
                fft_mesh[i] += 1
        return fft_mesh

    @staticmethod
    def _get_cutoff_factor(prec: Optional[str]) -> int:
        """Return factor to multiply to cutoff.

        Note this is the VASP6 convention.

        """
        if prec == "singlen" or prec is None:
            return 3
        if prec[0].lower() in ("h", "a", "s"):
            return 4
        else:
            return 3

    @staticmethod
    def _is_factorized_by_2357(n: int) -> bool:
        """Check if n can be dividable by 2 and be factorized only by 2, 3, 5, and 7."""
        if (n // 2) * 2 != n:
            return False

        _n = n
        for div in (2, 3, 5, 7):
            while (_n // div) * div == _n:
                _n = _n // div

        if _n == 1:
            return True
        else:
            return False


def read_magmom(magmom: str) -> Optional[list[float]]:
    """Read text file to obtain MAGMOM information.

    Parameters
    ----------
    magmom : str
        String corresponding to INCAR MAGMOM tag value, e.g., "24*1" or "0 0 1".

    """
    return VaspIncar().expand(magmom.strip())


def read_crystal_structure_from_h5(f_vaspout_h5: h5py.File, group: str) -> PhonopyAtoms:
    """Read crystal structure from vaspout.h5."""
    direct = int(f_vaspout_h5[f"{group}/direct_coordinates"][()])
    scale = f_vaspout_h5[f"{group}/scale"][()]
    lattice = f_vaspout_h5[f"{group}/lattice_vectors"][:] * scale
    positions = f_vaspout_h5[f"{group}/position_ions"][:]
    number_ion_types = f_vaspout_h5[f"{group}/number_ion_types"][:]
    ion_types = f_vaspout_h5[f"{group}/ion_types"][:]

    symbols = []
    for symbol, number in zip(ion_types, number_ion_types):
        symbols += [symbol.decode()] * number

    if not direct:
        positions = positions @ np.linalg.inv(lattice)

    assert len(symbols) == len(positions)

    cell = PhonopyAtoms(cell=lattice, scaled_positions=positions, symbols=symbols)

    return cell
