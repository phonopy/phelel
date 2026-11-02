"""Tests for the readers of VASP output files."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pytest

from phelel.interface.vasp.file_IO import (
    get_CHGCAR,
    read_inwap_yaml,
    read_local_potential_vaspouth5,
    read_PAW_Dij_qij,
    read_PAW_Dij_qij_vaspouth5,
)

cwd = Path(__file__).parent


@pytest.mark.parametrize("name", ["Cr", "Fe-ncl", "Fe-lsorbit"])
def test_read_PAW_Dij_qij_two_formats(name: str):
    """Dij and qij of the old format are equal to those of vaspout.h5.

    Both are written by one VASP calculation with LWAP = .TRUE.  Dij of Cr
    differs between the spin components and between the atoms, and Dij of
    Fe-ncl and Fe-lsorbit has imaginary parts and four spin components that
    differ from each other, so mixing up these axes or the real and imaginary
    parts in one of the readers changes Dij.

    Without spin-orbit coupling (Cr and Fe-ncl), Dij and qij are symmetric in
    the two lm indices to rounding errors, so mixing them up changes Dij by no
    more than the rounding errors.  With spin-orbit coupling (Fe-lsorbit), Dij
    is not symmetric in them, so mixing up the two lm indices in one of the
    readers changes Dij by 0.28 eV.  qij is symmetric in all of the data.

    """
    data_dir = cwd / "Dij_qij_two_formats" / name
    inwap = read_inwap_yaml(data_dir / "inwap.yaml")
    dij, qij = read_PAW_Dij_qij_vaspouth5(data_dir / "vaspout-paw.h5")
    if name == "Cr":
        assert np.abs(dij[0] - dij[1]).max() > 0.1
        assert np.abs(dij[:, 0] - dij[:, 1]).max() > 0.1
    else:
        assert np.abs(dij.imag).max() > 0.1
        for s, t in itertools.combinations(range(4), 2):
            assert np.abs(dij[s] - dij[t]).max() > 0.1, (s, t)
    dij_T = np.swapaxes(dij, -1, -2)
    if name == "Fe-lsorbit":
        assert np.abs(dij - dij_T).max() > 0.1
    else:
        np.testing.assert_allclose(dij, dij_T, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(qij, np.swapaxes(qij, -1, -2))

    shape = (inwap["ncdij"], inwap["nions"], inwap["lmdim"], inwap["lmdim"])
    assert dij.shape == shape
    np.testing.assert_array_equal(
        read_PAW_Dij_qij(inwap, data_dir / "PAW-STRENGTH.bin"), dij
    )
    np.testing.assert_array_equal(
        read_PAW_Dij_qij(inwap, data_dir / "PAW-OVERLAP.bin"), qij
    )


def _write_vaspout_potential(
    filename: Path, pot: np.ndarray, version: Sequence[int] | None
):
    with h5py.File(filename, "w") as h5:
        h5.create_dataset("results/potential/total", data=pot)
        if version is not None:
            for key, value in zip(("major", "minor", "patch"), version, strict=True):
                h5.create_dataset(f"version/{key}", data=np.int32(value))


@pytest.mark.parametrize(
    "ncdij,version,is_error",
    [
        (1, (6, 6, 1), False),
        (2, (6, 6, 1), True),
        (4, (6, 6, 1), True),
        (2, (6, 5, 1), True),
        (2, None, True),
        (2, (6, 6, 2), False),
        (4, (6, 7, 0), False),
    ],
)
def test_read_local_potential_vaspouth5_version(
    tmp_path: Path, ncdij: int, version: tuple[int, int, int] | None, is_error: bool
):
    """The local potential of VASP 6.6.1 and older is not read for ncdij > 1.

    Otherwise (v, v_z) is returned as (v + v_z, v - v_z), and
    (v, v_x, v_y, v_z) as (v + v_z, v_x - i v_y, v_x + i v_y, v - v_z).

    """
    pot = np.random.default_rng(0).random((ncdij, 2, 3, 4))
    filename = tmp_path / "vaspout.h5"
    _write_vaspout_potential(filename, pot, version)
    if is_error:
        with pytest.raises(RuntimeError, match="LWAP"):
            read_local_potential_vaspouth5(filename)
        return

    if ncdij == 1:
        expected = pot
    elif ncdij == 2:
        v, vz = pot
        expected = np.array([v + vz, v - vz])
    else:
        v, vx, vy, vz = pot
        expected = np.array([v + vz, vx - 1j * vy, vx + 1j * vy, v - vz])
    np.testing.assert_allclose(read_local_potential_vaspouth5(filename), expected)


def _get_CHGCAR_by_loop(charge: np.ndarray, header: str) -> str:
    """Return the text of get_CHGCAR by the loop over the points one by one."""
    text = header + "\n%5d%5d%5d\n" % charge.shape[::-1]
    count = 0
    for z in range(charge.shape[0]):
        for y in range(charge.shape[1]):
            for x in range(charge.shape[2]):
                text += " %17.11e" % charge[z, y, x]
                count += 1
                if (count % 5) == 0:
                    count = 0
                    text += "\n"
    return text


@pytest.mark.parametrize("shape", [(2, 3, 5), (3, 2, 7), (1, 1, 4)])
def test_get_CHGCAR_equals_loop(shape: tuple[int, int, int]):
    """get_CHGCAR gives the same text as the loop over the points.

    The number of values is a multiple of five for (2, 3, 5), where the text
    ends with a newline, and not for the others.  The values have both signs
    and exponents of two and three digits.

    """
    rng = np.random.default_rng(0)
    charge = rng.standard_normal(shape) * 10.0 ** rng.integers(-120, 120, shape)
    header = "header\n1.0"
    assert get_CHGCAR(charge, header) == _get_CHGCAR_by_loop(charge, header)


def test_get_CHGCAR_text():
    """Values are written five in a line with x running fastest."""
    charge = np.arange(6, dtype="double").reshape(1, 2, 3) - 2.5
    expected = (
        "H\n    3    2    1\n"
        " -2.50000000000e+00 -1.50000000000e+00 -5.00000000000e-01"
        " 5.00000000000e-01 1.50000000000e+00\n"
        " 2.50000000000e+00"
    )
    assert get_CHGCAR(charge, "H") == expected
