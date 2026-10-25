"""Tests for the readers of VASP output files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pytest

from phelel.interface.vasp.file_IO import (
    read_inwap_yaml,
    read_local_potential_vaspouth5,
    read_PAW_Dij_qij,
    read_PAW_Dij_qij_vaspouth5,
)

cwd = Path(__file__).parent


@pytest.mark.parametrize("name", ["Cr", "Fe-ncl"])
def test_read_PAW_Dij_qij_two_formats(name: str):
    """Dij and qij of the old format are equal to those of vaspout.h5.

    Both are written by one VASP calculation with LWAP = .TRUE.  Dij of Cr
    differs between the spin components and between the atoms, and Dij of
    Fe-ncl has imaginary parts, so mixing up these axes or the real and
    imaginary parts in one of the readers changes Dij.  The order of the two lm
    indices is not tested, because Dij and qij are symmetric in them.

    """
    data_dir = cwd / "Dij_qij_two_formats" / name
    inwap = read_inwap_yaml(data_dir / "inwap.yaml")
    dij, qij = read_PAW_Dij_qij_vaspouth5(data_dir / "vaspout-paw.h5")
    if name == "Cr":
        assert np.abs(dij[0] - dij[1]).max() > 0.1
        assert np.abs(dij[:, 0] - dij[:, 1]).max() > 0.1
    else:
        assert np.abs(dij.imag).max() > 0.1

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
