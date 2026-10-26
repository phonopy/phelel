"""Test for classes in Dij_qij.py."""

from __future__ import annotations

import json
import pathlib
import warnings
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

from phelel.api_phelel import Phelel, PhelelDataset
from phelel.base.Dij_qij import (
    DDijQij,
    DDijQijFit,
    DeltaDijQij,
    _check_Dij_symmetry,
    _rotate_Dij_in_orbital_space,
)
from phelel.exception import MagneticMomentsWarning
from phelel.utils.spherical_harmonics import LxLyLzMatrices, SHRotationMatrices

cwd = pathlib.Path(__file__).parent


def test_DeltaDijQij(
    phelel_empty_CdAs2_111: Phelel, phelel_input_CdAs2_111: PhelelDataset
):
    """Test DeltaDijQij."""
    phe_in = phelel_input_CdAs2_111
    assert phelel_empty_CdAs2_111.dataset is not None
    displacements = phelel_empty_CdAs2_111.dataset["first_atoms"]
    for i, (Dij_disp, qij_disp) in enumerate(
        zip(phe_in.Dijs[1:], phe_in.qijs[1:], strict=True)
    ):
        DeltaDijQij(
            phe_in.Dijs[0],
            Dij_disp,
            phe_in.qijs[0],
            qij_disp,
            displacements[i],
            phe_in.lm_channels,
        )


def test_DDijQijFit(
    phelel_empty_CdAs2_111: Phelel, phelel_input_CdAs2_111: PhelelDataset
):
    """Test DDijQijFit."""
    phe = phelel_empty_CdAs2_111
    phe_in = phelel_input_CdAs2_111
    assert phe.dataset is not None
    displacements = phe.dataset["first_atoms"]

    delta_Dij_qijs = []
    for i, (Dij_disp, qij_disp) in enumerate(
        zip(phe_in.Dijs[1:], phe_in.qijs[1:], strict=True)
    ):
        delta_Dij_qijs.append(
            DeltaDijQij(
                phe_in.Dijs[0],
                Dij_disp,
                phe_in.qijs[0],
                qij_disp,
                displacements[i],
                phe_in.lm_channels,
            )
        )
    ddijqij = DDijQijFit(
        phe.supercell,
        phe.symmetry,
        atom_indices=phe.primitive.p2s_map,
    )
    ddijqij.run(delta_Dij_qijs)


def test_DDijQij(phelel_empty_CdAs2_111: Phelel, phelel_input_CdAs2_111: PhelelDataset):
    """Test DDijQij."""
    phe = phelel_empty_CdAs2_111
    phe_in = phelel_input_CdAs2_111
    assert phe.dataset is not None
    displacements = phe.dataset["first_atoms"]
    dDijdu = DDijQij(phe.supercell, phe.symmetry, phe.primitive.p2s_map)
    dDijdu.run(
        phe_in.Dijs[0],
        phe_in.Dijs[1:],
        phe_in.qijs[0],
        phe_in.qijs[1:],
        displacements,
        phe_in.lm_channels,
    )


def test_DDijQijFit_with_atom_indices_in_any_order(
    phelel_empty_CdAs2_111: Phelel, phelel_input_CdAs2_111: PhelelDataset
):
    """dDij/du of each atom does not depend on the order of atom_indices.

    The site-symmetry operations are collected for the equivalent atoms in
    ascending order, and they have to be matched to atom_indices given in
    another order.  Atoms 1 and 2 of CdAs2 are equivalent to the displaced
    atom 1.

    """
    phe = phelel_empty_CdAs2_111
    phe_in = phelel_input_CdAs2_111
    assert phe.dataset is not None
    displacements = phe.dataset["first_atoms"]
    delta_Dij_qijs = [
        DeltaDijQij(
            phe_in.Dijs[0],
            phe_in.Dijs[i + 1],
            phe_in.qijs[0],
            phe_in.qijs[i + 1],
            disp,
            phe_in.lm_channels,
        )
        for i, disp in enumerate(displacements)
        if disp["number"] == 0
    ]
    dDijdu = {}
    for atom_indices in ([0, 1], [1, 0]):
        ddijqij = DDijQijFit(
            phe.supercell,
            phe.symmetry,
            atom_indices=atom_indices,
            verbose=False,
        )
        ddijqij.run(delta_Dij_qijs)
        np.testing.assert_array_equal(ddijqij.atom_indices, atom_indices)
        dDijdu[tuple(atom_indices)] = ddijqij.dDijdu
    assert np.abs(dDijdu[(0, 1)]).max() > 0.1
    np.testing.assert_allclose(dDijdu[(1, 0)], dDijdu[(0, 1)][:, ::-1], atol=1e-12)


def test_DDijQij_with_undisplaced_atoms(
    phelel_empty_CdAs2_111: Phelel, phelel_input_CdAs2_111: PhelelDataset
):
    """Atoms not equivalent to any displaced atom are an error.

    In CdAs2, the displacements of Cd atom 1 give no dDij/du of the As atoms,
    which would be left zero.  The atoms of the derivatives are those of the
    primitive cell, 1, 3, 5, 7, 9 and 11, of which 5, 7, 9 and 11 are As.

    """
    phe = phelel_empty_CdAs2_111
    phe_in = phelel_input_CdAs2_111
    assert phe.dataset is not None
    displacements = phe.dataset["first_atoms"][:2]
    assert [disp["number"] for disp in displacements] == [0, 0]
    np.testing.assert_array_equal(phe.primitive.p2s_map, [0, 2, 4, 6, 8, 10])
    dDijdu = DDijQij(phe.supercell, phe.symmetry, phe.primitive.p2s_map)
    with pytest.raises(ValueError, match="atoms 5, 7, 9, 11 of the supercell"):
        dDijdu.run(
            phe_in.Dijs[0],
            phe_in.Dijs[1:3],
            phe_in.qijs[0],
            phe_in.qijs[1:3],
            displacements,
            phe_in.lm_channels,
        )


def _load_Dij_perfect(
    name: str, magmoms: str | Sequence[float] | Sequence[Sequence[float]] | None
) -> tuple[NDArray, list[dict], PhonopyAtoms]:
    """Return Dij, lm channels and the supercell of Dij_perfect/name.npz.

    magmoms="vasp" gives MAGMOM of the VASP calculation.

    """
    data = np.load(cwd / "Dij_perfect" / f"{name}.npz")
    cell = PhonopyAtoms(
        cell=data["lattice"],
        scaled_positions=data["scaled_positions"],
        symbols=[str(symbol) for symbol in data["symbols"]],
        magnetic_moments=data["magnetic_moments"] if magmoms == "vasp" else magmoms,
    )
    return data["dij"], json.loads(str(data["lm_channels"])), cell


@pytest.mark.parametrize(
    "name,magmoms,is_changed",
    [
        ("Fe", "vasp", False),
        ("Fe", None, False),
        ("Fe", [3, -3], True),
        ("Fe", [0, 0], True),
        ("Cr", "vasp", False),
        ("Cr", None, True),
        ("Cr", [2, 2], True),
        ("Cr", [0, 0], True),
        ("Fe-ncl", "vasp", False),
        ("Fe-ncl", None, True),
        ("Fe-ncl", [[0, 0, 3], [0, 0, 3]], True),
        ("Fe-ncl", [[3, 3, 3], [-3, -3, -3]], True),
        ("C-ncl", "vasp", False),
        ("C-ncl", None, False),
    ],
)
def test_check_Dij_symmetry(
    name: str,
    magmoms: str | Sequence[float] | Sequence[Sequence[float]] | None,
    is_changed: bool,
):
    """Dij of the perfect supercell is unchanged by fitting symmetry operations.

    Fe is ferromagnetic and collinear, so the symmetry operations without
    magnetic moments also fit it. Cr is antiferromagnetic and collinear, and
    half of its symmetry operations are combined with time reversal. Fe-ncl
    has the magnetic moments along (1, 1, 1). C-ncl is non-magnetic.

    """
    Dij, lm_channels, cell = _load_Dij_perfect(name, magmoms)
    symmetry = Symmetry(cell)
    if is_changed:
        with pytest.warns(MagneticMomentsWarning, match="PAW strengths Dij"):
            _check_Dij_symmetry(Dij, lm_channels, cell, symmetry)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MagneticMomentsWarning)
            _check_Dij_symmetry(Dij, lm_channels, cell, symmetry)


def test_check_Dij_symmetry_without_time_reversal(monkeypatch: pytest.MonkeyPatch):
    """Dij of antiferromagnetic Cr is changed without time reversal.

    The operations that move an atom onto the other atom are combined with
    time reversal, which exchanges spin up and spin down. Without it, these
    operations change Dij.

    """
    Dij, lm_channels, cell = _load_Dij_perfect("Cr", "vasp")
    symmetry = Symmetry(cell)
    n_ops = len(symmetry.symmetry_operations["rotations"])
    monkeypatch.setattr(
        "phelel.base.Dij_qij.get_time_reversals",
        lambda symmetry: np.zeros(n_ops, dtype=bool),
    )
    with pytest.warns(MagneticMomentsWarning, match="PAW strengths Dij"):
        _check_Dij_symmetry(Dij, lm_channels, cell, symmetry)


def test_DDijQij_checks_Dij_symmetry():
    """DDijQij.run warns when the symmetry operations do not fit Dij."""
    Dij, lm_channels, cell = _load_Dij_perfect("Fe", [3, -3])
    qij = np.zeros_like(Dij)
    displacements = [{"number": 0, "displacement": [0.01, 0, 0]}]
    dDijdu = DDijQij(cell, Symmetry(cell), verbose=False)
    with pytest.warns(MagneticMomentsWarning, match="PAW strengths Dij"):
        dDijdu.run(Dij, [Dij], qij, [qij], displacements, lm_channels)


def test_rotate_Dij_in_orbital_space_direction():
    """Dij is transformed by the operation, not by its inverse.

    Four atoms on a square are related by the four-fold rotation R about z,
    which moves x to y. Atom 0 has Dij = |p_x><p_z| + |p_z><p_x|, and the other
    atoms have zero. The rotated Dij is |p_y><p_z| + |p_z><p_y| at the atom
    that atom 0 is moved to. With the inverse rotation, the sign would be
    reversed, and with the inverse permutation, it would be at another atom.

    """
    cell = PhonopyAtoms(
        symbols=["Si"] * 4 + ["Ge"],
        cell=np.diag([5.0, 5.0, 7.0]),
        scaled_positions=[
            [0.3, 0.1, 0],
            [0.9, 0.3, 0],
            [0.7, 0.9, 0],
            [0.1, 0.7, 0],
            [0.5, 0.5, 0.3],
        ],
    )
    symmetry = Symmetry(cell)
    rotations = symmetry.symmetry_operations["rotations"]
    translations = symmetry.symmetry_operations["translations"]
    c4z = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    (i_op,) = [
        i
        for i, (r, t) in enumerate(zip(rotations, translations, strict=True))
        if np.array_equal(r, c4z) and np.allclose(t, 0)
    ]
    perm = symmetry.atomic_permutations[i_op]
    assert perm[0] == 1

    shr = SHRotationMatrices(rotations[i_op], cell.cell.T, LxLyLzMatrices().run())
    shr.run()
    assert shr.Delta is not None

    # Real spherical harmonics of l=1 in the order m=-1, 0, 1, i.e., y, z, x.
    y, z, x = 0, 1, 2
    lm_channels = [{"channels": [{"l": 1, "m": [-1, 0, 1]}]}] * len(cell)
    Dij = np.zeros((len(cell), 3, 3), dtype=complex)
    Dij[0, x, z] = Dij[0, z, x] = 1
    expected = np.zeros_like(Dij)
    expected[1, y, z] = expected[1, z, y] = 1

    rot_Dij = _rotate_Dij_in_orbital_space(Dij, lm_channels, shr.Delta, perm)
    np.testing.assert_allclose(rot_Dij, expected, atol=1e-12)
