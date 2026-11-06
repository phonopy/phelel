"""Test for Phelel class."""

import itertools
import pathlib
import sys

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray
from phonopy.file_IO import write_FORCE_SETS
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture

import phelel
from phelel import Phelel
from phelel.file_IO import _get_smallest_vectors, read_phelel_params_hdf5
from phelel.utils.data import cmplx2real

cwd = pathlib.Path(__file__).parent


def test_api_phelel_C111(phelel_C111: Phelel):
    """Test by diamond conv. unit cell 1x1x1."""
    filename = cwd / "phelel_params_C111.hdf5"
    _compare(filename, phelel_C111)


def test_api_phelel_NaCl111(phelel_NaCl111: Phelel):
    """Test by NaCl conv. unit cell 1x1x1."""
    filename = cwd / "phelel_params_NaCl111.hdf5"
    _compare(filename, phelel_NaCl111)


def test_api_phelel_CdAs2_111(phelel_CdAs2_111: Phelel):
    """Test by CdAs2 conv. unit cell 1x1x1 (I-centred tetragonal)."""
    filename = cwd / "phelel_params_CdAs2_111.hdf5"
    _compare(filename, phelel_CdAs2_111)


@pytest.mark.parametrize(
    "name,yaml_filename",
    [
        ("C111", "phelel_disp_C111.yaml"),
        ("NaCl111", "phelel_disp_NaCl111.yaml"),
        ("CdAs2_111", "phelel_disp_CdAs2.yaml"),
    ],
)
def test_api_phelel_default_dv_evaluation_method(
    name: str, yaml_filename: str, request: pytest.FixtureRequest
):
    """The default gives real dVdu with the real part of the reference.

    The default is dv_evaluation_method="shortest".  The reference in
    phelel_params_*.hdf5 was computed with finufft, whose box of reciprocal
    lattice vectors gives an imaginary part to dVdu at the grid points of
    fft_mesh=[14, 14, 14], which are not on the FFT mesh of the supercell.
    "shortest" places the coefficients at the shortest vectors, which keeps
    dVdu real.  The real parts differ by the choice of the
    vectors, by up to 2.5e-6 relative to the largest value for these data.
    dDijdu does not depend on dv_evaluation_method.

    """
    phe = phelel.load(cwd / yaml_filename)
    phe.fft_mesh = [14, 14, 14]
    phe.run_derivatives(request.getfixturevalue(f"phelel_input_{name}"))
    assert phe.dVdu is not None
    with h5py.File(cwd / f"phelel_params_{name}.hdf5", "r") as f:
        dVdu_ref = f["dVdu"][:]
    scale = np.abs(dVdu_ref).max()
    dVdu = phe.dVdu.dVdu
    np.testing.assert_allclose(dVdu.real, dVdu_ref[..., 0], rtol=0, atol=1e-5 * scale)
    assert np.abs(dVdu.imag).max() < 1e-12 * scale


def test_read_phelel_params_hdf5(phelel_CdAs2_111: Phelel):
    """The arrays read from phelel_params.hdf5 are those computed, as complex.

    The arrays are written as pairs of real numbers, which have to be read back
    as complex numbers of the shapes of the computed arrays.

    """
    filename = cwd / "phelel_params_CdAs2_111.hdf5"
    dVdu, dDijdu, _, _ = read_phelel_params_hdf5(filename=filename)
    phe_ref = phelel_CdAs2_111

    np.testing.assert_allclose(dVdu.dVdu, phe_ref.dVdu.dVdu, rtol=1e-5, atol=1e-5)
    for name in ("dDijdu", "dqijdu", "Dij", "qij"):
        value, ref = getattr(dDijdu, name), getattr(phe_ref.dDijdu, name)
        assert value.dtype == np.complex128
        assert value.shape == ref.shape, name
        np.testing.assert_allclose(value, ref, rtol=1e-5, atol=1e-5)


def _compare(filename: pathlib.Path, phe: Phelel):
    """Assert results.

    shortest_vectors and shortest_vector_multiplicities are included at later
    versions of Phelel. So, these are not included in old reference files.

    """
    with h5py.File(filename, "r") as f:
        dVdu_ref = f["dVdu"][:]
        dDijdu_ref = f["dDijdu"][:]

        dVdu = cmplx2real(phe.dVdu.dVdu)
        dDijdu = cmplx2real(phe.dDijdu.dDijdu)
        np.testing.assert_allclose(dVdu, dVdu_ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(dDijdu, dDijdu_ref, rtol=1e-4, atol=1e-4)

        if "shortest_vectors" in f:
            shortest_vectors_ref = f["shortest_vectors"][:]
            multiplicities_ref = f["shortest_vector_multiplicities"][:]

            shortest_vectors, multiplicities = _get_smallest_vectors(phe.primitive)
            np.testing.assert_array_equal(
                shortest_vectors.shape, shortest_vectors_ref.shape
            )
            np.testing.assert_array_equal(multiplicities, multiplicities_ref)


def _get_GeSnTe_weighted_cell() -> PhonopyAtoms:
    """Return CsCl-like cell with Ge and Sn (0.5 each) co-located and Te."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        symbols=["Ge", "Sn", "Te"],
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
    )
    return apply_site_mixture(cell, [0.5, 0.5, 1.0])


def test_phelel_merge_scheme_yaml(tmp_path: pathlib.Path):
    """With the merge scheme, phelel_disp.yaml has the unmerged cells."""
    phe = Phelel(
        _get_GeSnTe_weighted_cell(),
        supercell_matrix=np.diag([2, 2, 2]),
        phonon_supercell_matrix=np.diag([2, 2, 2]),
        primitive_matrix="P",
    )
    assert phe.site_mixture_scheme == "merge"
    assert phe.unitcell.symbols == ["GeSn", "Te"]
    assert phe.unmerged_unitcell is not None
    assert phe.unmerged_unitcell.symbols == ["Ge", "Sn", "Te"]
    assert phe.phonon_supercell is not None
    assert phe.phonon_unmerged_supercell is not None
    assert len(phe.phonon_unmerged_supercell) == 24
    assert len(phe.phonon_supercell) == 16
    phe.generate_displacements()
    phe.generate_phonon_displacements()

    filename = tmp_path / "phelel_disp.yaml"
    text = str(phe.to_phelel_yaml())
    assert "site_mixture_scheme: merge" in text
    assert "mixture:" not in text
    filename.write_text(text)
    phe2 = phelel.load(filename, log_level=0)
    assert phe2.site_mixture_scheme == "merge"
    assert phe2.unitcell.symbols == ["GeSn", "Te"]
    assert phe2.phonon_dataset is not None
    assert phe2.phonon_dataset["natom"] == 16
    with pytest.raises(RuntimeError, match="merge scheme of site mixture"):
        phe2.run_derivatives(None)  # type: ignore[arg-type]


def test_phelel_load_force_sets_of_phonon_supercell(tmp_path: pathlib.Path):
    """FORCE_SETS is read for the phonon supercell."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        symbols=["Na", "Cl"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    force_sets = tmp_path / "FORCE_SETS"
    write_FORCE_SETS(
        {
            "natom": 16,
            "first_atoms": [
                {
                    "number": 0,
                    "displacement": np.array([0.01, 0, 0]),
                    "forces": np.zeros((16, 3)),
                }
            ],
        },
        filename=force_sets,
    )
    phe = phelel.load(
        unitcell=cell,
        supercell_matrix=np.eye(3, dtype=int),
        phonon_supercell_matrix=np.diag([2, 2, 2]),
        primitive_matrix="P",
        force_sets_filename=force_sets,
    )
    assert phe.phonon_dataset is not None
    assert phe.phonon_dataset["natom"] == 16


@pytest.mark.parametrize("with_phonon_supercell", [True, False])
def test_phelel_unit_conversion_factor_and_dynamical_matrix(
    with_phonon_supercell: bool,
):
    """Both come from the Phonopy instance holding the force constants.

    Without the phonon supercell, the force constants are those of the
    supercell, and with it, those of the phonon supercell.

    """
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        symbols=["Na", "Cl"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    if with_phonon_supercell:
        phonon_supercell_matrix = np.diag([2, 2, 2])
        natom = 16
    else:
        phonon_supercell_matrix = None
        natom = 2
    phe = Phelel(
        cell,
        supercell_matrix=np.eye(3, dtype=int),
        phonon_supercell_matrix=phonon_supercell_matrix,
        primitive_matrix="P",
    )
    assert phe.unit_conversion_factor is not None
    phe.force_constants = np.zeros((natom, natom, 3, 3))
    assert phe.dynamical_matrix is not None
    assert len(phe.dynamical_matrix.supercell) == natom


@pytest.mark.parametrize(
    "dv_evaluation_method,is_error",
    [
        (None, False),
        pytest.param("finufft", False, marks=pytest.mark.finufft),
        ("shortest", False),
        ("box", False),
        ("nfft", True),
    ],
)
def test_phelel_dv_evaluation_method(dv_evaluation_method: str | None, is_error: bool):
    """Only "shortest", "box", "finufft" and None are accepted."""
    unitcell = phelel.load(cwd / "phelel_disp_C111.yaml").unitcell

    def _phelel():
        return Phelel(
            unitcell,
            supercell_matrix=np.eye(3, dtype=int),
            primitive_matrix="P",
            dv_evaluation_method=dv_evaluation_method,
        )

    if is_error:
        with pytest.raises(ValueError, match='not "nfft"'):
            _phelel()
    else:
        _phelel()


@pytest.mark.parametrize(
    "dv_evaluation_method,is_error",
    [
        (None, True),
        pytest.param("finufft", False, marks=pytest.mark.finufft),
        ("shortest", True),
        ("box", True),
    ],
)
def test_phelel_finufft_eps_with_dv_evaluation_method(
    dv_evaluation_method: str | None, is_error: bool
):
    """finufft_eps is accepted only with dv_evaluation_method="finufft".

    dv_evaluation_method=None is "shortest".

    """
    unitcell = phelel.load(cwd / "phelel_disp_C111.yaml").unitcell

    def _phelel():
        return Phelel(
            unitcell,
            supercell_matrix=np.eye(3, dtype=int),
            primitive_matrix="P",
            dv_evaluation_method=dv_evaluation_method,
            finufft_eps=1e-8,
        )

    if is_error:
        with pytest.raises(ValueError, match="finufft_eps"):
            _phelel()
    else:
        _phelel()


def test_phelel_dv_evaluation_method_finufft_not_installed(
    monkeypatch: pytest.MonkeyPatch,
):
    """Method "finufft" without finufft stops with ImportError to install it.

    finufft is not a dependency of phelel.  It is hidden by None in sys.modules,
    which makes "import finufft" raise ImportError.

    """
    unitcell = phelel.load(cwd / "phelel_disp_C111.yaml").unitcell
    monkeypatch.setitem(sys.modules, "finufft", None)
    with pytest.raises(ImportError, match="Install the finufft package"):
        Phelel(
            unitcell,
            supercell_matrix=np.eye(3, dtype=int),
            primitive_matrix="P",
            dv_evaluation_method="finufft",
        )


def test_force_constants_keep_magnetic_symmetry():
    """The symfc projector does not raise the symmetry lowered by magnetic moments.

    The spring constants depend on the product of the magnetic moments, so the
    force constants have only the symmetry of the operations of the magnetic
    space group, whose time reversal is not applied.

    """
    magnetic_moments = np.array([1, -1, -1, 1, 0, 0, 0, 0], dtype=float)
    unitcell = _get_NaCl_conventional_cell()
    spring_force_constants = _get_spring_force_constants(
        unitcell, 1 + 0.5 * np.outer(magnetic_moments, magnetic_moments)
    )
    unitcell.magnetic_moments = magnetic_moments
    phe = _run_force_constants(unitcell, spring_force_constants)
    assert len(phe.symmetry.symmetry_operations["rotations"]) == 64
    np.testing.assert_allclose(
        phe.force_constants, spring_force_constants, rtol=0, atol=1e-10
    )


def test_force_constants_without_symmetry():
    """With is_symmetry=False, the symfc projector uses no crystal symmetry.

    The force constants from random spring constants have no symmetry of the
    crystal.  They have only the translational and permutation symmetries, which
    the symfc projector imposes.

    """
    rng = np.random.default_rng(0)
    spring_constants = rng.uniform(0.5, 1.5, (8, 8))
    unitcell = _get_NaCl_conventional_cell()
    spring_force_constants = _get_spring_force_constants(
        unitcell, (spring_constants + spring_constants.T) / 2
    )
    phe = _run_force_constants(unitcell, spring_force_constants, is_symmetry=False)
    assert len(phe.symmetry.symmetry_operations["rotations"]) == 1
    np.testing.assert_allclose(
        phe.force_constants, spring_force_constants, rtol=0, atol=1e-10
    )


def test_force_constants_with_symprec_larger_than_distortion():
    """The symfc projector works with the symmetry found with symprec.

    An atom is off its site by more than the default symprec of symfc, and by
    less than the symprec given here.

    """
    spring_force_constants = _get_spring_force_constants(
        _get_NaCl_conventional_cell(), np.ones((8, 8))
    )
    unitcell = _get_NaCl_conventional_cell()
    positions = unitcell.scaled_positions
    positions[0, 0] += 5e-4
    unitcell.scaled_positions = positions
    phe = _run_force_constants(unitcell, spring_force_constants, symprec=1e-2)
    assert len(phe.symmetry.symmetry_operations["rotations"]) == 192
    np.testing.assert_allclose(
        phe.force_constants, spring_force_constants, rtol=0, atol=1e-10
    )


def _get_NaCl_conventional_cell() -> PhonopyAtoms:
    """Return NaCl conventional unit cell."""
    return PhonopyAtoms(
        symbols=["Na"] * 4 + ["Cl"] * 4,
        cell=np.eye(3) * 5.6,
        scaled_positions=[
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0.5],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
        ],
    )


def _get_spring_force_constants(
    unitcell: PhonopyAtoms, spring_constants: NDArray[np.double]
) -> NDArray[np.double]:
    """Return force constants of springs between atoms up to second neighbors.

    The unit cell is taken as the supercell.  spring_constants has to be
    symmetric.  The force constants have the translational and permutation
    symmetries.

    """
    natom = len(unitcell)
    force_constants = np.zeros((natom, natom, 3, 3))
    for i, j in itertools.permutations(range(natom), 2):
        for lattice_point in itertools.product((-1, 0, 1), repeat=3):
            scaled_vector = (
                unitcell.scaled_positions[j]
                - unitcell.scaled_positions[i]
                + lattice_point
            )
            vector = scaled_vector @ unitcell.cell
            distance = np.linalg.norm(vector)
            if distance < 4.5:
                force_constants[i, j] -= (
                    spring_constants[i, j] * np.outer(vector, vector) / distance**2
                )
    for i in range(natom):
        force_constants[i, i] = -force_constants[i].sum(axis=0)
    return force_constants


def _run_force_constants(
    unitcell: PhonopyAtoms,
    spring_force_constants: NDArray[np.double],
    symprec: float = 1e-5,
    is_symmetry: bool = True,
) -> Phelel:
    """Return Phelel with force constants from forces of the spring model."""
    phe = Phelel(
        unitcell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix=np.eye(3),
        fft_mesh=[6, 6, 6],
        symprec=symprec,
        is_symmetry=is_symmetry,
    )
    phe.generate_displacements(distance=0.01)
    forces = [
        -spring_force_constants[:, first_atom["number"]] @ first_atom["displacement"]
        for first_atom in phe._get_first_atoms()
    ]
    phe._prepare_phonon(forces=np.array(forces), show_drift=False)
    return phe
