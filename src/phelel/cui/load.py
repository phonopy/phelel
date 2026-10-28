"""Phelel loader."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import phonopy.cui.load_helper as phonopy_load_helper
from numpy.typing import NDArray
from phonopy import Phonopy
from phonopy.harmonic.displacement import DisplacementDataset
from phonopy.harmonic.dynamical_matrix import NacParams
from phonopy.physical_units import get_calculator_physical_units
from phonopy.structure.atoms import PhonopyAtoms

from phelel import Phelel
from phelel.file_IO import read_phelel_params_hdf5
from phelel.interface.phelel_yaml import PhelelYaml
from phelel.interface.vasp.derivatives import read_files


def load(
    phonopy_yaml: str | os.PathLike | None = None,
    supercell_matrix: Sequence[int] | Sequence[Sequence[int]] | NDArray | None = None,
    primitive_matrix: Literal["P", "F", "I", "A", "C", "R", "auto"]
    | Sequence[Sequence[float]]
    | NDArray
    | None = None,
    phonon_supercell_matrix: Sequence[Sequence[int]] | NDArray | None = None,
    is_nac: bool = True,
    calculator: str | None = None,
    fft_mesh: Sequence[int] | NDArray | None = None,
    unitcell: PhonopyAtoms | None = None,
    supercell: PhonopyAtoms | None = None,
    dir_names: Sequence[str | os.PathLike] | None = None,
    phonon_dir_names: Sequence[str | os.PathLike] | None = None,
    unitcell_filename: str | os.PathLike | None = None,
    supercell_filename: str | os.PathLike | None = None,
    force_sets_filename: str | os.PathLike | None = None,
    force_constants_filename: str | os.PathLike | None = None,
    subtract_rfs: bool = True,
    symprec: float = 1e-5,
    is_symmetry: bool = True,
    nufft: Literal["finufft", "native"] | None = None,
    log_level: int = 0,
) -> Phelel:
    """Loader function.

    phonopy_yaml : str, optional
        Filename of "phonopy.yaml"-like file. If this is given, the data
        in the file are parsed. Default is None.
    supercell_matrix : array_like, optional
        Supercell matrix multiplied to input cell basis vectors.
        shape=(3, ) or (3, 3), where the former is considered a diagonal
        matrix. Default is the unit matrix.
        dtype=int
    primitive_matrix : array_like or str, optional
        Primitive matrix multiplied to input cell basis vectors. Default is
        None, which is equivalent to 'auto'.
        For array_like, shape=(3, 3), dtype=float.
        When 'F', 'I', 'A', 'C', or 'R' is given instead of a 3x3 matrix,
        the primitive matrix for the character found at
        https://spglib.github.io/spglib/definition.html
        is used.
    phonon_supercell_matrix : array_like, optional
        Supercell matrix used for phonon calculation. Supercell matrix for
        derivatives of potentials and phonon (derivative of forces) can be
        different. Unless setting this, supercell_matrix is used as
        phonon_supercell_matrix. Default is None.
    is_nac : bool, optional
        If True, the NAC parameters are those in phonopy_yaml, or else those
        in "BORN" in the current directory if it exists. If False, NAC is
        turned off. Default is True.
    calculator : str, optional
        Name of the calculator interface, "vasp" or None. With phonopy_yaml,
        None means the calculator in the file. Default is None.
    fft_mesh : array_like, optional
        FFT mesh numbers for primitive cell used for generating local potential
        derivative interpolation grid. This has to be set to run displacement
        derivative calculations not only dV/du, but also dDij/du. Default is None.
        dtype='int64', shape=(3,)
    unitcell : PhonopyAtoms, optional
        Input unit cell. Default is None.
    supercell : PhonopyAtoms, optional
        Input supercell. With given, default value of primitive_matrix is set
        to 'auto' (can be overwitten). supercell_matrix is ignored. Default is
        None.
    dir_names : list of str, optional
        List of supercell directory names, e.g.,
            ["perfect", "disp-001", "disp-002"]
        Default is None.
    phonon_dir_names : list of str, optional
        List of supercell directory names for phonon calculation, e.g.,
            ["phonon-perfect", "phonon-disp-001", "phonon-disp-002"]
        Default is None.
    unitcell_filename : str, optional
        Input unit cell filename. Default is None.
    supercell_filename : str, optional
        Input supercell filename. When this is specified, supercell_matrix is
        ignored. Default is None.
    subtract_rfs : bool, optional
        Subtract residual forces of perfect supercell from forces of displaced
        supercells. Default is True.
    symprec : float, optional
        Symmetry tolerance used to search crystal symmetry. Default
        is 1e-5.
    is_symmetry : bool, optional
        Use crystal symmetry or not. Default is True.
    nufft : {"finufft", "native"} or None, optional
        Method to evaluate the Fourier series of the change of the local
        potential at the grid points.  See Phelel.  Default is None, which is
        "finufft".
    log_level : int, optional
        Log level. 0 is most quiet. Default is 0.

    """
    smat: Sequence[int] | Sequence[Sequence[int]] | NDArray | None
    pmat: (
        Literal["P", "F", "I", "A", "C", "R", "auto"]
        | Sequence[Sequence[float]]
        | NDArray
        | None
    )
    ph_smat: Sequence[Sequence[int]] | NDArray | None
    if phonopy_yaml is not None:
        (
            cell,
            smat,
            pmat,
            ph_smat,
            dataset,
            phonon_dataset,
            _,
            _nac_params,
            calculator,
            site_mixture_scheme,
        ) = _read_phelel_yaml(phonopy_yaml, primitive_matrix, None, is_nac, calculator)
    else:
        cell, smat, pmat = phonopy_load_helper.get_cell_settings(
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            unitcell=unitcell,
            supercell=supercell,
            unitcell_filename=unitcell_filename,
            supercell_filename=supercell_filename,
            calculator=calculator,
            symprec=symprec,
        )
        ph_smat = phonon_supercell_matrix
        dataset = None
        phonon_dataset = None
        _nac_params = None
        site_mixture_scheme = None

    assert cell is not None
    phelel = Phelel(
        cell,
        supercell_matrix=smat,
        primitive_matrix=pmat,
        phonon_supercell_matrix=ph_smat,
        fft_mesh=fft_mesh,
        symprec=symprec,
        calculator=calculator,
        is_symmetry=is_symmetry,
        nufft=nufft,
        log_level=log_level,
        site_mixture_scheme=site_mixture_scheme or "merge",
    )
    if dataset:
        phelel.dataset = dataset
    if phonon_dataset:
        phelel.phonon_dataset = phonon_dataset
    nac_params = phonopy_load_helper.get_nac_params(
        primitive=phelel.primitive,
        nac_params=_nac_params,
        is_nac=is_nac,
        nac_factor=get_calculator_physical_units(calculator).nac_factor,
        log_level=log_level,
    )
    if nac_params is not None:
        phelel.nac_params = nac_params

    try:
        dDVdu, dDijdu, _fft_mesh, fc = read_phelel_params_hdf5(log_level=log_level)
        phelel.dVdu = dDVdu
        phelel.dDijdu = dDijdu
        if fft_mesh is None:
            phelel.fft_mesh = _fft_mesh
        elif not np.array_equal(_fft_mesh, fft_mesh):
            raise RuntimeError("FFT mesh is inconsistent.")
        if fc is not None:
            phelel.force_constants = fc
    except FileNotFoundError:
        pass

    if dir_names:
        phe_input = read_files(
            phelel,
            dir_names,
            phonon_dir_names=phonon_dir_names,
            subtract_rfs=subtract_rfs,
            log_level=log_level,
        )
        if phelel.fft_mesh is not None:
            phelel.run_derivatives(phe_input)
    elif force_constants_filename is not None:
        _fc = phonopy_load_helper.select_and_extract_force_constants(
            cast(Phonopy, phelel),
            force_constants_filename=force_constants_filename,
            log_level=log_level,
        )
        if _fc is not None:
            phelel.force_constants = _fc
    elif force_sets_filename is not None:
        if phelel.phonon_supercell is None:
            raise RuntimeError("Phonon instance is not initialized.")
        phelel.phonon_dataset = phonopy_load_helper.read_force_sets(
            force_sets_filename,
            supercell=phelel.phonon_supercell,
            unmerged_supercell=phelel.phonon_unmerged_supercell,
        )

    return phelel


def _read_phelel_yaml(
    filename: str | os.PathLike,
    primitive_matrix: Literal["P", "F", "I", "A", "C", "R", "auto"]
    | Sequence[Sequence[float]]
    | NDArray
    | None,
    nac_params: NacParams | None,
    is_nac: bool,
    calculator: str | None,
) -> tuple[
    PhonopyAtoms | None,
    NDArray[np.int64] | None,
    Literal["P", "F", "I", "A", "C", "R", "auto"]
    | Sequence[Sequence[float]]
    | NDArray
    | None,
    NDArray | None,
    DisplacementDataset | None,
    DisplacementDataset | None,
    NDArray[np.double] | None,
    NacParams | None,
    str | None,
    Literal["merge", "split"] | None,
]:
    """Read properties from phelel.yaml."""
    phe_yml = PhelelYaml().read(filename)
    cell = phe_yml.unitcell
    smat = phe_yml.supercell_matrix
    ph_smat = phe_yml.phonon_supercell_matrix
    # As in phonopy.load(), None and "auto" mean the primitive matrix in the
    # file.  Without it, the primitive matrix is guessed by Phonopy.
    if primitive_matrix is None or (
        isinstance(primitive_matrix, str) and primitive_matrix == "auto"
    ):
        pmat = phe_yml.primitive_matrix
    else:
        pmat = primitive_matrix
    if nac_params is not None:
        _nac_params = nac_params
    elif is_nac:
        _nac_params = phe_yml.nac_params
    else:
        _nac_params = None
    dataset = phe_yml.dataset
    ph_dataset = phe_yml.phonon_dataset
    fc = phe_yml.force_constants
    if calculator is None:
        _calculator = phe_yml.calculator
    else:
        _calculator = calculator
    return (
        cell,
        smat,
        pmat,
        ph_smat,
        dataset,
        ph_dataset,
        fc,
        _nac_params,
        _calculator,
        phe_yml.site_mixture_scheme,
    )
