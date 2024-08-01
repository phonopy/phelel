"""Test for classes in Dij_qij.py."""

import pathlib

from phelel.api_phelel import Phelel, PhelelDataset
from phelel.base.Dij_qij import DDijQij, DDijQijFit, DeltaDijQij

cwd = pathlib.Path(__file__).parent


def test_DeltaDijQij(
    phelel_empty_CdAs2_111: Phelel, phelel_input_CdAs2_111: PhelelDataset
):
    """Test DeltaDijQij."""
    phe_in = phelel_input_CdAs2_111
    displacements = phelel_empty_CdAs2_111.dataset["first_atoms"]
    for i, (Dij_disp, qij_disp) in enumerate(zip(phe_in.Dijs[1:], phe_in.qijs[1:])):
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
    displacements = phe.dataset["first_atoms"]

    delta_Dij_qijs = []
    for i, (Dij_disp, qij_disp) in enumerate(zip(phe_in.Dijs[1:], phe_in.qijs[1:])):
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
        delta_Dij_qijs, phe.supercell, phe.symmetry, phe.atom_indices_in_derivatives
    )
    ddijqij.run()


def test_DDijQij(phelel_empty_CdAs2_111: Phelel, phelel_input_CdAs2_111: PhelelDataset):
    """Test DDijQij."""
    phe = phelel_empty_CdAs2_111
    phe_in = phelel_input_CdAs2_111
    displacements = phe.dataset["first_atoms"]
    dDijdu = DDijQij(phe.supercell, phe.symmetry, phe.atom_indices_in_derivatives)
    dDijdu.run(
        phe_in.Dijs[0],
        phe_in.Dijs[1:],
        phe_in.qijs[0],
        phe_in.qijs[1:],
        displacements,
        phe_in.lm_channels,
    )
