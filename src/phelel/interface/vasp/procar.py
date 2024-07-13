"""Generate VASP PROCAR equivalent information."""

import numpy as np


class Projector:
    """Wave function character class."""

    def __init__(self, projectors, lm_orbitals):
        """Init method.

        Parameters
        ----------
        projectors : ndarray
            Wave function charactor <p|Psi>
            dtype=complex128
            shape=(nkpts, nbtot, ispin, nrspinors, nions, lmdim, 4)
            The last index corresponds to usual projector and its derivatives
            along x, y, z.
            Note that exp(ikR) is not multiplied.
        lm_orbitals: dict
            Numbers of orbitals in lm channels of atoms. For example of Si::

               lm_orbitals:
               - atom_index:       1
                 num_l_channels:   4
                 channels:
                 - l :   0
                   m : [   0 ]
                 - l :   0
                   m : [   0 ]
                 - l :   1
                   m : [  -1,   0,   1 ]
                 - l :   1
                   m : [  -1,   0,   1 ]
               - atom_index:       2
                 num_l_channels:   4
                 channels:
                 - l :   0
                   m : [   0 ]
                 - l :   0
                   m : [   0 ]
                 - l :   1
                   m : [  -1,   0,   1 ]
                 - l :   1
                   m : [  -1,   0,   1 ]

        """
        self._proj = projectors
        self._lm_orbitals = lm_orbitals

    def _get_lm_pairs(self, i_atom):
        lm_pairs = []
        lm_to_l = []
        for i, ll in enumerate(self._lm_orbitals[i_atom]["channels"]):
            for m in ll["m"]:
                lm_to_l.append(i)
                lm_pairs.append((ll["l"], m))
        return lm_to_l, lm_pairs

    def _get_channel_pairs(self, lm_pairs):
        channel_pairs = []
        channels = []
        for i, pair_i in enumerate(lm_pairs):
            for j, pair_j in enumerate(lm_pairs):
                if pair_i == pair_j:
                    channel_pairs.append((i, j))
                    channels.append(pair_i)
        return channels, channel_pairs


class CoreCharge:
    """Class to compute core charges.

    Full charge density is given by <phi_k|phi_l> = <~phi_k|S|~phi_l>.
    <~psi_k|S|~psi_l> = <~psi_k|~psi_l> + <~psi_k|p_i> q_ij < p_j|~psi_l>

    This class computes the second term on rhs.

    """

    def __init__(self, projectors, qij):
        """Init method.

        Parameters
        ----------
        projectors : ndarray
            Wave function characters
            dtype=complex128
            shape=(nkpts, nbtot, ispin, nrspinors, nions, lmdim, 4)
        qij : ndarray
            <phi_i|phi_j> - <phi_i~|phi_j~>
            dtype=complex128
            shape=(ncdij, nions, lmdim, lmdim')

        """
        self._proj = projectors
        self._qij = qij

        self._cc = None
        self._run()

    def __str__(self):
        """Return summary of core charge data as text."""
        text = "Core charges\n"
        text += "(nkpts, nbtot) = "
        text += str(self._cc.shape) + "\n"
        for i_k, i_b in np.ndindex(self._cc.shape):
            text += "Sum over left-over charges of cores at "
            text += "k-point %d, band %d: %f\n" % (
                i_k + 1,
                i_b + 1,
                self._cc[i_k, i_b].real,
            )
        return text

    def _run(self):
        proj = self._proj[:, :, :, :, :, :, 0]
        qij = self._qij
        self._cc = np.zeros(proj.shape[0:2], dtype=proj.dtype)
        for i_i in range(proj.shape[4]):
            for i_s in range(proj.shape[2]):
                # Here assume not using non-collinear
                #
                # ncdij = i_s + i_n + j_n*2 (two loops for non-collinear)
                # In fortran
                #   i_s: 1 or 2
                #   i_n: 0 or 1
                #   i_n + j_n * 2: 0, 1, 2, or 3
                #   i_s i_n +j_n * 2: 1, 2, 3, or 4
                for lm, lmp in np.ndindex(qij.shape[2:]):
                    self._cc[:, :] += (
                        proj[:, :, i_s, 0, i_i, lm]
                        * qij[i_s, i_i, lm, lmp]
                        * proj[:, :, i_s, 0, i_i, lmp].conj()
                    )


class Procar(Projector):
    """Class to reproduce VASP PROCAR values."""

    def __init__(self, projectors, lm_orbitals, qtot):
        """Init method.

        Parameters
        ----------
        qtot : ndarray
            <phi_i|phi_j> - <phi~_i|phi~_j>
            dtype=complex128
            shape=(nions, ldim, ldim')

        """
        super(Procar, self).__init__(projectors, lm_orbitals)
        self._qtot = qtot
        self._run()

    def __str__(self):
        """Return array shape of wave function character as text."""
        text = "Wave function character\n"
        text += "(nkpts, nbtot, ispin, nrspinors, nions, lmdim, 4) = "
        text += str(self._proj.shape)
        text += "\n" + "\n".join(self._get_procar_lines())
        return text

    @property
    def procar(self):
        """Return PROCAR value as ndarray.

        The array shape can be seen by __str__.

        """
        return self._procar.real

    def _run(self):
        proj = self._proj[:, :, :, :, :, :, 0]
        procar = np.zeros(
            (proj.shape[2], proj.shape[0], proj.shape[1], proj.shape[4], 9),
            dtype="complex128",
        )

        for i_atom in range(self._proj.shape[4]):
            lm_to_l, lm_pairs = self._get_lm_pairs(i_atom)
            channels, channel_pairs = self._get_channel_pairs(lm_pairs)
            for i_k, i_b, i_s, i_n in np.ndindex(self._proj.shape[0:4]):
                procar[i_s, i_k, i_b, i_atom] += self._get_l_overlap(
                    proj[i_k, i_b, i_s, i_n, i_atom],
                    lm_to_l,
                    channels,
                    channel_pairs,
                    i_atom,
                )
        self._procar = procar

    def _get_procar_lines(self):
        lines = []
        for i_s in range(self._proj.shape[2]):
            for i_k in range(self._proj.shape[0]):
                for i_b in range(self._proj.shape[1]):
                    lines.append("k-point %d, band %d" % (i_k + 1, i_b + 1))
                    for i_atom in range(self._proj.shape[4]):
                        lines.append(
                            self._get_procar_text(
                                i_atom, self._procar[i_s, i_k, i_b, :]
                            )
                        )
                lines.append("")
        return lines

    def _get_procar_text(self, i_atom, procar):
        text = "%3d " % (i_atom + 1)
        text += " ".join(["%6.3f" % x for x in procar[i_atom].real])
        return text

    def _get_l_overlap(self, proj_lm, lm_to_l, channels, channel_pairs, i_atom):
        """Compute <wf-char_m>^* qtot_mm' <wf-char_m'>."""
        p_sum = np.zeros(9, dtype=proj_lm.dtype)
        for lm, c_pair in zip(channels, channel_pairs):
            # print(str(c_pair) + " " + str(lm))
            ll = lm_to_l[c_pair[0]]
            lp = lm_to_l[c_pair[1]]
            qtot = self._qtot[i_atom][ll][lp]
            lm_index = lm[0] ** 2 + lm[1] + lm[0]
            p_sum[lm_index] += proj_lm[c_pair[0]] * qtot * np.conj(proj_lm[c_pair[1]])
        return p_sum


class QTOT:
    """Sum over l channels of <phi_i|phi_j> - <phi~_i|phi~_j>."""

    def __init__(self, qtot):
        """Init method.

        These data are in PAW datasets that are assuemed frozen.

        Parameters
        ----------
        qtot : ndarray
            sum over lm channles of the same l channel of <phi_i|phi_j>
            dtype=complex128
            shape=(nions, ldim, ldim')

        """
        self._qtot = qtot

    def __str__(self):
        """Return q_tot as text."""
        text = "q_tot (nions, ldim, ldim') = "
        text += str(self._qtot.shape) + "\n"
        for i in range(self._qtot.shape[0]):
            text += "Atom %d/%d (%d, %d)\n" % ((i + 1,) + self._qtot.shape)
            for qtot_l in np.transpose(self._qtot[i]):
                text += " ".join(["%10.5f" % x for x in qtot_l])
                text += "\n"
        return text
