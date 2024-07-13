"""Routines to handle spherical harmonics rotation.

Old implementation`SHRotationMatricesEularAngle` relies on `EulerAngles`,
`LyMatrices`, `WignerMatricesEularAngle`. These are currently unused.

The current implementation `SHRotationMatrice`s depends on `LxLyLzMatrices` and
get_n_and_rotation_order.

Both implementations call `make_C`.

Agreement of these two implementations are checked in the test. The new
implementation is expected to be numerically more robust.

"""

from __future__ import annotations

from typing import Optional

import numpy as np
from phonopy.structure.cells import determinant


class EulerAngles:
    """Calculate Euler angles.

    Implements Z1Y2Z3 given in Euler angles page of wikipedia.

    """

    def __init__(self, R: np.ndarray):
        """Init method.

        Proper rotation of R is transformed to Eular angles.

        Parameters
        ----------
        R(np_double): Rotation matrix in Cartesian coordinates

        """
        self.R = np.array(R, dtype="double", order="C")
        self._angles = None

    @property
    def angles(self) -> Optional[np.ndarray]:
        """Return Eular angles."""
        return self._angles

    def run(self):
        """Calculate Euler angles.

        R = R_z(alpha) * R_y(beta) * R_z(gamma)

        cb: cos(beta)
        sb: sin(beta) = sqrt(1 - cos(beta)**2)

        ca: cos(alpha)
        sa: sin(alpha)
        cg: cos(gamma)
        sg: sin(gamma)

        """
        R = self.R * np.linalg.det(self.R)
        cb = R[2, 2]

        cb2 = cb**2
        if cb2 > 1:
            if cb2 < 1 + 1e-10:
                cb2 = 1
            else:
                raise RuntimeError("Rotation matrix is largely distorted.")
        # sqrt may not be very numerically accurate if cb**2 is close to 1.
        sb = np.sqrt(1 - cb2)

        if 1 - cb < 1e-10:  # beta ~ 0
            # A choice with gamma=0
            cg = 1
            sg = 0
            ca = R[0, 0]
            sa = R[1, 0]
        elif cb + 1 < 1e-10:  # beta ~ pi
            # A choice with gamma=0
            cg = 1
            sg = 0
            ca = -R[0, 0]
            sa = -R[1, 0]
        else:
            ca = R[0, 2] / sb
            sa = R[1, 2] / sb
            cg = -R[2, 0] / sb
            sg = R[2, 1] / sb

        self._angles = np.array([ca, sa, cb, sb, cg, sg], dtype="double")

    def recover_R(self) -> np.ndarray:
        """Recover R matrix from angles.

        This is mainly used for the test.

        """
        R = np.zeros((3, 3), dtype="double", order="C")
        ca, sa, cb, sb, cg, sg = self.angles
        R[0, 0] = ca * cb * cg - sa * sg
        R[0, 1] = -ca * cb * sg - sa * cg
        R[0, 2] = ca * sb
        R[1, 0] = sa * cb * cg + ca * sg
        R[1, 1] = -sa * cb * sg + ca * cg
        R[1, 2] = sa * sb
        R[2, 0] = -sb * cg
        R[2, 1] = sb * sg
        R[2, 2] = cb

        return R * np.linalg.det(self.R)


class LyMatrices:
    """Create matrix representation of <lm'|Ly|lm>.

    ihbar/2 is not multiplied.

    """

    def __init__(self, l_max: int = 3):
        """Init method."""
        self._l_max = l_max
        self._Ly = None

    @property
    def Ly(self) -> Optional[list[np.ndarray]]:
        """Return <lm'|Ly|lm> for all l channel."""
        return self._Ly

    def run(self):
        """Calculate <lm'|Ly|lm> for all l channel."""
        self._Ly = []
        for ll in range(self._l_max + 1):
            self._Ly.append(self._get_Ly(ll))

    def _get_Ly(self, ll: int) -> np.ndarray:
        """Compute <lm'|Ly|lm> at l.

        i hbar/2 is not multiplied.

        """
        Ly = np.zeros((2 * ll + 1, 2 * ll + 1), dtype="double", order="C")
        for mp in range(-ll, ll + 1):
            for m in range(-ll, ll + 1):
                if mp == m - 1:
                    Ly[mp + ll, m + ll] += np.sqrt(ll * (ll + 1) - m * (m - 1))
                if mp == m + 1:
                    Ly[mp + ll, m + ll] -= np.sqrt(ll * (ll + 1) - m * (m + 1))
        return Ly


class WignerMatricesEularAngle:
    """Wigner matrices dm'm^l = <lm'|e^ibLy|lm>  is calcualted."""

    def __init__(self, l_max: int = 6):
        """Init method."""
        self._ly = LyMatrices(l_max=l_max)
        self._ly.run()
        self._d = None

    @property
    def d(self) -> Optional[list[np.ndarray]]:
        """Return <lm'|e^ibLy|lm> for all l channels."""
        return self._d

    def run(self, cb: float, sb: float):
        """Calculate <lm'|e^ibLy|lm> for all l channels."""
        self._d = []
        for Ly_l in self._ly.Ly:
            w, U = np.linalg.eigh(0.5j * Ly_l)  # i/2 is multipled to Ly_l.
            exp_w = (cb - 1j * sb) ** w
            self._d.append(np.dot(U * exp_w, U.T.conj()))


class SHRotationMatricesEularAngle:
    """Rotation matrix for spherical harmonics, Dm'm^l.

    Dm'm = e^-iam' dm'm(b) e^-igm (a: alpha, b: beta, g: gamma)
    for usual spherical harmonics and
    Delta_m'm for real spherical harmonics

    """

    def __init__(self, R: np.ndarray, lattice: np.ndarray):
        """Init method.

        Parameters
        ----------
        R: ndarray
            Rotation matrix wrt basis vectors (not Cartesian).
            shape=(3, 3).
        lattice: ndarray.
            Basis vectors in column vectors.
            shape=(3, 3)

        """
        self._R_cart = lattice @ R @ np.linalg.inv(lattice)
        self._detR = np.linalg.det(R)
        self._Delta = None

    @property
    def Delta(self) -> list[np.ndarray]:
        """Return rotation matrix of spherical harmonics."""
        return self._Delta

    def run(self, l_max: int = 6):
        """Rotate spherical harmonics matrices."""
        prop_R_cart = self._R_cart * self._detR
        ea = EulerAngles(prop_R_cart)
        ea.run()
        ca, sa, cb, sb, cg, sg = ea.angles
        wm = WignerMatricesEularAngle(l_max=l_max)
        wm.run(cb, sb)
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        delta = []
        for ll, d_l in enumerate(wm.d):
            D = np.array(d_l, dtype=dtype, order="C")
            for mp in range(-ll, ll + 1):
                D[mp + ll, :] *= (ca - 1j * sa) ** mp
            for m in range(-ll, ll + 1):
                D[:, m + ll] *= (cg - 1j * sg) ** m
            D *= self._detR**ll
            C = make_C(ll, D)
            delta.append(C.conj() @ D @ C.T)
        self._Delta = delta


class LxLyLzMatrices:
    """Create matrix representation of <lm'|Lx|lm>, <lm'|Ly|lm>, <lm'|Lz|lm>.

    hbar is not multiplied. Only integer l's are considered.

    """

    def __init__(self, l_max: int = 6):
        """Init method."""
        self._l_max = l_max
        self._Lx = None
        self._Ly = None
        self._Lz = None

    @property
    def Lx(self) -> Optional[list[np.ndarray]]:
        """Return <lm'|Lx|lm> for all l channel."""
        return self._Lx

    @property
    def Ly(self) -> Optional[list[np.ndarray]]:
        """Return <lm'|Ly|lm> for all l channel."""
        return self._Ly

    @property
    def Lz(self) -> Optional[list[np.ndarray]]:
        """Return <lm'|Lz|lm> for all l channel."""
        return self._Lz

    def run(self):
        """Calculate <lm'|Lx|lm>, <lm'|Lz|lm>, <lm'|Lz|lm> for all l channel."""
        self._Lx = []
        self._Ly = []
        self._Lz = []
        for j in range(self._l_max + 1):
            self._Lx.append(self._get_Lx(j))
            self._Ly.append(self._get_Ly(j))
            self._Lz.append(self._get_Lz(j))
        return self

    def _get_Lx(self, ll: int) -> np.ndarray:
        lx = np.zeros((2 * ll + 1, 2 * ll + 1), dtype=complex, order="C")
        for mp in range(-ll, ll + 1):
            for m in range(-ll, ll + 1):
                if mp == m + 1:
                    lx[mp + ll, m + ll] += np.sqrt((ll - m) * (ll + m + 1))
                if mp == m - 1:
                    lx[mp + ll, m + ll] += np.sqrt((ll + m) * (ll - m + 1))
        return lx / 2

    def _get_Ly(self, ll: int) -> np.ndarray:
        ly = np.zeros((2 * ll + 1, 2 * ll + 1), dtype=complex, order="C")
        for mp in range(-ll, ll + 1):
            for m in range(-ll, ll + 1):
                if mp == m + 1:
                    ly[mp + ll, m + ll] += -np.sqrt((ll - m) * (ll + m + 1))
                if mp == m - 1:
                    ly[mp + ll, m + ll] += np.sqrt((ll + m) * (ll - m + 1))
        return 1j * ly / 2

    def _get_Lz(self, ll: int) -> np.ndarray:
        lz = np.zeros((2 * ll + 1, 2 * ll + 1), dtype=complex, order="C")
        for mp in range(-ll, ll + 1):
            for m in range(-ll, ll + 1):
                if mp == m:
                    lz[mp + ll, m + ll] = m
        return lz


class SHRotationMatrices:
    r"""Rotation matrix for spherical harmonics, Dlm'lm(n,alpha).

    D^{lm'lm}(\mathbf{n}, \alpha) =
        <lm'|e^{-i/\hbar \mathbf{L}\cdot\mathbf{n}\alpha}|lm>.

    """

    def __init__(self, R: np.ndarray, lattice: np.ndarray, lxlylz: LxLyLzMatrices):
        """Init method.

        Parameters
        ----------
        R : ndarray
            Rotation matrix wrt basis vectors (not Cartesian).
            shape=(3, 3).
        lattice : ndarray.
            Basis vectors in column vectors.
            shape=(3, 3)
        lxlylz : LxLyLzMatrices

        """
        self._lxlylz = lxlylz
        self._d = None
        self._Delta = None
        self._n_vec, self._r_order, self._detR = get_n_and_rotation_order(R, lattice)
        if self._r_order == 1:
            self._alpha = 0.0
        else:
            self._alpha = 2 * np.pi / self._r_order

    @property
    def Delta(self) -> Optional[list[np.ndarray]]:
        """Return rotation matrix of spherical harmonics."""
        return self._Delta

    def run(self):
        """Rotate spherical harmonics matrices."""
        self._compute_d()
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        delta = []
        for ll, d_l in enumerate(self._d):
            D = np.array(d_l, dtype=dtype, order="C")
            D *= self._detR**ll
            C = make_C(ll, D)
            delta.append(C.conj() @ D @ C.T)
        self._Delta = delta

    def _compute_d(self):
        r"""Calculate <lm'|e^iL.n\alpha|lm> for all l channels."""
        self._d = []
        n = self._n_vec
        for lx, ly, lz in zip(self._lxlylz.Lx, self._lxlylz.Ly, self._lxlylz.Lz):
            ln = n[0] * lx + n[1] * ly + n[2] * lz
            w, U = np.linalg.eigh(ln)
            self._d.append((U * np.exp(-1j * w * self._alpha)) @ U.T.conj())


def get_n_and_rotation_order(
    R: np.ndarray, L: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Return proper-rotation axis, order of rotation, and det(R).

    r_order = 1 :
        n = (0, 0, 1), alpha = 0.
    r_order = 2 :
        n is taken as one of eigvecs of proper rotation in Cartesian
        coordinates whose eigenvalue = -1. alpha = pi.
    r_order in (3, 4, 6) :
        Given e_1=(1, 0, 0), e_2=(0, 1, 0), e_3=(0, 0, 1),
            v = R.e - e
        is perpendicular to the rotationa axis. Compute cross product,
            v_a = cross(v, R.v).
        The unit vector of the rotation axis is
            n = v_a / |v_a|
        Order of rotation (N) is obtained by iterating rotation matrix until
        being identity. The rotation angle is
            alpha = 2pi / N.

    Parameters
    ----------
    R: ndarray
        Rotation matrix wrt basis vectors (not Cartesian). shape=(3, 3).
    L: ndarray.
        Basis vectors in column vectors. shape=(3, 3)

    Returns
    -------
    tuple :
        n : Unit vector of rotation axis
        r_order : Order of rotation (0 <= theta < pi).
        detR : Determinant of rotation matrix, expected -1 or 1.

    """
    detR = determinant(R)
    R_prop = R * detR
    identity = np.eye(3, dtype=int)
    if np.array_equal(R_prop, identity):
        return np.array([0, 0, 1], dtype="double"), 1, detR

    rbyr = identity.copy()
    for r_order in range(1, 7):  # noqa B007
        rbyr = rbyr @ R_prop
        if np.array_equal(rbyr, identity):
            break

    assert r_order > 1

    r_cart = L @ R @ np.linalg.inv(L) * detR

    if r_order == 2:
        eigvals, eigvecs = np.linalg.eig(r_cart)
        n = None
        for v, vec in zip(eigvals, eigvecs.T):
            if abs(v - 1) < 1e-5:
                n = vec

        assert n is not None

        return n, 2, detR
    else:
        assert r_order in (3, 4, 6)
        vecs = r_cart - identity
        vec = vecs[np.argmax(np.linalg.norm(vecs, axis=0))]
        n = np.cross(vec, r_cart @ vec)
        n /= np.linalg.norm(n)
        return n, r_order, detR


def make_C(ll: int, D: np.ndarray) -> np.ndarray:
    """Make C matrix.

    See L. Chaput et al., Phys. Rev. B 100, 174304 (2019).

    """
    C = np.zeros_like(D)
    C[ll, ll] = np.sqrt(2)
    for i in range(ll):
        C[i, i] = 1j
        C[i, 2 * ll - i] = -1j * (-1) ** (ll - i)
        C[2 * ll - i, i] = 1
        C[2 * ll - i, 2 * ll - i] = (-1) ** (ll - i)
    C /= np.sqrt(2)

    assert (abs(C.conj().T - np.linalg.inv(C)) < 1e-5).all()

    return C
