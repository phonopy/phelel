"""Exceptions of phelel."""


class MagneticMomentsError(ValueError):
    """Magnetic moments do not fit the spin of the VASP calculation."""


class MagneticMomentsWarning(UserWarning):
    """Magnetic moments are not given for a spin-polarized VASP calculation."""
