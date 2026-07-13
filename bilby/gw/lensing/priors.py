"""
Prior dictionaries for lensed CBC analyses.
"""
import math

from ...core.prior import PriorDict, Uniform, LogUniform
from ..prior import BBHPriorDict


def _base_aligned_spin_priors():
    """Aligned-spin BBH priors without precession parameters."""
    return BBHPriorDict(aligned_spin=True)


class AlignedSpinBBHPriorDict(PriorDict):
    """Aligned-spin BBH priors suitable for IMRPhenomD lensing analyses."""

    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        if dictionary is not None or filename is not None:
            super().__init__(
                dictionary=dictionary,
                filename=filename,
                conversion_function=conversion_function,
            )
        else:
            super().__init__(_base_aligned_spin_priors())


class GenericLensedPriorDict(AlignedSpinBBHPriorDict):
    """BBH priors extended with generic lensing parameters."""

    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        if dictionary is not None or filename is not None:
            super().__init__(
                dictionary=dictionary,
                filename=filename,
                conversion_function=conversion_function,
            )
            return
        super().__init__()
        self['relative_mass'] = Uniform(0.0, 2.0, name='relative_mass',
                                        latex_label=r'$\mathcal{M}_{c,2}/\mathcal{M}_{c,1}$')
        self['relative_distance'] = Uniform(0.1, 10.0, name='relative_distance',
                                            latex_label=r'$d_{L,2}/d_{L,1}$')
        self['delta_iota'] = Uniform(-math.pi / 2, math.pi / 2, name='delta_iota',
                                     latex_label=r'$\Delta\iota$')
        self['delta_phase'] = Uniform(-math.pi / 2, math.pi / 2, name='delta_phase',
                                      latex_label=r'$\Delta\phi$')
        self['delta_psi'] = Uniform(-math.pi / 2, math.pi / 2, name='delta_psi',
                                    latex_label=r'$\Delta\psi$')
        self['delta_time'] = Uniform(-1.0, 1.0, name='delta_time',
                                     latex_label=r'$\Delta t$ [days]')


class AGNLensedPriorDict(AlignedSpinBBHPriorDict):
    """BBH priors extended with AGN lensing parameters."""

    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        if dictionary is not None or filename is not None:
            super().__init__(
                dictionary=dictionary,
                filename=filename,
                conversion_function=conversion_function,
            )
            return
        super().__init__()
        self['R_orbit'] = LogUniform(10.0, 2000.0, name='R_orbit',
                                     latex_label=r'$R_{\rm orbit}/R_S$')
        self['log10_M_lz'] = Uniform(3.0, 7.0, name='log10_M_lz',
                                     latex_label=r'$\log_{10} M_{\rm lz}$')
        self['src_pos'] = Uniform(-1.0, 1.0, name='src_pos',
                                  latex_label=r'$y_{\rm src}$')


class UnlensedBBHPriorDict(AlignedSpinBBHPriorDict):
    """Aligned-spin BBH priors for unlensed model comparison."""

    pass
