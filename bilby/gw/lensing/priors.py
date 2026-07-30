"""
Prior dictionaries for lensed CBC analyses.
"""
import math

from ...core.prior import ConditionalUniform, PriorDict, Uniform, LogUniform
from ..prior import BBHPriorDict, secondary_mass_condition_function
from .lensing_utils import DAY_TO_SEC


def _base_aligned_spin_priors():
    """Aligned-spin BBH priors sampling mass_1/mass_2 (not chirp_mass/q)."""
    priors = BBHPriorDict(aligned_spin=True)
    m1_min, m1_max = priors['mass_1'].minimum, priors['mass_1'].maximum
    m2_min, m2_max = priors['mass_2'].minimum, priors['mass_2'].maximum
    for key in ('chirp_mass', 'mass_ratio'):
        if key in priors:
            del priors[key]
    priors['mass_1'] = Uniform(
        name='mass_1', minimum=m1_min, maximum=m1_max, unit=r'$M_{\odot}$')
    priors['mass_2'] = ConditionalUniform(
        condition_func=secondary_mass_condition_function,
        name='mass_2', minimum=m2_min, maximum=m2_max, unit=r'$M_{\odot}$')
    return priors


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
        self['relative_mass'] = Uniform(0.7, 1.3, name='relative_mass',
                                        latex_label=r'$\mathcal{M}_2/\mathcal{M}_1$')
        self['relative_distance'] = Uniform(0.1, 10., name='relative_distance',
                                            latex_label=r'$d_{L,2}/d_{L,1}$')
        self['delta_iota'] = Uniform(-math.pi / 4, math.pi / 4, name='delta_iota',
                                     latex_label=r'$\Delta\iota$')
        self['delta_phase'] = Uniform(-math.pi / 4, math.pi / 4, name='delta_phase',
                                      latex_label=r'$\Delta\phi$')
        self['delta_psi'] = Uniform(-math.pi / 4, math.pi / 4, name='delta_psi',
                                    latex_label=r'$\Delta\psi$')
        self['delta_time'] = Uniform(-30/DAY_TO_SEC, 30/DAY_TO_SEC, name='delta_time',
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
        self['R_orbit'] = LogUniform(1., 2000., name='R_orbit',
                                     latex_label=r'$R_{\rm orbit}/R_S$')
        self['log10_M_lz'] = Uniform(4.0, 7.0, name='log10_M_lz',
                                     latex_label=r'$\log_{10} M_{\rm lz}$')
        self['src_pos'] = Uniform(-1.0, 1.0, name='src_pos',
                                  latex_label=r'$y_{\rm src}$')


class SimpleLensedPriorDict(AlignedSpinBBHPriorDict):
    """
    BBH priors with only geometric/simple lensing parameters (delta_time and relative_distance).
    """

    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        if dictionary is not None or filename is not None:
            super().__init__(
                dictionary=dictionary,
                filename=filename,
                conversion_function=conversion_function,
            )
            return
        super().__init__()
        self['relative_mass'] = 1.0
        self['delta_iota'] = 0.0
        self['delta_phase'] = 0.0
        self['delta_psi'] = 0.0
        self['relative_distance'] = Uniform(0.1, 10., name='relative_distance',
                                            latex_label=r'$d_{L,2}/d_{L,1}$')
        self['delta_time'] = Uniform(-30/DAY_TO_SEC, 30/DAY_TO_SEC, name='delta_time',
                                     latex_label=r'$\Delta t$ [days]')


class UnlensedBBHPriorDict(AlignedSpinBBHPriorDict):
    """Aligned-spin BBH priors for unlensed model comparison."""

    pass
