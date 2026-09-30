"""
Prior dictionaries for lensed CBC analyses.
"""
import math

from ...core.prior import Uniform, LogUniform
from ..prior import BBHPriorDict
DEFAULT_SEGMENT_DURATION = 32.0
SEGMENT_PAD = 4.0
PRIOR_TIME_PAD = 1.0
MIN_PRIOR_R_ORBIT = 10.0


def delta_time_prior_bounds(duration=DEFAULT_SEGMENT_DURATION, pad=PRIOR_TIME_PAD):
    """Half-width of delta_time (seconds) that stays inside an unwrapped FD segment."""
    half_seconds = duration / 2.0 - pad
    if half_seconds <= 0:
        raise ValueError(
            f"duration={duration} s is shorter than 2*pad={2 * pad} s")
    return -half_seconds, half_seconds


class AlignedSpinBBHPriorDict(BBHPriorDict):
    """Aligned-spin BBH priors sampling chirp_mass and mass_ratio."""

    def __init__(self, dictionary=None, filename=None, conversion_function=None,
                 aligned_spin=True):
        if dictionary is not None or filename is not None:
            super().__init__(
                dictionary=dictionary,
                filename=filename,
                conversion_function=conversion_function,
                aligned_spin=aligned_spin,
            )
        else:
            super().__init__(
                aligned_spin=True,
                conversion_function=conversion_function,
            )


class GenericLensedPriorDict(AlignedSpinBBHPriorDict):
    """BBH priors extended with generic lensing parameters."""

    def __init__(self, dictionary=None, filename=None, conversion_function=None,
                 duration=DEFAULT_SEGMENT_DURATION, pad=PRIOR_TIME_PAD):
        if dictionary is not None or filename is not None:
            super().__init__(
                dictionary=dictionary,
                filename=filename,
                conversion_function=conversion_function,
            )
            return
        super().__init__()
        self['relative_mass'] = Uniform(0.9, 1.1, name='relative_mass',
                                        latex_label=r'$\mathcal{M}_2/\mathcal{M}_1$')
        self['relative_distance'] = Uniform(0.1, 20., name='relative_distance',
                                            latex_label=r'$d_{L,2}/d_{L,1}$')
        self['delta_iota'] = Uniform(-0.1, 0.1, name='delta_iota',
                                     latex_label=r'$\Delta\iota$')
        self['delta_phi_12'] = Uniform(-math.pi / 4, math.pi / 4, name='delta_phi_12',
                                       latex_label=r'$\Delta\phi_{12}$')
        self['delta_psi'] = Uniform(-0.1, 0.1, name='delta_psi',
                                    latex_label=r'$\Delta\psi$')
        dt_min, dt_max = delta_time_prior_bounds(duration=duration, pad=pad)
        self['delta_time'] = Uniform(dt_min, dt_max, name='delta_time',
                                     latex_label=r'$\Delta t$ [s]')


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
        self['R_orbit'] = LogUniform(MIN_PRIOR_R_ORBIT, 2000., name='R_orbit',
                                     latex_label=r'$R_{\rm orbit}/R_S$')
        self['log10_M_lz'] = Uniform(4.0, 7.0, name='log10_M_lz',
                                     latex_label=r'$\log_{10} M_{\rm lz}$')
        self['src_pos'] = Uniform(-1.0, 1.0, name='src_pos',
                                  latex_label=r'$y_{\rm src}$')


class SimpleLensedPriorDict(AlignedSpinBBHPriorDict):
    """
    BBH priors with only geometric/simple lensing parameters (delta_time and relative_distance).
    """

    def __init__(self, dictionary=None, filename=None, conversion_function=None,
                 duration=DEFAULT_SEGMENT_DURATION, pad=PRIOR_TIME_PAD):
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
        self['delta_phi_12'] = 0.0
        self['delta_psi'] = 0.0
        self['relative_distance'] = Uniform(0.1, 20., name='relative_distance',
                                            latex_label=r'$d_{L,2}/d_{L,1}$')
        dt_min, dt_max = delta_time_prior_bounds(duration=duration, pad=pad)
        self['delta_time'] = Uniform(dt_min, dt_max, name='delta_time',
                                     latex_label=r'$\Delta t$ [s]')


class UnlensedBBHPriorDict(AlignedSpinBBHPriorDict):
    """Aligned-spin BBH priors for unlensed model comparison."""

    pass
