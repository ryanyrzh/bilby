"""
Lensed gravitational-wave source models for bilby.
"""
import numpy as np

from ..source import _base_lal_cbc_fd_waveform
from .conversion import (
    bilby_to_gwfast_params,
    get_image_time_delay_seconds,
    get_lensed_parameter_sets,
    gwfast_image_to_bilby_params,
)
from .lensing_utils import get_agn_lensed_parameters

DEFAULT_WAVEFORM_KWARGS = dict(
    waveform_approximant='IMRPhenomD',
    reference_frequency=50.0,
    minimum_frequency=10.0,
    catch_waveform_errors=True,
    pn_amplitude_order=0,
    pn_phase_order=-1,
    pn_spin_order=-1,
    pn_tidal_order=-1,
)


def _aligned_spin_kwargs():
    return dict(tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0)


def _sum_lensed_waveforms(frequency_array, params_1, params_2, waveform_kwargs):
    """Generate and sum two aligned-spin CBC waveforms with inter-image time delay."""
    for params in (params_1, params_2):
        # Reject non-finite / non-positive masses before calling LAL — NaNs
        # slip past `<= 0` and otherwise spam XLAL/bilby warnings to stderr.
        m1, m2 = params['mass_1'], params['mass_2']
        if not (np.isfinite(m1) and np.isfinite(m2) and m1 > 0 and m2 > 0):
            return None
    spin_kwargs = _aligned_spin_kwargs()
    waveform_kwargs = waveform_kwargs.copy()
    waveform_kwargs.setdefault('maximum_frequency', frequency_array[-1])
    h1 = _base_lal_cbc_fd_waveform(
        frequency_array,
        mass_1=params_1['mass_1'],
        mass_2=params_1['mass_2'],
        luminosity_distance=params_1['luminosity_distance'],
        theta_jn=params_1['theta_jn'],
        phase=params_1['phase'],
        a_1=params_1.get('a_1', 0.0),
        a_2=params_1.get('a_2', 0.0),
        **spin_kwargs,
        **waveform_kwargs,
    )
    if h1 is None:
        return None

    h2 = _base_lal_cbc_fd_waveform(
        frequency_array,
        mass_1=params_2['mass_1'],
        mass_2=params_2['mass_2'],
        luminosity_distance=params_2['luminosity_distance'],
        theta_jn=params_2['theta_jn'],
        phase=params_2['phase'],
        a_1=params_2.get('a_1', 0.0),
        a_2=params_2.get('a_2', 0.0),
        **spin_kwargs,
        **waveform_kwargs,
    )
    if h2 is None:
        return None

    # TODO: double check this
    # Detector response applies a single psi (image 1). Rotate image-2
    # polarizations by delta_psi so the relative polarization angle is kept.
    delta_psi = params_2.get('psi', 0.0) - params_1.get('psi', 0.0)
    if delta_psi != 0.0:
        c2 = np.cos(2.0 * delta_psi)
        s2 = np.sin(2.0 * delta_psi)
        h2_plus = h2['plus'] * c2 - h2['cross'] * s2
        h2_cross = h2['plus'] * s2 + h2['cross'] * c2
    else:
        h2_plus = h2['plus']
        h2_cross = h2['cross']

    dt_seconds = get_image_time_delay_seconds(params_1, params_2)
    time_shift = np.exp(-1j * 2 * np.pi * frequency_array * dt_seconds)
    return {
        'plus': h1['plus'] + h2_plus * time_shift,
        'cross': h1['cross'] + h2_cross * time_shift,
    }


def _pop_extrinsic_kwargs(kwargs):
    """Remove bilby extrinsic parameters not used by the source model."""
    for key in ['ra', 'dec', 'geocent_time', 'time_jitter', 'azimuth', 'zenith',
                'H1_time', 'L1_time', 'V1_time', 'reference_frame', 'chirp_mass',
                'mass_ratio', 'total_mass', 'symmetric_mass_ratio', 'R_orbit',
                'log10_M_lz', 'src_pos', 'tcoal_days']:
        kwargs.pop(key, None)
    return kwargs


def general_lensed_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, a_2,
        theta_jn, phase, psi,
        delta_iota=0.0, delta_phase=0.0, delta_psi=0.0,
        relative_distance=1.0, relative_mass=1.0, delta_time=0.0,
        **kwargs):
    """
    Generic dual-image lensed BBH source model (aligned spins, IMRPhenomD).

    Lensing is parameterized by phenomenological deltas relative to the
    plus image.
    """
    waveform_kwargs = DEFAULT_WAVEFORM_KWARGS.copy()
    waveform_kwargs.update(_pop_extrinsic_kwargs(kwargs))
    waveform_kwargs = {k: v for k, v in waveform_kwargs.items() if k in DEFAULT_WAVEFORM_KWARGS}

    parameters = dict(
        mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance,
        a_1=a_1, a_2=a_2,
        theta_jn=theta_jn, phase=phase, psi=psi,
        delta_iota=delta_iota, delta_phase=delta_phase, delta_psi=delta_psi,
        relative_distance=relative_distance, relative_mass=relative_mass,
        delta_time=delta_time, tcoal_days=0.0,
    )
    params_1, params_2 = get_lensed_parameter_sets(parameters)
    params_2['delta_time'] = delta_time
    return _sum_lensed_waveforms(frequency_array, params_1, params_2, waveform_kwargs)


def agn_lensed_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, a_2,
        theta_jn, phase, psi,
        R_orbit, log10_M_lz, src_pos,
        **kwargs):
    """
    AGN-embedded lensed BBH source model (aligned spins, IMRPhenomD).

    Uses AGN disk geometry from lensing_utils to produce two lensed images.
    """
    waveform_kwargs = DEFAULT_WAVEFORM_KWARGS.copy()
    waveform_kwargs.update(_pop_extrinsic_kwargs(kwargs))
    waveform_kwargs = {k: v for k, v in waveform_kwargs.items() if k in DEFAULT_WAVEFORM_KWARGS}

    bilby_params = dict(
        mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance,
        a_1=a_1, a_2=a_2,
        theta_jn=theta_jn, phase=phase, psi=psi,
        R_orbit=R_orbit, log10_M_lz=log10_M_lz, src_pos=src_pos,
        tcoal=0.0,
    )
    gwfast_params = bilby_to_gwfast_params(bilby_params)
    params_1, params_2 = get_agn_lensed_parameters(gwfast_params)
    bilby_1 = gwfast_image_to_bilby_params(params_1, bilby_params)
    bilby_2 = gwfast_image_to_bilby_params(params_2, bilby_params)
    return _sum_lensed_waveforms(frequency_array, bilby_1, bilby_2, waveform_kwargs)
