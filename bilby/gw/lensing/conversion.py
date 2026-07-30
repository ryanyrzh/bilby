"""
Parameter conversion between bilby and gwfast lensing conventions.
"""
import numpy as np

from ..conversion import (
    chirp_mass_and_mass_ratio_to_component_masses,
    component_masses_to_chirp_mass,
    component_masses_to_symmetric_mass_ratio,
    symmetric_mass_ratio_to_mass_ratio,
)
from .lensing_utils import DAY_TO_SEC, get_agn_lensed_parameters


GPC_TO_MPC = 1000.0


def mpc_to_gpc(luminosity_distance_mpc):
    return luminosity_distance_mpc / GPC_TO_MPC


def gpc_to_mpc(luminosity_distance_gpc):
    return luminosity_distance_gpc * GPC_TO_MPC


def chirp_mass_and_eta_to_component_masses(chirp_mass, eta):
    """Convert chirp mass and symmetric mass ratio to component masses."""
    mass_ratio = symmetric_mass_ratio_to_mass_ratio(eta)
    return chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)


def bilby_to_gwfast_params(parameters):
    """Convert bilby CBC parameters to gwfast lensing parameter dict."""
    params = dict(parameters)
    if 'mass_1' in params and 'mass_2' in params:
        mass_1, mass_2 = params['mass_1'], params['mass_2']
    else:
        mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
            params['chirp_mass'], params['mass_ratio'])
    params['Mc'] = component_masses_to_chirp_mass(mass_1, mass_2)
    params['eta'] = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
    params['iota'] = params.get('theta_jn', params.get('iota'))
    params['chi1z'] = params.get('a_1', params.get('chi1z', 0.0))
    params['chi2z'] = params.get('a_2', params.get('chi2z', 0.0))
    params['dL'] = mpc_to_gpc(params['luminosity_distance'])
    params['phase'] = params.get('phase', 0.0)
    params['psi'] = params.get('psi', 0.0)
    params['tcoal'] = params.get('tcoal', 0.0)
    if 'log10_M_lz' in params:
        params['M_lz'] = 10.0 ** params['log10_M_lz']
    return params


def gwfast_image_to_bilby_params(image_params, base_parameters):
    """Convert gwfast parameters to bilby parameters (for one image)."""
    mass_1, mass_2 = chirp_mass_and_eta_to_component_masses(
        image_params['Mc'], image_params['eta'])
    out = dict(base_parameters)
    out.update({
        'mass_1': mass_1,
        'mass_2': mass_2,
        'luminosity_distance': gpc_to_mpc(image_params['dL']),
        'theta_jn': image_params['iota'],
        'phase': image_params['phase'],
        'psi': image_params.get('psi', out.get('psi', 0.0)),
        'a_1': image_params.get('chi1z', out.get('a_1', 0.0)),
        'a_2': image_params.get('chi2z', out.get('a_2', 0.0)),
        'tilt_1': 0.0,
        'tilt_2': 0.0,
        'phi_12': 0.0,
        'phi_jl': 0.0,
    })
    if 'tcoal' in image_params:
        out['tcoal_days'] = image_params['tcoal']
    return out


def _get_parameter_pairs(key, reference_parameters, mode):
    alternative_keys = {
        'luminosity_distance': 'distance',
        'chirp_mass': 'mass',
        'geocent_time': 'time',
        'tcoal_days': 'time',
    }

    param_1 = reference_parameters.pop(f"{key}_1", None)
    if param_1 is None:
        param_1 = reference_parameters.pop(key, None)
    if param_1 is None:
        return None, None

    param_2 = reference_parameters.pop(f"{key}_2", None)
    if param_2 is None:
        alt_key = alternative_keys.get(key, key)
        variation = reference_parameters.pop(f"{mode}_{alt_key}", None)
        if variation is None:
            if mode == 'relative' and key == 'chirp_mass':
                variation = reference_parameters.pop('relative_mass', None)
            elif mode == 'delta' and key == 'theta_jn':
                variation = reference_parameters.pop('delta_iota', None)
            elif mode == 'delta' and key == 'phase':
                variation = reference_parameters.pop('delta_phase', None)
            elif mode == 'delta' and key == 'psi':
                variation = reference_parameters.pop('delta_psi', None)
            elif mode == 'relative' and key == 'luminosity_distance':
                variation = reference_parameters.pop('relative_distance', None)
            elif mode == 'delta' and key == 'tcoal_days':
                variation = reference_parameters.pop('delta_time', None)
        if variation is None:
            return param_1, None
        if mode == 'relative':
            param_2 = param_1 * variation
        elif mode == 'delta':
            param_2 = param_1 + variation

    return param_1, param_2


def get_lensed_parameter_sets(parameters_dict):
    """
    Split bilby parameters into two image parameter sets.

    Supports paired (param_1/param_2) or delta/relative forms.
    """
    ref_parameters_dict = parameters_dict.copy()
    theta_jn_1, theta_jn_2 = _get_parameter_pairs('theta_jn', ref_parameters_dict, 'delta')
    phase_1, phase_2 = _get_parameter_pairs('phase', ref_parameters_dict, 'delta')
    psi_1, psi_2 = _get_parameter_pairs('psi', ref_parameters_dict, 'delta')
    distance_1, distance_2 = _get_parameter_pairs('luminosity_distance', ref_parameters_dict, 'relative')

    if 'mass_1' in parameters_dict:
        mass_1 = parameters_dict['mass_1']
        mass_2 = parameters_dict['mass_2']
    else:
        mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
            parameters_dict['chirp_mass'], parameters_dict['mass_ratio'])

    rel_mass = parameters_dict.get('relative_mass', 1.0)
    mass_1_2 = mass_1 * rel_mass
    mass_2_2 = mass_2 * rel_mass

    if theta_jn_2 is None:
        theta_jn_2 = theta_jn_1 + parameters_dict.get('delta_iota', 0.0)
    if phase_2 is None:
        phase_2 = phase_1 + parameters_dict.get('delta_phase', 0.0)
    if psi_2 is None:
        psi_2 = psi_1 + parameters_dict.get('delta_psi', 0.0)
    if distance_2 is None:
        distance_2 = distance_1 * parameters_dict.get('relative_distance', 1.0)

    lensing_keys = {
        'delta_iota', 'delta_phase', 'delta_psi', 'relative_distance',
        'relative_mass', 'delta_time', 'R_orbit', 'log10_M_lz', 'src_pos',
    }
    base = {k: v for k, v in parameters_dict.items() if k not in lensing_keys}

    signal_1_params = base.copy()
    signal_2_params = base.copy()

    signal_1_params.update({
        'mass_1': mass_1,
        'mass_2': mass_2,
        'theta_jn': theta_jn_1,
        'phase': phase_1,
        'psi': psi_1,
        'luminosity_distance': distance_1,
        'tcoal_days': parameters_dict.get('tcoal_days', 0.0),
    })
    signal_2_params.update({
        'mass_1': mass_1_2,
        'mass_2': mass_2_2,
        'theta_jn': theta_jn_2,
        'phase': phase_2,
        'psi': psi_2,
        'luminosity_distance': distance_2,
        'tcoal_days': (
            parameters_dict.get('tcoal_days', 0.0)
            + parameters_dict.get('delta_time', 0.0)),
    })

    return signal_1_params, signal_2_params


def convert_agn_to_generic_lensed(parameters):
    """Convert AGN lensed parameters to generic lensing delta parameters."""
    gwfast_params = bilby_to_gwfast_params(parameters)
    params_1, params_2 = get_agn_lensed_parameters(gwfast_params)

    output_params = bilby_to_gwfast_params(parameters)
    output_params['tcoal'] = params_1.get('tcoal', 0.0)
    output_params['delta_time'] = params_2['tcoal'] - params_1.get('tcoal', 0.0)
    output_params.update({
        'dL': params_1['dL'],
        'relative_distance': params_2['dL'] / params_1['dL'],
        'iota': params_1['iota'],
        'delta_iota': params_2['iota'] - params_1['iota'],
        'phase': params_1['phase'],
        'delta_phase': params_2['phase'] - params_1['phase'],
        'psi': params_1['psi'],
        'delta_psi': params_2['psi'] - params_1['psi'],
        'Mc': params_1['Mc'],
        'eta': params_1['eta'],
        'relative_mass': params_2['Mc'] / params_1['Mc'],
    })
    return output_params

def convert_agn_to_simple_lensed(parameters):
    """Convert AGN lensed parameters to simple lensing parameters."""
    output_params = convert_agn_to_generic_lensed(parameters)
    output_params.update({
        'delta_iota': 0.0,
        'delta_phase': 0.0,
        'delta_psi': 0.0,
        'relative_mass': 1.0,
    })
    return output_params

def generic_gwfast_to_bilby_lensed(generic_gwfast_params, base_bilby_params):
    """Convert generic gwfast lensing params to bilby params for waveform generation."""
    bilby_params = dict(base_bilby_params)
    mass_1, mass_2 = chirp_mass_and_eta_to_component_masses(
        generic_gwfast_params['Mc'], generic_gwfast_params['eta'])
    bilby_params.update({
        'mass_1': mass_1,
        'mass_2': mass_2,
        'luminosity_distance': gpc_to_mpc(generic_gwfast_params['dL']),
        'theta_jn': generic_gwfast_params['iota'],
        'phase': generic_gwfast_params['phase'],
        'psi': generic_gwfast_params.get('psi', bilby_params.get('psi', 0.0)),
        'delta_iota': generic_gwfast_params.get('delta_iota', 0.0),
        'delta_phase': generic_gwfast_params.get('delta_phase', 0.0),
        'delta_psi': generic_gwfast_params.get('delta_psi', 0.0),
        'relative_distance': generic_gwfast_params.get('relative_distance', 1.0),
        'relative_mass': generic_gwfast_params.get('relative_mass', 1.0),
        'delta_time': generic_gwfast_params.get('delta_time', 0.0),
        'tcoal_days': generic_gwfast_params.get('tcoal', 0.0),
    })
    return bilby_params


def get_image_time_delay_seconds(params_1, params_2):
    """Inter-image time delay in seconds from tcoal difference (days)."""
    t1 = params_1.get('tcoal_days', 0.0)
    t2 = params_2.get('tcoal_days', 0.0)
    return (t2 - t1) * DAY_TO_SEC
