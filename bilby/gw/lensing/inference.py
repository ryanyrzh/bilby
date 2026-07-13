"""
Nested-sampling inference helpers for lensed CBC model comparison.
"""
import json
import os
from copy import deepcopy

import numpy as np

from ..detector import InterferometerList
from ..likelihood import GravitationalWaveTransient
from ..source import lal_binary_black_hole
from ..waveform_generator import WaveformGenerator
from .conversion import gpc_to_mpc
from .lensing_utils import convert_y_from_Einstein_to_Rorbit
from .priors import AGNLensedPriorDict, GenericLensedPriorDict, UnlensedBBHPriorDict
from .source import (
    DEFAULT_WAVEFORM_KWARGS,
    agn_lensed_binary_black_hole,
    general_lensed_binary_black_hole,
)

REFERENCE_GWFAST_INJECTION = dict(
    Mc=30.0, eta=0.24, iota=0.99 * np.pi / 2, phase=2.0,
    chi1z=0.3, chi2z=0.5, tcoal=0.0,
    R_orbit=50.0, log10_M_lz=4.0, src_pos=0.5,
    dL=1.0, psi=1.0,
)


def reference_bilby_injection(geocent_time=1126259642.413, ra=1.375, dec=-1.2108):
    """Reference injection parameters in bilby units."""
    from ..conversion import (
        chirp_mass_and_mass_ratio_to_component_masses,
        symmetric_mass_ratio_to_mass_ratio,
    )

    mass_ratio = symmetric_mass_ratio_to_mass_ratio(REFERENCE_GWFAST_INJECTION['eta'])
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
        REFERENCE_GWFAST_INJECTION['Mc'], mass_ratio)
    return dict(
        mass_1=mass_1,
        mass_2=mass_2,
        a_1=REFERENCE_GWFAST_INJECTION['chi1z'],
        a_2=REFERENCE_GWFAST_INJECTION['chi2z'],
        luminosity_distance=gpc_to_mpc(REFERENCE_GWFAST_INJECTION['dL']),
        theta_jn=REFERENCE_GWFAST_INJECTION['iota'],
        phase=REFERENCE_GWFAST_INJECTION['phase'],
        psi=REFERENCE_GWFAST_INJECTION['psi'],
        geocent_time=geocent_time,
        ra=ra,
        dec=dec,
        R_orbit=REFERENCE_GWFAST_INJECTION['R_orbit'],
        log10_M_lz=REFERENCE_GWFAST_INJECTION['log10_M_lz'],
        src_pos=REFERENCE_GWFAST_INJECTION['src_pos'],
    )


def build_agn_injection(Mc, R_orbit, y_Eins, log10_M_lz=4.0,
                        geocent_time=1126259642.413, ra=1.375, dec=-1.2108,
                        dL_gpc=1.0, eta=0.24):
    """Build AGN injection parameters for a grid point."""
    from ..conversion import (
        chirp_mass_and_mass_ratio_to_component_masses,
        symmetric_mass_ratio_to_mass_ratio,
    )

    mass_ratio = symmetric_mass_ratio_to_mass_ratio(eta)
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(Mc, mass_ratio)
    src_pos = float(convert_y_from_Einstein_to_Rorbit(y_Eins, R_orbit))
    return dict(
        mass_1=mass_1, mass_2=mass_2,
        a_1=REFERENCE_GWFAST_INJECTION['chi1z'],
        a_2=REFERENCE_GWFAST_INJECTION['chi2z'],
        luminosity_distance=gpc_to_mpc(dL_gpc),
        theta_jn=REFERENCE_GWFAST_INJECTION['iota'],
        phase=REFERENCE_GWFAST_INJECTION['phase'],
        psi=REFERENCE_GWFAST_INJECTION['psi'],
        geocent_time=geocent_time, ra=ra, dec=dec,
        R_orbit=R_orbit, log10_M_lz=log10_M_lz, src_pos=src_pos,
    )


def make_waveform_generator(source_model, duration=8.0, sampling_frequency=2048.0):
    return WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=source_model,
        waveform_arguments=DEFAULT_WAVEFORM_KWARGS.copy(),
    )


def build_injection_ifos(
        injection_parameters, detector_names=('H1', 'L1', 'V1'),
        duration=8.0, sampling_frequency=2048.0,
        source_model=agn_lensed_binary_black_hole):
    """Set up detectors and inject a lensed signal."""
    ifos = InterferometerList(list(detector_names))
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters['geocent_time'] - duration / 2,
    )
    wfg = make_waveform_generator(source_model, duration, sampling_frequency)
    ifos.inject_signal(waveform_generator=wfg, parameters=injection_parameters)
    return ifos, wfg


def run_pe(likelihood, priors, outdir, label, nlive=100, injection_parameters=None):
    """Run nested sampling via bilby."""
    from ...core.sampler import run_sampler
    from ..result import CBCResult

    return run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        nlive=nlive,
        outdir=outdir,
        label=label,
        resume=False,
        sample='unif',
        injection_parameters=injection_parameters,
        result_class=CBCResult,
    )


def network_snr(ifos, injection_parameters, waveform_generator):
    """Optimal network SNR for injected parameters."""
    snr_sq = 0.0
    for ifo in ifos:
        signal = ifo.get_detector_response(
            waveform_generator.frequency_domain_strain(injection_parameters),
            injection_parameters,
        )
        snr_sq += ifo.optimal_snr_squared(signal=signal)
    return float(np.real(snr_sq) ** 0.5)


def _make_likelihood(ifos, waveform_generator, priors):
    return GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        distance_marginalization=False,
        phase_marginalization=False,
        time_marginalization=False,
    )


def compare_models(
        ifos, injection_parameters, lensed_source_model, lensed_priors,
        outdir, label_prefix, nlive=100, unlensed_priors=None):
    """
    Run nested sampling for lensed and unlensed models on the same data.

    Returns dict with log evidences and Bayes factors.
    """
    if unlensed_priors is None:
        unlensed_priors = UnlensedBBHPriorDict()

    duration = ifos[0].strain_data.duration
    sampling_frequency = ifos[0].strain_data.sampling_frequency
    lensed_wfg = make_waveform_generator(
        lensed_source_model, duration, sampling_frequency)
    unlensed_wfg = make_waveform_generator(
        lal_binary_black_hole, duration, sampling_frequency)

    lensed_likelihood = _make_likelihood(ifos, lensed_wfg, lensed_priors)
    unlensed_likelihood = _make_likelihood(ifos, unlensed_wfg, unlensed_priors)

    result_lensed = run_pe(
        lensed_likelihood, lensed_priors, outdir, f'{label_prefix}_lensed',
        nlive=nlive, injection_parameters=injection_parameters,
    )
    result_unlensed = run_pe(
        unlensed_likelihood, unlensed_priors, outdir, f'{label_prefix}_unlensed',
        nlive=nlive, injection_parameters=injection_parameters,
    )

    log_bf_lensed_vs_unlensed = (
        result_lensed.log_evidence - result_unlensed.log_evidence)
    return {
        'log_evidence_lensed': result_lensed.log_evidence,
        'log_evidence_lensed_err': result_lensed.log_evidence_err,
        'log_evidence_unlensed': result_unlensed.log_evidence,
        'log_evidence_unlensed_err': result_unlensed.log_evidence_err,
        'log_bf_lensed_vs_unlensed': log_bf_lensed_vs_unlensed,
        'log10_bf_lensed_vs_unlensed': log_bf_lensed_vs_unlensed / np.log(10),
        'log_bf_signal_vs_noise_lensed': result_lensed.log_bayes_factor,
        'log_bf_signal_vs_noise_unlensed': result_unlensed.log_bayes_factor,
        'result_lensed': result_lensed,
        'result_unlensed': result_unlensed,
    }


def newton_search_required_snr(
        injection_parameters, model='agn', target_log10_bf=2.0, n_steps=2,
        nlive=50, outdir='outdir', label='newton',
        duration=4.0, sampling_frequency=2048.0, y_Eins=None):
    """
    Newton search on luminosity_distance to reach a target log10 Bayes factor.

    Returns required network SNR and final distance.
    """
    params = deepcopy(injection_parameters)
    ref_distance = params['luminosity_distance']

    if model == 'agn':
        source_model = agn_lensed_binary_black_hole
        priors = AGNLensedPriorDict()
    elif model == 'generic':
        from .conversion import convert_agn_to_generic_lensed, generic_gwfast_to_bilby_lensed
        generic = convert_agn_to_generic_lensed(params)
        params = generic_gwfast_to_bilby_lensed(generic, params)
        source_model = general_lensed_binary_black_hole
        priors = GenericLensedPriorDict()
    else:
        raise ValueError(f"Unknown model={model}")

    ifos, wfg = build_injection_ifos(
        params, duration=duration, sampling_frequency=sampling_frequency,
        source_model=source_model)
    orig_snr = network_snr(ifos, params, wfg)

    label_suffix = f'_y{y_Eins:g}' if y_Eins is not None else ''
    current_distance = ref_distance

    for step in range(n_steps):
        comparison = compare_models(
            ifos, params, source_model, priors,
            outdir=outdir, label_prefix=f'{label}{label_suffix}_step{step}',
            nlive=nlive,
        )
        log10_bf = comparison['log10_bf_lensed_vs_unlensed']
        if abs(log10_bf - target_log10_bf) < 0.1:
            break
        scale = 10 ** ((log10_bf - target_log10_bf) / max(abs(log10_bf), 1.0))
        scale = np.clip(scale, 0.1, 10.0)
        current_distance *= scale
        params['luminosity_distance'] = current_distance
        ifos, wfg = build_injection_ifos(
            params, duration=duration, sampling_frequency=sampling_frequency,
            source_model=source_model)

    req_snr = orig_snr * ref_distance / current_distance
    return {
        'required_snr': req_snr,
        'original_snr': orig_snr,
        'ref_distance_mpc': ref_distance,
        'final_distance_mpc': current_distance,
        'log10_bf': comparison['log10_bf_lensed_vs_unlensed'],
    }


def save_comparison_json(results, path):
    """Save model comparison results (without Result objects) to JSON."""
    serializable = {k: v for k, v in results.items()
                    if not k.startswith('result_')}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
