"""
Nested-sampling inference helpers for lensed CBC model comparison.
"""
import json
import os
from copy import deepcopy

import numpy as np

from ..detector import InterferometerList
from ..likelihood import GravitationalWaveTransient
from ..source import PARAMETER_SETS
from ..waveform_generator import WaveformGenerator
from .conversion import bilby_to_gwfast_params, gpc_to_mpc
from .lensing_utils import DAY_TO_SEC, convert_y_from_Einstein_to_Rorbit, get_agn_lensed_parameters
from .priors import (
    AGNLensedPriorDict,
    DEFAULT_SEGMENT_DURATION,
    GenericLensedPriorDict,
    SEGMENT_PAD,
    SimpleLensedPriorDict,
)
from .source import (
    DEFAULT_WAVEFORM_KWARGS,
    agn_lensed_binary_black_hole,
    general_lensed_binary_black_hole,
)

# Parameter order for corner plots
_INTRINSIC_CORNER_ORDER = (
    'chirp_mass', 'mass_ratio', 'mass_1', 'mass_2', 'total_mass',
    'symmetric_mass_ratio',
    'chi_1', 'chi_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
    'chi_1_in_plane', 'chi_2_in_plane',
    'lambda_1', 'lambda_2', 'lambda_tilde', 'delta_lambda_tilde',
    'phase', 'delta_phase',
)
_EXTRINSIC_CORNER_ORDER = (
    'luminosity_distance', 'redshift',
    'ra', 'dec', 'azimuth', 'zenith',
    'theta_jn', 'cos_theta_jn',
    'psi',
    'geocent_time', 'time_jitter',
    'H1_time', 'L1_time', 'V1_time',
)
_LENSING_CORNER_ORDER = (
    'relative_mass', 'relative_distance',
    'delta_iota', 'delta_phi_12', 'delta_psi', 'delta_time',
    'R_orbit', 'log10_M_lz', 'src_pos',
)

reference_params = dict(
    Mc=30.0, eta=0.24, iota=0.99 * np.pi / 2, phase=2.0,
    chi1z=0.3, chi2z=0.5, tcoal=0.0,
    R_orbit=50.0, log10_M_lz=5.0, src_pos=0.5,
    dL=0.5, psi=1.0,
)


def reference_bilby_injection(geocent_time=1126259642.413, ra=1.375, dec=-1.2108):
    """
    Reference injection parameters in bilby units.

    Masses are chirp_mass and mass_ratio. Waveform generation converts those
    to mass_1 and mass_2.
    """
    from ..conversion import symmetric_mass_ratio_to_mass_ratio

    return dict(
        chirp_mass=reference_params['Mc'],
        mass_ratio=symmetric_mass_ratio_to_mass_ratio(reference_params['eta']),
        a_1=reference_params['chi1z'],
        a_2=reference_params['chi2z'],
        tilt_1=0.0,
        tilt_2=0.0,
        luminosity_distance=gpc_to_mpc(reference_params['dL']),
        theta_jn=reference_params['iota'],
        phase=reference_params['phase'],
        psi=reference_params['psi'],
        geocent_time=geocent_time,
        ra=ra,
        dec=dec,
        R_orbit=reference_params['R_orbit'],
        log10_M_lz=reference_params['log10_M_lz'],
        src_pos=reference_params['src_pos'],
    )


def set_example_pe_time_prior(priors, geocent_time, window=0.1):
    """Sample coalescence time in a narrow window around the injection."""
    from ...core.prior import Uniform
    priors['geocent_time'] = Uniform(
        geocent_time - window, geocent_time + window, name='geocent_time')


def build_agn_injection(Mc, R_orbit, y_Eins, log10_M_lz=4.0,
                        geocent_time=1126259642.413, ra=1.375, dec=-1.2108,
                        dL_gpc=1.0, eta=0.24):
    """
    Build AGN injection parameters for a grid point.

    Masses are chirp_mass and mass_ratio. Waveform generation converts those
    to mass_1 and mass_2.
    """
    from ..conversion import symmetric_mass_ratio_to_mass_ratio

    src_pos = float(convert_y_from_Einstein_to_Rorbit(y_Eins, R_orbit))
    return dict(
        chirp_mass=Mc,
        mass_ratio=symmetric_mass_ratio_to_mass_ratio(eta),
        a_1=reference_params['chi1z'],
        a_2=reference_params['chi2z'],
        tilt_1=0.0,
        tilt_2=0.0,
        luminosity_distance=gpc_to_mpc(dL_gpc),
        theta_jn=reference_params['iota'],
        phase=reference_params['phase'],
        psi=reference_params['psi'],
        geocent_time=geocent_time, ra=ra, dec=dec,
        R_orbit=R_orbit, log10_M_lz=log10_M_lz, src_pos=src_pos,
    )


def make_waveform_generator(source_model, duration=DEFAULT_SEGMENT_DURATION,
                            sampling_frequency=2048.0):
    return WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=source_model,
        waveform_arguments=DEFAULT_WAVEFORM_KWARGS.copy(),
    )


def legal_duration(duration, sampling_frequency=2048.0):
    """Smallest duration >= requested with integer sampling_frequency * duration."""
    n_samples = int(np.ceil(float(duration) * float(sampling_frequency) - 1e-12))
    return n_samples / float(sampling_frequency)


def lensing_segment_times(geocent_time, delta_t_seconds,
                          duration=DEFAULT_SEGMENT_DURATION, pad=SEGMENT_PAD,
                          sampling_frequency=2048.0):
    """
    Duration and start_time so both image coalescences plus pad lie in-band.

    Frequency-domain time shifts wrap on ``duration``; the returned window
    keeps both coalescences away from the periodic identification. Duration is
    always a legal bilby value (fs * T is an integer).
    """
    dt = float(delta_t_seconds)
    needed = abs(dt) + 2.0 * pad
    if duration is None:
        duration = max(DEFAULT_SEGMENT_DURATION, needed)
    duration = float(duration)
    if duration < needed:
        duration = needed
    duration = legal_duration(duration, sampling_frequency=sampling_frequency)
    t_min = min(0.0, dt)
    start_time = geocent_time + t_min - pad
    return duration, start_time


def both_images_in_segment(geocent_time, delta_t_seconds, duration, start_time,
                           pad=SEGMENT_PAD):
    """True if both coalescences plus pad fit in [start_time, start_time+duration)."""
    t1 = geocent_time
    t2 = geocent_time + delta_t_seconds
    lo = start_time + pad
    hi = start_time + duration - pad
    return (lo <= t1 <= hi) and (lo <= t2 <= hi)


def injection_delta_t_seconds(injection_parameters):
    """Inter-image delay in seconds from generic or AGN injection dict."""
    if 'delta_time' in injection_parameters:
        return float(injection_parameters['delta_time'])
    if all(k in injection_parameters for k in ('R_orbit', 'log10_M_lz', 'src_pos')):
        gwfast = bilby_to_gwfast_params(injection_parameters)
        plus, minus = get_agn_lensed_parameters(gwfast)
        if plus is None:
            return 0.0
        t1 = plus.get('tcoal', 0.0)
        t2 = minus.get('tcoal', 0.0)
        return (t2 - t1) * DAY_TO_SEC
    return 0.0


def build_injection_ifos(
        injection_parameters, detector_names=('H1', 'L1', 'V1'),
        duration=DEFAULT_SEGMENT_DURATION, sampling_frequency=2048.0,
        source_model=agn_lensed_binary_black_hole, start_time=None, pad=SEGMENT_PAD):
    """Set up detectors and inject a lensed signal without FD wrap of image 2."""
    geocent_time = injection_parameters['geocent_time']
    if start_time is None:
        dt_seconds = injection_delta_t_seconds(injection_parameters)
        duration, start_time = lensing_segment_times(
            geocent_time, dt_seconds, duration=duration, pad=pad,
            sampling_frequency=sampling_frequency)
    else:
        duration = legal_duration(duration, sampling_frequency=sampling_frequency)
    ifos = InterferometerList(list(detector_names))
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )
    wfg = make_waveform_generator(source_model, duration, sampling_frequency)
    ifos.inject_signal(waveform_generator=wfg, parameters=injection_parameters)
    return ifos, wfg


def run_pe(likelihood, priors, outdir, label, nlive=100, npool=1,
           injection_parameters=None, resume=False, **sampler_kwargs):
    """Run nested sampling via bilby."""
    from ...core.sampler import run_sampler
    from ..result import CBCResult

    sampler_kwargs = {
        key: value for key, value in sampler_kwargs.items() if value is not None
    }
    sampler_kwargs.setdefault('sample', 'acceptance-walk')
    return run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        nlive=nlive,
        npool=npool,
        outdir=outdir,
        label=label,
        resume=resume,
        injection_parameters=injection_parameters,
        result_class=CBCResult,
        **sampler_kwargs,
    )


def _with_component_masses(sample):
    from ..conversion import generate_mass_parameters
    return generate_mass_parameters(sample)


def _mass_summary(values):
    median, low, high = np.percentile(np.asarray(values, dtype=float), [50, 16, 84])
    return median, median - low, high - median


def print_component_masses(result):
    """
    Print image-1 component masses derived from chirp_mass and mass_ratio.

    When relative_mass is present, also print image-2 masses
    (image-1 masses scaled by relative_mass).
    """
    posterior = _with_component_masses(result.posterior)
    result.posterior = posterior
    inj = getattr(result, 'injection_parameters', None) or {}
    inj_masses = _with_component_masses(dict(inj)) if inj else {}

    print('Image 1 component masses, converted from chirp_mass and mass_ratio:')
    for key in ('mass_1', 'mass_2'):
        median, minus, plus = _mass_summary(posterior[key])
        injected = inj_masses.get(key)
        injected_text = 'n/a' if injected is None else f'{float(injected):.6g}'
        print(
            f'  {key}: injected={injected_text}  '
            f'posterior={median:.6g} -{minus:.3g} +{plus:.3g}'
        )

    if 'relative_mass' not in posterior.columns and 'relative_mass' not in inj_masses:
        return
    if 'relative_mass' in posterior.columns:
        relative_mass = posterior['relative_mass'].to_numpy()
        injected_relative = inj_masses.get('relative_mass', np.nan)
    else:
        relative_mass = float(inj_masses['relative_mass'])
        injected_relative = relative_mass
    print('Image 2 component masses = image 1 * relative_mass:')
    for key in ('mass_1', 'mass_2'):
        median, minus, plus = _mass_summary(posterior[key].to_numpy() * relative_mass)
        if key in inj_masses and np.isfinite(injected_relative):
            injected_text = f'{float(inj_masses[key]) * float(injected_relative):.6g}'
        else:
            injected_text = 'n/a'
        print(
            f'  {key}: injected={injected_text}  '
            f'posterior={median:.6g} -{minus:.3g} +{plus:.3g}'
        )


def _finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _aligned_spin_chi(injection, a_key, tilt_key):
    """Aligned-spin chi = a * cos(tilt). tilt defaults to 0."""
    a_value = _finite_or_none(injection.get(a_key))
    if a_value is None:
        return None
    tilt = _finite_or_none(injection.get(tilt_key))
    if tilt is None:
        tilt = 0.0
    return a_value * float(np.cos(tilt))


def _injection_for_corner(injection):
    """
    Copy of the injection with sampled-parameter names filled in.

    The prior samples chirp_mass, mass_ratio, chi_1, and chi_2. Injections
    store chirp_mass, mass_ratio, a_1, and a_2.
    """
    filled = _with_component_masses(dict(injection))
    injection = dict(injection)
    for key in ('chirp_mass', 'mass_ratio'):
        if key in filled:
            injection.setdefault(key, filled[key])
    if 'chi_1' not in injection:
        chi_1 = _aligned_spin_chi(injection, 'a_1', 'tilt_1')
        if chi_1 is not None:
            injection['chi_1'] = chi_1
    if 'chi_2' not in injection:
        chi_2 = _aligned_spin_chi(injection, 'a_2', 'tilt_2')
        if chi_2 is not None:
            injection['chi_2'] = chi_2
    return injection


def _range_including_truth(samples, truth):
    """Axis limits covering the samples and, when finite, the injection."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return 1
    low = float(np.min(samples))
    high = float(np.max(samples))
    if truth is not None:
        low = min(low, truth)
        high = max(high, truth)
    if high <= low:
        pad = max(abs(low) * 1e-6, 1e-6)
    else:
        pad = 0.05 * (high - low)
    return (low - pad, high + pad)


def order_lensing_corner_parameters(keys):
    """
    Order corner plot keys: intrinsic/source, extrinsic/detector, then lensing.

    Keys outside those sets keep their input order and are appended last.
    """
    keys = list(keys)
    groups = (
        (PARAMETER_SETS['intrinsic'], _INTRINSIC_CORNER_ORDER),
        (PARAMETER_SETS['extrinsic'], _EXTRINSIC_CORNER_ORDER),
        (PARAMETER_SETS['lensing'], _LENSING_CORNER_ORDER),
    )
    ordered = []
    used = set()
    present = set(keys)
    for members, preferred in groups:
        for name in preferred:
            if name in present and name in members and name not in used:
                ordered.append(name)
                used.add(name)
        for name in keys:
            if name in members and name not in used:
                ordered.append(name)
                used.add(name)
    for name in keys:
        if name not in used:
            ordered.append(name)
    return ordered


def plot_lensing_corner(result, dpi=100, parameters=None, **kwargs):
    """
    Corner plot of the searched parameters, including chirp_mass and mass_ratio.

    Default axis order is intrinsic/source parameters, then extrinsic/detector
    parameters, then lensing parameters. Pass ``parameters`` to set a different
    order. Fills chirp_mass, mass_ratio, chi_1, and chi_2 on the injection so
    truth lines match the sampled parameters. Missing truths are omitted. Axis
    limits include the injection when it lies outside the posterior.
    """
    inj = getattr(result, 'injection_parameters', None)
    user_truths = 'truths' in kwargs or 'truth' in kwargs
    if parameters is None and not user_truths:
        parameters = order_lensing_corner_parameters(result.search_parameter_keys)
    if inj is not None and not user_truths:
        inj = _injection_for_corner(inj)
        result.injection_parameters = inj
        truths = [_finite_or_none(inj.get(key)) for key in parameters]
        kwargs['truths'] = truths
        if 'range' not in kwargs:
            kwargs['range'] = [
                _range_including_truth(result.posterior[key], truth)
                for key, truth in zip(parameters, truths)
            ]
    return result.plot_corner(parameters=parameters, dpi=dpi, **kwargs)


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
        outdir, label_prefix, nlive=100, npool=1, simple_priors=None,
        **sampler_kwargs):
    """
    Run nested sampling for full lensed vs simple-lensed models on the same data.
    Returns dict with log evidences and Bayes factors.
    """
    duration = ifos[0].strain_data.duration
    sampling_frequency = ifos[0].strain_data.sampling_frequency
    if simple_priors is None:
        simple_priors = SimpleLensedPriorDict(duration=duration)

    geocent_time = injection_parameters['geocent_time']
    lensed_priors['geocent_time'] = geocent_time
    simple_priors['geocent_time'] = geocent_time
    lensed_wfg = make_waveform_generator(
        lensed_source_model, duration, sampling_frequency)
    simple_wfg = make_waveform_generator(
        general_lensed_binary_black_hole, duration, sampling_frequency)

    lensed_likelihood = _make_likelihood(ifos, lensed_wfg, lensed_priors)
    simple_likelihood = _make_likelihood(ifos, simple_wfg, simple_priors)

    result_lensed = run_pe(
        lensed_likelihood, lensed_priors, outdir, f'{label_prefix}_lensed',
        nlive=nlive, npool=npool, injection_parameters=injection_parameters,
        **sampler_kwargs,
    )
    result_simple = run_pe(
        simple_likelihood, simple_priors, outdir, f'{label_prefix}_simple',
        nlive=nlive, npool=npool, injection_parameters=injection_parameters,
        **sampler_kwargs,
    )

    log_bf_lensed_vs_simple = (
        result_lensed.log_evidence - result_simple.log_evidence)
    return {
        'log_evidence_lensed': result_lensed.log_evidence,
        'log_evidence_lensed_err': result_lensed.log_evidence_err,
        'log_evidence_simple': result_simple.log_evidence,
        'log_evidence_simple_err': result_simple.log_evidence_err,
        'log_bf_lensed_vs_simple': log_bf_lensed_vs_simple,
        'log10_bf_lensed_vs_simple': log_bf_lensed_vs_simple / np.log(10),
        'log_bf_signal_vs_noise_lensed': result_lensed.log_bayes_factor,
        'log_bf_signal_vs_noise_simple': result_simple.log_bayes_factor,
        'result_lensed': result_lensed,
        'result_simple': result_simple,
    }


def newton_search_required_snr(
        injection_parameters, model='agn', target_log10_bf=2.0, n_steps=2,
        nlive=50, outdir='outdir', label='newton',
        duration=DEFAULT_SEGMENT_DURATION, sampling_frequency=2048.0, y_Eins=None):
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
        priors = GenericLensedPriorDict(duration=duration)
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
        log10_bf = comparison['log10_bf_lensed_vs_simple']
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
        'log10_bf': comparison['log10_bf_lensed_vs_simple'],
    }


def save_comparison_json(results, path):
    """Save model comparison results (without Result objects) to JSON."""
    serializable = {k: v for k, v in results.items()
                    if not k.startswith('result_')}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
