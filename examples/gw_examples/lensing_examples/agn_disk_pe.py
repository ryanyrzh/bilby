#!/usr/bin/env python
"""
Recover AGN disk parameters with the binary held fixed.

Samples only R_orbit, log10_M_lz, and src_pos. Every other CBC parameter is
pinned to the AGN injection. Coalescence time stays free in a narrow window:
the lensing delay cannot absorb a millisecond time-origin offset.
"""
import argparse
import os

import numpy as np

from bilby.core.utils import random
from bilby.gw.conversion import (
    component_masses_to_chirp_mass,
    component_masses_to_mass_ratio,
)
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.gw.lensing import (
    AGNLensedPriorDict,
    DEFAULT_SEGMENT_DURATION,
    agn_lensed_binary_black_hole,
    build_injection_ifos,
    network_snr,
    plot_lensing_corner,
    reference_bilby_injection,
    run_pe,
    set_example_pe_time_prior,
)

DISK_PARAMETERS = ('R_orbit', 'log10_M_lz', 'src_pos')


def _default_npool():
    return int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))


def _fixed_binary_parameters(injection):
    """Sampled CBC values corresponding to the AGN injection."""
    fixed = dict(injection)
    if 'chirp_mass' not in fixed or 'mass_ratio' not in fixed:
        fixed['chirp_mass'] = component_masses_to_chirp_mass(
            injection['mass_1'], injection['mass_2'])
        fixed['mass_ratio'] = component_masses_to_mass_ratio(
            injection['mass_1'], injection['mass_2'])
    if 'chi_1' not in fixed:
        fixed['chi_1'] = injection['a_1'] * np.cos(injection.get('tilt_1', 0.0))
    if 'chi_2' not in fixed:
        fixed['chi_2'] = injection['a_2'] * np.cos(injection.get('tilt_2', 0.0))
    return fixed


def _pin_binary(priors, injection):
    """Replace every non-disk, non-time sampled parameter with the injection."""
    fixed = _fixed_binary_parameters(injection)
    for key in list(priors.non_fixed_keys):
        if key in DISK_PARAMETERS or key == 'geocent_time':
            continue
        if key not in fixed:
            raise KeyError(f'No injection value for sampled parameter {key}')
        priors[key] = float(fixed[key])
    priors.convert_floats_to_delta_functions()


def main():
    random.seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='outdir')
    parser.add_argument('--label', default='')
    parser.add_argument('--nlive', type=int, default=50)
    parser.add_argument('--npool', type=int, default=_default_npool(),
                        help='Dynesty worker processes '
                             '(default: SLURM_CPUS_PER_TASK or 1)')
    parser.add_argument('--duration', type=float, default=DEFAULT_SEGMENT_DURATION)
    parser.add_argument('--sampling-frequency', type=float, default=2048.0)
    parser.add_argument('--dlogz', type=float, default=None)
    parser.add_argument('--maxcall', type=int, default=None)
    parser.add_argument('--sample', default=None,
                        help="Dynesty sampling method (e.g. unif, acceptance-walk)")
    parser.add_argument('--no-check-point', dest='check_point', action='store_false')
    parser.add_argument('--no-check-point-plot', dest='check_point_plot',
                        action='store_false')
    parser.add_argument('--no-plot-corner', dest='plot_corner', action='store_false')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing *_resume.pickle if present')
    parser.set_defaults(check_point=True, check_point_plot=True, plot_corner=True)
    args = parser.parse_args()

    print(f'Using npool={args.npool}')

    injection_parameters = reference_bilby_injection()
    print('Injection parameters:')
    for key, value in sorted(injection_parameters.items()):
        print(f'  {key}: {value}')

    ifos, wfg = build_injection_ifos(
        injection_parameters,
        duration=args.duration,
        sampling_frequency=args.sampling_frequency,
        source_model=agn_lensed_binary_black_hole,
    )
    print(f'Network SNR: {network_snr(ifos, injection_parameters, wfg):.1f}')

    priors = AGNLensedPriorDict()
    set_example_pe_time_prior(priors, injection_parameters['geocent_time'])
    _pin_binary(priors, injection_parameters)
    print(f'Sampling: {list(priors.non_fixed_keys)}')
    print('Fixed:')
    for key in priors.fixed_keys:
        print(f'  {key}: {priors[key].peak}')

    likelihood = GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=wfg,
        priors=priors,
        distance_marginalization=False,
        phase_marginalization=False,
        time_marginalization=False,
    )
    result = run_pe(
        likelihood, priors, args.outdir, f'{args.label}_agn_disk',
        injection_parameters=injection_parameters,
        nlive=args.nlive, npool=args.npool,
        dlogz=args.dlogz, maxcall=args.maxcall,
        sample=args.sample,
        check_point=args.check_point,
        check_point_plot=args.check_point_plot,
        resume=args.resume,
    )
    if args.plot_corner:
        plot_lensing_corner(result, dpi=100)


if __name__ == '__main__':
    main()
