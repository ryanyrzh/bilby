#!/usr/bin/env python
"""Generic-lensed PE on AGN-injected data (model comparison suite)."""
import argparse
import os

from bilby.core.utils import random
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.gw.lensing import (
    DEFAULT_SEGMENT_DURATION,
    GenericLensedPriorDict,
    agn_lensed_binary_black_hole,
    build_injection_ifos,
    convert_agn_to_generic_lensed,
    general_lensed_binary_black_hole,
    generic_gwfast_to_bilby_lensed,
    make_waveform_generator,
    plot_lensing_corner,
    print_component_masses,
    reference_bilby_injection,
    run_pe,
    set_example_pe_time_prior,
)


def _default_npool():
    return int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))


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

    agn_injection = reference_bilby_injection()
    generic_gwfast = convert_agn_to_generic_lensed(agn_injection)
    injection_parameters = generic_gwfast_to_bilby_lensed(generic_gwfast, agn_injection)
    print('Converted generic parameters:')
    for key, value in sorted(injection_parameters.items()):
        print(f'  {key}: {value}')
    
    print(f"Time delay in seconds: {injection_parameters['delta_time']:.3f}")

    ifos, _ = build_injection_ifos(
        agn_injection,
        duration=args.duration,
        sampling_frequency=args.sampling_frequency,
        source_model=agn_lensed_binary_black_hole,
    )
    duration = ifos[0].strain_data.duration

    priors = GenericLensedPriorDict(duration=duration)
    set_example_pe_time_prior(priors, agn_injection['geocent_time'])
    wfg = make_waveform_generator(
        general_lensed_binary_black_hole, duration, args.sampling_frequency)
    likelihood = GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=wfg,
        priors=priors,
        distance_marginalization=False,
        phase_marginalization=False,
        time_marginalization=False,
    )
    result = run_pe(
        likelihood, priors, args.outdir, f'{args.label}_generic',
        injection_parameters=injection_parameters,
        nlive=args.nlive, npool=args.npool,
        dlogz=args.dlogz, maxcall=args.maxcall,
        sample=args.sample,
        check_point=args.check_point,
        check_point_plot=args.check_point_plot,
        resume=args.resume,
    )
    print_component_masses(result)
    if args.plot_corner:
        plot_lensing_corner(result, dpi=100)


if __name__ == '__main__':
    main()
