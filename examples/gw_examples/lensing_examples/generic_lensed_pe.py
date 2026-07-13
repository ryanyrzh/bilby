#!/usr/bin/env python
"""
Parameter estimation on an injected generic lensed BBH signal.

Local smoke test: use --nlive 50. For production runs on a cluster use --nlive 1000+.
"""
import argparse

from bilby.core.utils import random
from bilby.gw.lensing import (
    GenericLensedPriorDict,
    build_injection_ifos,
    convert_agn_to_generic_lensed,
    general_lensed_binary_black_hole,
    generic_gwfast_to_bilby_lensed,
    reference_bilby_injection,
    run_pe,
)
from bilby.gw.likelihood import GravitationalWaveTransient

random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='outdir')
parser.add_argument('--label', default='generic_lensed_pe')
parser.add_argument('--nlive', type=int, default=50)
parser.add_argument('--duration', type=float, default=4.0)
parser.add_argument('--sampling-frequency', type=float, default=2048.0)
args = parser.parse_args()

agn_injection = reference_bilby_injection()
generic_gwfast = convert_agn_to_generic_lensed(agn_injection)
injection_parameters = generic_gwfast_to_bilby_lensed(generic_gwfast, agn_injection)

ifos, wfg = build_injection_ifos(
    injection_parameters,
    duration=args.duration,
    sampling_frequency=args.sampling_frequency,
    source_model=general_lensed_binary_black_hole,
)

priors = GenericLensedPriorDict()
priors['geocent_time'] = injection_parameters['geocent_time']

likelihood = GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=wfg,
    priors=priors,
    distance_marginalization=False,
    phase_marginalization=False,
    time_marginalization=False,
)

result = run_pe(
    likelihood, priors, args.outdir, args.label,
    nlive=args.nlive, injection_parameters=injection_parameters,
)
result.plot_corner()
