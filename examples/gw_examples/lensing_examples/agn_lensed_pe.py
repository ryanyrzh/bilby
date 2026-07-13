#!/usr/bin/env python
"""
Parameter estimation on an injected AGN-lensed BBH signal.

Local smoke test: use --nlive 50. For production runs on a cluster use --nlive 1000+.
"""
import argparse

import bilby
from bilby.core.utils import random
from bilby.gw.lensing import (
    AGNLensedPriorDict,
    agn_lensed_binary_black_hole,
    build_injection_ifos,
    reference_bilby_injection,
    run_pe,
)
from bilby.gw.likelihood import GravitationalWaveTransient

random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='outdir')
parser.add_argument('--label', default='agn_lensed_pe')
parser.add_argument('--nlive', type=int, default=50)
parser.add_argument('--duration', type=float, default=4.0)
parser.add_argument('--sampling-frequency', type=float, default=2048.0)
args = parser.parse_args()

injection_parameters = reference_bilby_injection()
ifos, wfg = build_injection_ifos(
    injection_parameters,
    duration=args.duration,
    sampling_frequency=args.sampling_frequency,
    source_model=agn_lensed_binary_black_hole,
)

priors = AGNLensedPriorDict()
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
