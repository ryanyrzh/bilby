#!/usr/bin/env python
"""
Compare unlensed, generic-lensed, and AGN-lensed models via nested sampling.

Bayes factors are computed from log-evidence differences:
    log BF = log Z_lensed - log Z_unlensed
"""
import argparse
import os

from bilby.core.utils import random
from bilby.gw.lensing import (
    AGNLensedPriorDict,
    GenericLensedPriorDict,
    agn_lensed_binary_black_hole,
    build_injection_ifos,
    compare_models,
    general_lensed_binary_black_hole,
    reference_bilby_injection,
    save_comparison_json,
)

random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='outdir')
parser.add_argument('--label', default='model_comparison')
parser.add_argument('--nlive', type=int, default=50)
parser.add_argument('--duration', type=float, default=4.0)
parser.add_argument('--sampling-frequency', type=float, default=2048.0)
args = parser.parse_args()

injection_parameters = reference_bilby_injection()
ifos, _ = build_injection_ifos(
    injection_parameters,
    duration=args.duration,
    sampling_frequency=args.sampling_frequency,
    source_model=agn_lensed_binary_black_hole,
)

agn_comparison = compare_models(
    ifos, injection_parameters,
    agn_lensed_binary_black_hole, AGNLensedPriorDict(),
    outdir=args.outdir, label_prefix=f'{args.label}_agn', nlive=args.nlive,
)

generic_comparison = compare_models(
    ifos, injection_parameters,
    general_lensed_binary_black_hole, GenericLensedPriorDict(),
    outdir=args.outdir, label_prefix=f'{args.label}_generic', nlive=args.nlive,
)

summary = {
    'agn': {k: v for k, v in agn_comparison.items() if not k.startswith('result_')},
    'generic': {k: v for k, v in generic_comparison.items() if not k.startswith('result_')},
}
save_comparison_json(summary, os.path.join(args.outdir, f'{args.label}_summary.json'))
print('Model comparison summary:')
for model, vals in summary.items():
    print(f'  {model}: {vals}')
