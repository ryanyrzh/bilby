#!/usr/bin/env python
"""
Gather fixed-binary AGN / generic / simple PE results and report Bayes factors
vs simple:

    log BF(agn vs simple)     = log Z_agn_disk - log Z_simple_disk
    log BF(generic vs simple) = log Z_generic_disk - log Z_simple_disk
"""
import argparse
import os

import numpy as np
from bilby.core.result import read_in_result
from bilby.gw.lensing import save_comparison_json


def _vs_simple(result, result_simple):
    log_bf = result.log_evidence - result_simple.log_evidence
    return {
        'log_evidence_lensed': result.log_evidence,
        'log_evidence_lensed_err': result.log_evidence_err,
        'log_evidence_simple': result_simple.log_evidence,
        'log_evidence_simple_err': result_simple.log_evidence_err,
        'log_bf_lensed_vs_simple': log_bf,
        'log10_bf_lensed_vs_simple': log_bf / np.log(10),
        'log_bf_signal_vs_noise_lensed': result.log_bayes_factor,
        'log_bf_signal_vs_noise_simple': result_simple.log_bayes_factor,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='outdir')
    parser.add_argument('--label', default='')
    args = parser.parse_args()

    results = {
        name: read_in_result(outdir=args.outdir, label=f'{args.label}_{name}_disk')
        for name in ('agn', 'generic', 'simple')
    }

    summary = {
        'agn': _vs_simple(results['agn'], results['simple']),
        'generic': _vs_simple(results['generic'], results['simple']),
    }

    save_comparison_json(
        summary, os.path.join(args.outdir, f'{args.label}_disk_summary.json'))
    print('Fixed-binary model comparison summary:')
    for model, vals in summary.items():
        print(f'  {model}: {vals}')


if __name__ == '__main__':
    main()
