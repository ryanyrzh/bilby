#!/usr/bin/env python
"""
Compute required network SNR to reach a target Bayes factor (lensed vs unlensed)
on a Mc x R_orbit grid using full nested-sampling evidence.

Plot layout mirrors gwfast production/compute_SNR_threshold_sddr_Mc.py.
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D

from bilby.gw.lensing import build_agn_injection, newton_search_required_snr

parser = argparse.ArgumentParser(description='SNR threshold from nested-sampling Bayes factors.')
parser.add_argument('--nR', type=int, default=3, help='Grid cells in R_orbit.')
parser.add_argument('--nMc', type=int, default=3, help='Grid cells in chirp mass.')
parser.add_argument('--cores', type=int, default=1, help='Parallel workers.')
parser.add_argument('--model', type=str, default='agn', choices=['agn', 'generic'])
parser.add_argument('--logB', type=float, default=2.0, help='Target log10(BF).')
parser.add_argument('--steps', type=int, default=2, help='Newton steps on luminosity_distance.')
parser.add_argument('--nlive', type=int, default=50, help='Dynesty live points.')
parser.add_argument('--duration', type=float, default=4.0)
parser.add_argument('--sampling-frequency', type=float, default=2048.0)
parser.add_argument('--label', type=str, default=None)
parser.add_argument('--outdir', type=str, default='output')
parser.add_argument('--plotdir', type=str, default='plots')
args = parser.parse_args()

Y_EINS_1 = 0.99
Y_EINS_2 = 0.2


def _grid_worker(RMc_tuple_sublist):
    results_1, results_2 = [], []
    worker = partial(
        newton_search_required_snr,
        model=args.model,
        target_log10_bf=args.logB,
        n_steps=args.steps,
        nlive=args.nlive,
        outdir=args.outdir,
        duration=args.duration,
        sampling_frequency=args.sampling_frequency,
    )
    for R_orbit, Mc in RMc_tuple_sublist:
        inj_1 = build_agn_injection(Mc, R_orbit, Y_EINS_1)
        inj_2 = build_agn_injection(Mc, R_orbit, Y_EINS_2)
        results_1.append(worker(inj_1, label=f'Mc{Mc:.1f}_R{R_orbit:.1f}', y_Eins=Y_EINS_1)['required_snr'])
        results_2.append(worker(inj_2, label=f'Mc{Mc:.1f}_R{R_orbit:.1f}', y_Eins=Y_EINS_2)['required_snr'])
    return np.array(results_1), np.array(results_2)


if __name__ == '__main__':
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    mc_array = np.linspace(10, 100, args.nMc)
    r_array = np.geomspace(10, 2e3, args.nR)
    r_mesh, mc_mesh = np.meshgrid(r_array, mc_array, indexing='xy')
    grid = np.vstack([r_mesh.flatten(), mc_mesh.flatten()]).T

    with Pool(args.cores) as pool:
        chunks = np.array_split(grid, args.cores)
        batch = pool.map(_grid_worker, chunks)

    snr_1 = np.concatenate([b[0] for b in batch]).reshape(args.nMc, args.nR)
    snr_2 = np.concatenate([b[1] for b in batch]).reshape(args.nMc, args.nR)

    label_suffix = f'_{args.label}' if args.label else ''
    np.savez(
        os.path.join(args.outdir,
                      f'result_Mc{args.nMc}_R{args.nR}_{args.model}_logB{args.logB:g}_{args.steps}steps{label_suffix}.npz'),
        Mc=mc_mesh.flatten(),
        R_orbit=r_mesh.flatten(),
        y_Eins_1=Y_EINS_1,
        y_Eins_2=Y_EINS_2,
        snr_1=snr_1.flatten(),
        snr_2=snr_2.flatten(),
    )

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), constrained_layout=True)
    log10_snr = np.log10(snr_1)
    ax.set_facecolor('lightgrey')
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='lightgrey')
    midpoint = (np.nanmax(log10_snr) + np.nanmin(log10_snr)) / 2
    half_range = (np.nanmax(log10_snr) - np.nanmin(log10_snr)) / 2 * 1.05
    norm = colors.TwoSlopeNorm(
        vmin=midpoint - half_range, vcenter=midpoint, vmax=midpoint + half_range)
    im = ax.pcolormesh(r_array, mc_array, log10_snr, cmap=cmap, norm=norm, shading='nearest')

    legend_handles = []
    for snr_grid, color, contour_label in zip(
            [snr_1, snr_2], ['black', 'white'],
            [rf'$y = {Y_EINS_1:g}$', rf'$y = {Y_EINS_2:g}$']):
        cont_snrs = [0.5, 1.0, 1.5, 2.0, 2.5]
        cont = ax.contour(r_array, mc_array, np.log10(snr_grid), colors=[color], levels=cont_snrs)
        labels = {lvl: f'{snr}' for lvl, snr in zip(cont.levels, cont_snrs)}
        ax.clabel(cont, fmt=labels, fontsize=10)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=contour_label))
    ax.legend(handles=legend_handles, loc='lower left', fontsize=10)
    ax.set_xscale('log')
    ax.set_xlabel(r'$R_{\rm orbit}\,/\,R_S$')
    ax.set_ylabel(r'$\mathcal{M}_c\,/\,M_\odot$')
    fig.colorbar(im, ax=ax, label=r'$\log_{10}(\rho_{\rm req})$')
    fig.savefig(os.path.join(
        args.plotdir,
        f'snr_{args.model}_logB{args.logB:g}_{args.steps}steps{label_suffix}_RMcplot.pdf'))
