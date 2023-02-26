'''
Simulate splicing bond counts for mouse Actb gene

Assumes the following project directory structure:
- project directory/
  - data_aux/
  - modules/
    - simulate.py
    - stats_transcripts.py
  - scripts/
    - simulate_means.py (this file)
'''

import os
import sys
import argparse
import numpy as np

dir_project = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
sys.path.append(os.path.join(dir_project, 'modules'))
import simulate
import stats_transcripts
import utils_genomics

dir_results = os.path.join(dir_project, 'data_aux', 'sim')
os.makedirs(dir_results, exist_ok=True)

# mouse Actb main isoform (ENSMUST00000100497.10)
gene_length = 3640
pos_intron = np.array(
    [[ 828, 1135, 1669, 2363, 2579],
     [ 952, 1229, 2122, 2449, 3537]])
pos_exon = utils_genomics.pos_intron_to_pos_exon(pos_intron, gene_length)
time_steps = np.array([0, 10, 15, 20, 25, 30, 45, 60, 75, 90, 120, 240]) * 60 + 600

file_params = os.path.join(dir_results, 'params.npy')
if not os.path.exists(file_params):
    params = []
    for k_init in (0.01, 0.1, 1, 10): # transcripts / sec
        for k_decay in (0.0001, 0.001, 0.01, 0.1): # 1 / sec
            for k_splice in (0.0005, 0.005, 0.05, 0.5): # 1 / sec
                for k_elong in (25, 50, 75): # nt / sec
                    params.append((k_init, k_decay, k_splice, k_elong))
    params = np.array(params)
    np.save(file_params, params)

def main(n, index):
    params = np.load(file_params)
    time_points, agg_stats = simulate.parallel_simulations(
        n,
        params[index],
        pos_intron,
        gene_length,
        time_steps[-1],
        multi_stats=True,
        stats_fun=stats_transcripts.multi_stats,
        stats_kwargs={'stats_kwargs': {
            'junction_counts': {'pos_exon': pos_exon},
            'spliced_fraction': None,
            'splice_site_counts': None}},
        aggfun=simulate.mean_nan,
        alt_splicing=False,
        seed=0)
    for name, array in agg_stats.items():
        file = os.path.join(dir_results, f'{name}-{index}.npy')
        np.save(file, array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate transcription.')
    parser.add_argument('index', type=int, help='parameter set index')
    parser.add_argument('--n', type=int, default=250,
                        help='number of simulations to average over')
    args = parser.parse_args()
    main(args.n, args.index)
