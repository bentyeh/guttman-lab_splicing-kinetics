'''
Simulate spliced fraction for mouse Actb gene

Assumes the following project directory structure:
- project directory/
  - data_aux/
  - modules/
    - simulate.py
    - stats_transcripts.py
  - scripts/
    - simulate_sfs.py (this file)
'''

import os
import sys
import argparse
import numpy as np

dir_project = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
sys.path.append(os.path.join(dir_project, 'modules'))
import simulate
import stats_transcripts

dir_results = os.path.join(dir_project, 'data_aux', 'sim_spliced_fraction')
os.makedirs(dir_results, exist_ok=True)

# mouse Actb main isoform (ENSMUST00000100497.10)
gene_length = 3640
pos_intron = np.array(
    [[ 828, 1135, 1669, 2363, 2579],
     [ 952, 1229, 2122, 2449, 3537]])
time_steps = np.array([0, 10, 15, 20, 25, 30, 45, 60, 75, 90, 120, 240]) * 60 + 600

file_params = os.path.join(dir_results, 'params.npy')
if not os.path.exists(file_params):
    params = []
    for k_init in (0.1, 1, 10): # transcripts / sec
        for k_decay in (0.001, 0.01, 0.1): # 1 / sec
            for k_splice in (0.005, 0.05, 0.5): # 1 / sec
                for k_elong in (25, 50, 75): # nt / sec
                    params.append((k_init, k_decay, k_splice, k_elong))
    params = np.array(params)
    np.save(file_params, params)

def aggfun(x):
    '''
    Arg: shape=(n_simulations, time_point, n_introns)
    '''
    mean = np.mean(x, axis=0)
    mean_fillna = np.mean(np.nan_to_num(x, nan=1), axis=0)
    return mean, mean_fillna

def main(n, index):
    params = np.load(file_params)
    time_points, (mean, mean_fillna) = simulate.parallel_simulations(
        n,
        params[index],
        pos_intron,
        gene_length,
        time_steps[-1],
        stats_fun=stats_transcripts.spliced_fraction,
        aggfun=aggfun,
        alt_splicing=False,
        seed=0)
    file_mean = os.path.join(dir_results, f'mean-{index}.npy')
    file_mean_fillna = os.path.join(dir_results, f'mean_fillna-{index}.npy')
    np.save(file_mean, mean)
    np.save(file_mean_fillna, mean_fillna)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate transcription.')
    parser.add_argument('index', type=int, help='parameter set index')
    parser.add_argument('--n', type=int, default=100,
                        help='number of simulations to average over')
    args = parser.parse_args()
    main(args.n, args.index)