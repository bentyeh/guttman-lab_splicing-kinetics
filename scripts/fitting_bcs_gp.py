import os
import sys
import argparse
import numpy as np
import pickle
import time
import skopt

dir_project = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
sys.path.append(os.path.join(dir_project, 'modules'))
import fitting

dir_bcs = os.path.join(dir_project, 'data_aux', 'sim_bond_counts')

# mouse Actb main isoform (ENSMUST00000100497.10)
gene_length = 3640
pos_intron = np.array(
    [[ 828, 1135, 1669, 2363, 2579],
     [ 952, 1229, 2122, 2449, 3537]])
time_steps = np.array([0, 10, 15, 20, 25, 30, 45, 60, 75, 90, 120, 240]) * 60 + 600

def main(n, index):
    mean = np.load(os.path.join(dir_bcs, f'mean-{index}.npy'))
    data = mean[time_steps - 1, :]
    data_time = time_steps - 1
    bounds = np.log10(np.array([(0.01, 20), (5e-4, 0.5), (1e-3, 0.5), (10, 100)]))
    res = skopt.gp_minimize(
        lambda x: fitting.loss_sse(
            x,
            'count_per_splice_site',
            data_time,
            data,
            pos_intron,
            gene_length,
            n=n,
            kwargs=dict(log10=True, use_pool=True, use_tqdm=False)),
        bounds,      # the bounds on each dimension of x
        n_calls=1000,         # the number of evaluations of f
        n_initial_points=200,
        random_state=1234,
        verbose=True,   # the random seed
        callback=lambda res: fitting.callback_gp(res, log10=True))
    with open(os.path.join(dir_bcs, f'fit-gp-{index}.pkl'), 'wb') as f:
        pickle.dump(res, f)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit transcription and splicing parameters using Gaussian processes.')
    parser.add_argument('index', type=int, help='parameter set index')
    parser.add_argument('--n', type=int, default=25,
                        help='number of simulations to average over')
    args = parser.parse_args()
    main(args.n, args.index)