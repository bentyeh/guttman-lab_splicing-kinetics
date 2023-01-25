import os
import sys
import argparse
import numpy as np
import pickle
import time

dir_project = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
sys.path.append(os.path.join(dir_project, 'modules'))
import fitting

dir_sfs = os.path.join(dir_project, 'data_aux', 'sim_spliced_fraction')

# mouse Actb main isoform (ENSMUST00000100497.10)
gene_length = 3640
pos_intron = np.array(
    [[ 828, 1135, 1669, 2363, 2579],
     [ 952, 1229, 2122, 2449, 3537]])
time_steps = np.array([0, 10, 15, 20, 25, 30, 45, 60, 75, 90, 120, 240]) * 60 + 600

def main(n, index):
    mean_fillna = np.load(os.path.join(dir_sfs, f'mean_fillna-{index}.npy'))
    data = mean_fillna[time_steps - 1, :]
    data_time = time_steps - 1
    bounds = np.array([(0.01, 20), (5e-4, 0.5), (1e-3, 0.5), (10, 100)])
    x0 = np.array([np.mean(bound) for bound in np.log10(bounds)])
    time_start = time.time()
    res = scipy.optimize.minimize(
        fitting.loss_sse,
        x0,
        args=('count_per_splice_site', data_time, data, pos_intron, gene_length,
              n, int(1e9), None, kwargs=dict(log10=True, use_tqdm=True, use_pool=True)),
        method='L-BFGS-B',
        bounds=bounds,
        # options={'maxiter': 5, 'eps': 1e-1},
        callback=lambda xk: fitting.callback_scipy(
            xk,
            'count_per_splice_site', data_time, data, pos_intron, gene_length,
            log10=True,
            time_start=time_start,
            n=n,
            kwargs=dict(log10=True, use_tqdm=True, use_pool=True)))
    with open(os.path.join(dir_sfs, f'fit-lbfgsb-{index}.pkl'), 'wb') as f:
        pickle.dump(res)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit transcription and splicing parameters using iterative grid search.')
    parser.add_argument('index', type=int, help='parameter set index')
    parser.add_argument('--n', type=int, default=30,
                        help='number of simulations to average over')
    args = parser.parse_args()
    main(args.n, args.index)