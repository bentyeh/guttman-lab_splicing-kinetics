import os
import sys
import argparse
import numpy as np
import pickle

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
    res = fitting.iterative_grid_search(
        fitting.loss_sse,
        np.log10(bounds),
        num=4,
        max_depth=5,
        args=('spliced_fraction', data_time, data, pos_intron, gene_length),
        kwargs=dict(n=n, kwargs=dict(use_tqdm=True, log10=True, use_pool=False)),
        use_pool=True, use_tqdm=True,
        callback=fitting.callback_iter,
        kwargs_callback=dict(log10=True))
    with open(os.path.join(dir_sfs, f'fit-igs-{index}.pkl'), 'wb') as f:
        pickle.dump(res)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit transcription and splicing parameters using iterative grid search.')
    parser.add_argument('index', type=int, help='parameter set index')
    parser.add_argument('--n', type=int, default=30,
                        help='number of simulations to average over')
    args = parser.parse_args()
    main(args.n, args.index)