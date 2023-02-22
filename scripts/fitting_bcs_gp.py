import os
import sys
import argparse
import numpy as np
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

def main(n, index, n_calls=1000, n_initial_points=200):
    mean = np.load(os.path.join(dir_bcs, f'mean-{index}.npy'))
    data_time = time_steps - 1
    data = mean[data_time, :]
    bounds = np.log10(np.array([(0.01, 20), (5e-4, 0.5), (1e-3, 0.5), (10, 100)]))
    res = skopt.gp_minimize(
        lambda x: fitting.loss_sse(
            x,
            'splice_site_counts',
            data_time,
            data,
            pos_intron,
            gene_length,
            n=n,
            kwargs=dict(log10=True, use_pool=True, use_tqdm=False)),
        bounds,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=1234,
        verbose=True,
        callback=lambda res: fitting.callback_gp(res, log10=True))

    # 1. Delete the objective and callback functions (loss and callback) prior to dumping
    #    since lambda functions are not serializable. See
    #    https://scikit-optimize.github.io/stable/auto_examples/store-and-load-results.html
    # 2. Delete all models (which take up a lot of memory and consequently disk space)
    #    except the last model, which can be used for visualization (see skopt.plots)
    del res.specs['args']['func'], res.specs['args']['callback'], res.models[:-1]
    skopt.dump(res, os.path.join(dir_bcs, f'fit-gp-{index}.pkl.gz'), compress=9)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit transcription and splicing parameters using Gaussian processes.')
    parser.add_argument('index', type=int, help='parameter set index')
    parser.add_argument('--n', type=int, default=25,
                        help='number of simulations to average over')
    parser.add_argument('--n_calls', type=int, default=1000,
                        help='maxinum number of loss function evaluations')
    parser.add_argument('--n_initial_points', type=int, default=200,
                        help='number of initial random points to evaluate; see skopt.gp_minimize')
    args = parser.parse_args()
    main(args.n, args.index, args.n_calls, args.n_initial_points)
