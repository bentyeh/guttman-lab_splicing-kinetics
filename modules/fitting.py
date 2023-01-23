import itertools
import multiprocessing as mp
import os
import sys
import time
import pdb
import numpy as np
import scipy.ndimage
from tqdm.auto import tqdm

dir_project = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
sys.path.append(os.path.join(dir_project, 'modules'))
import simulate
import stats_transcripts

def loss_sf(x, data_time, data, pos_intron, gene_length,
            n=20, bounds_error=100, bounds=None, kwargs=None,
            constraints=None):
    '''
    For a set of splicing parameters, compute the loss as the sum of squared errors between
    real data and data simulated using the set of splicing parameters.

    Args
    - x: np.ndarray. shape=(n,)
        Parameters
    - data_time. len=m
        Time points of data
    - data: np.ndarray. shape=(m, n_introns)
        Spliced fraction at specified time points
    - pos_intron: np.ndarray. shape=(2, n_introns)
        Coordinates of introns, 1-indexed. If gene isoform is intronless, pos_intron has shape (2, 0).
    - gene_length: int
        Number of nucleotides from transcription start site to transcription end site.
    - n: int. default=20
        Number of simulations to aggregate
    - bounds_error: int. default=100
        Loss to return when a parameter value exceeds a bound
    - bounds: np.ndarray. shape=(n, 2)
        Columns 0 and 1 give inclusive lower and upper bounds
    - kwargs: dict. default=None
        Additional keyword arguments to pass to simulate.parallel_simulations()
    
    Returns: float
    '''
    n_time_steps = data_time[-1] + 1
    if bounds is not None:
        assert bounds.shape[0] == len(x)
        total_bounds_error = \
            np.any((x < bounds[:, 0]) | (x > bounds[:, 1])) * bounds_error + \
            np.sum((np.maximum(bounds[:, 0] - x, 0) + np.maximum(x - bounds[:, 1], 0))) * bounds_error
        if total_bounds_error:
            print('bounds error:', total_bounds_error)
            return total_bounds_error
    if kwargs is None:
        kwargs = dict()
    _, sim = simulate.parallel_simulations(
        n,
        x, pos_intron, gene_length, n_time_steps,
        stats_fun=stats_transcripts.spliced_fraction, stats_kwargs=dict(time_steps=data_time),
        aggfun=lambda x: np.mean(np.nan_to_num(x, nan=1), axis=0),
        **kwargs)
    return np.sum((data - sim)**2)

# def callback_scipy(xk, *args, log10=False, loss_fun=loss_sf, time_start=None, **kwargs):
#     '''
#     Print out current parameters and loss.

#     Args
#     - xk
#         Current parameter vector
#     - *args
#         Positional arguments to loss_fun
#     - log10: bool. default=False
#         Whether parameters are given as log10-transformed values
#     - loss_fun: callable. default=loss_sf
#         Loss function
#     - time_start: numeric. default=None
#         Start time in seconds
#     - **kwargs
#         Keyword arguments to loss_fun
    
#     Returns: None
#     '''
#     print('Current parameters:', 10**xk if log10 else xk)
#     current_loss = loss_fun(xk, *args, **kwargs)
#     print('Current loss:', current_loss)
#     if time_start:
#         print('Time elapsed (s):', time.time() - time_start)
#     print('-----', flush=True)

# def callback_gp(res, log10=False):
#     '''
#     Print out current and best parameters and losses.

#     Args
#     - res: scipy.optimize.OptimizeResult
#     - log10: bool. default=False
#         Whether parameters are given as log10-transformed values
    
#     Returns: None
#     '''
#     print('Current parameters:', 10**res.x_iters[-1] if log10 else res.x_iters[-1])
#     print('Current loss:', res.func_vals[-1])
#     print('Best parameters:', 10**res.x if log10 else res.x)
#     print('Best loss:', res.fun)
#     print('-----')

def callback_iter(params_min, losses_min, bounds, max_depth, log10=False):
    print('Current parameters:', 10**params_min if log10 else params_min)
    print('Current loss:', losses_min)
    print('Current bounds:', bounds),
    print('Current depth remaining:', max_depth)
    print('-----')

def iterative_grid_search(
    fun,
    bounds,
    args=(),
    kwargs=None,
    num=4,
    tol=1e-4,
    max_depth=5,
    greediness=0.1,
    use_pool=True,
    use_tqdm=True,
    callback=callback_iter,
    previous_xs=None,
    previous_losses=None):
    '''
    Args
    - fun: callable
        The objective function to be minimized.
          fun(x, *args) -> float
        where x is a 1-D array with shape (n,)
    - bounds: np.ndarray. shape=(n, 2)
        Columns 0 and 1 give inclusive lower and upper bounds
    - args: tuple. default=()
        Extra arguments passed to the objective function.
    - kwargs: dict. default=None
        Extra keyword arguments passed to the objective function.
    - num: int. default=4
        Number of samples for each parameter to generate at each iteration
    - tol: float. default=1e-4
        Absolute convergence tolerance. When the next depth of parameter search yields objective values
        all within tol of the best previous parameters, return the best parameter set found.
    - max_depth: int. default=5
        Maximum depths to search before returning the best parameter set found
    - greediness: float. default=0.1
    - use_pool: bool. default=True
        Parallelize grid search using a Process pool.
    - use_tqdm: bool. default=True
    - callback: callable. default=callback_iter
        Called after each level.
        Args:
        - x: np.ndarray. shape=(n,)
        - loss: float
        - bounds
        - max_depth
    - previous_xs: np.ndarray. shape=(z, n)
    - previous_losses: np.ndarray. shape=(z,)
    
    Returns: res
    - x: np.ndarray. shape=(n,)
        Best parameters found
    - loss: float
        Best loss found
    - bounds: np.ndarray. shape=(n, 2)
        Bounds at the depth the best loss was found
    - depth_remaining: int
        Depth remaining at the depth the best loss was found
    - status: str
        Describes stopping criteria that was met
    - xs: np.ndarray. shape=(z, n)
    - losses: np.ndarray. shape=(z,)
    
    <TODO> Bounds and previous_xs management
    - Do not need to recompute loss if x is in previous_xs
    - Handle case where len(xs) is not product(x0_per_param.shape)
    '''
    kwargs = dict() if kwargs is None else kwargs.copy()
    n = bounds.shape[0]
    assert all([bounds[i, 0] <= bounds[i, 1] for i in range(n)])
    assert num >= 1    

    x0_per_param = list(np.linspace(bounds[:, 0], bounds[:, 1], num=num, endpoint=True).T)  # shape: (num, n)
    x0_per_param = [np.sort(np.unique(x0)) for x0 in x0_per_param]
    x0_shapes = tuple(map(len, x0_per_param))
    xs = np.array(list(itertools.product(*x0_per_param))) # shape: (num**n, n)
    previous_xs = np.vstack((xs, previous_xs)) if previous_xs is not None else xs

    if use_pool:
        try:
            n_cpus = len(os.sched_getaffinity(0))
        except:
            n_cpus = os.cpu_count()
        if use_tqdm:
            pbar = tqdm(total=np.prod(x0_shapes))
            def update(*a):
                pbar.update()
    if use_pool and n_cpus > 1:
        with mp.Pool(n_cpus) as p:
            results = [p.apply_async(fun, (x, *args), kwargs, callback=update if use_tqdm else None)
                       for x in xs]
            p.close()
            p.join()
        losses = np.array([result.get() for result in results])
    else:
        losses = np.array([fun(x, *args, **kwargs) for x in xs])

    idx_min = np.argmin(losses)
    losses_min = losses[idx_min]
    params_min = xs[idx_min]

    max_depth -= 1
    if callback:
        callback(params_min, losses_min, bounds, max_depth)

    status = []
    if max_depth <= 0:
        status.append('max_depth reached')
    if num == 1:
        status.append('only parameter sample requested')
    if previous_losses is not None:
        if (np.abs(previous_losses.min() - losses_min) <= tol) or (previous_losses.min() < losses_min):
            status.append('convergence tolerance reached')
        previous_losses = np.concatenate((losses, previous_losses))
        idx_min_all = np.argmin(previous_losses)
        losses_min_all = previous_losses[idx_min_all]
        params_min_all = previous_xs[idx_min_all]
    else:
        previous_losses = losses

    # some stopping criteria met
    if len(status) > 0:
        status = '; '.join(status)
        return dict(
            x=params_min_all,
            loss=losses_min_all,
            bounds=bounds,
            depth_remaining=max_depth,
            status=status,
            xs=previous_xs,
            losses=previous_losses)

    # generate new bounds for next iteration of grid search
    losses_wide = losses.reshape(x0_shapes)
    losses_pass = losses_wide <= np.quantile(losses, greediness)
    label, n_features = scipy.ndimage.label(losses_pass)
    slices = scipy.ndimage.find_objects(label)
    results = []
    for slice_set in slices:
        # optional: exclude any slice that does not include the current min
        # if not np.any(losses_pass[slices[i]] <= losses_min):
        #     continue
        bounds_new = np.zeros_like(bounds)
        for i in range(n):
            bounds_new[i, :] = x0_per_param[i][[slice_set[i].start, slice_set[i].stop - 1]]
            if (bounds_new[i, 0] == bounds_new[i, 1]) and (bounds[i, 0] != bounds[i, 1]):
                if slice_set[i].start > 0:
                    bounds_new[i, 0] = (bounds[i, slice_set[i].start - 1] + bounds[i, slice_set[i].start]) / 2
                if slice_set[i].stop < x0_shapes[i]:
                    bounds_new[i, 1] = (bounds[i, slice_set[i].stop - 1] + bounds[i, slice_set[i].stop]) / 2
        results.append(iterative_grid_search(
            fun,
            bounds_new,
            args=args,
            kwargs=kwargs,
            num=num,
            tol=tol,
            max_depth=max_depth,
            greediness=greediness,
            use_pool=use_pool,
            callback=callback,
            previous_xs=previous_xs,
            previous_losses=previous_losses))

    # combine results
    idx_min_labels = np.argmin([res['loss'] for res in results])
    return dict(
        x=results[idx_min_labels]['x'],
        loss=results[idx_min_labels]['loss'],
        bounds=results[idx_min_labels]['bounds'],
        depth_remaining=results[idx_min_labels]['depth_remaining'],
        status=results[idx_min_labels]['status'],
        xs=np.vstack([res['xs'] for res in results]),
        losses=np.concatenate([res['losses'] for res in results]))

def main():
    gene_length = 3640
    pos_intron = np.array(
        [[ 828, 1135, 1669, 2363, 2579],
         [ 952, 1229, 2122, 2449, 3537]])
    time_steps = np.array([0, 10]) * 60 + 600 # , 15, 20]) * 60 + 600 # , 25, 30, 45, 60, 75, 90, 120, 240]) * 60 + 600
    mean_fillna0 = np.load(os.path.join(dir_project, 'data_aux', 'sim_spliced_fraction', 'mean_fillna-0.npy'))
    data = mean_fillna0[time_steps - 1, :]
    data_time = time_steps - 1
    bounds_np = np.array([(0.01, 50), (5e-3, 0.5), (5e-3, 0.5), (5, 100)])

    res = iterative_grid_search(
        loss_sf,
        np.log10(bounds_np),
        num=2,
        max_depth=2,
        args=(data_time, data, pos_intron, gene_length),
        kwargs=dict(n=1, kwargs=dict(use_tqdm=True, log10=True, use_pool=False)),
        use_pool=False, use_tqdm=True,
        callback=callback_iter)
    return res

if __name__ == '__main__':
    main()