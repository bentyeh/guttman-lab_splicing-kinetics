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

def loss_sse(x, stats_fun, data_time, data, pos_intron, gene_length,
             n=20, bounds_error=int(1e9), bounds=None, kwargs=None):
    '''
    For a set of splicing parameters, compute the loss as the sum of squared errors between
    real data and data simulated using the set of splicing parameters.

    Args
    - x: np.ndarray. shape=(n,)
        Parameters
    - stats_fun: str or callable
        If str: name of statistics function for simulating data.
            Current options: 'spliced_fraction' or 'count_per_splice_site'
        If callable: function for computing statistics from simulated transcripts.
            See stats_fun parameter of simulate.simulate_transcripts()
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
        Defaults:
        - stats_kwargs=dict(time_steps=data_time)
        - aggfun=simulate.mean_nan
    
    Returns: float
    '''
    if type(stats_fun) == str:
        stats_fun = getattr(stats_transcripts, stats_fun)
    assert callable(stats_fun)
    kwargs_default = dict(
        stats_kwargs=dict(time_steps=data_time),
        aggfun=simulate.mean_nan)
    if kwargs is None:
        kwargs = dict()
    kwargs_default.update(kwargs)
    n_time_steps = data_time[-1] + 1
    if bounds is not None:
        assert bounds.shape[0] == len(x)
        total_bounds_error = \
            np.any((x < bounds[:, 0]) | (x > bounds[:, 1])) * bounds_error + \
            np.sum((np.maximum(bounds[:, 0] - x, 0) + np.maximum(x - bounds[:, 1], 0))) * bounds_error
        if total_bounds_error:
            print('bounds error:', total_bounds_error)
            return total_bounds_error
    _, sim = simulate.parallel_simulations(
        n,
        x, pos_intron, gene_length, n_time_steps,
        stats_fun=stats_fun,
        **kwargs_default)
    return np.sum((data - sim)**2)

def callback_scipy(xk, *args, log10=False, loss_fun=loss_sse, time_start=None, **kwargs):
    '''
    Print out current parameters and loss.

    Args
    - xk
        Current parameter vector
    - *args
        Positional arguments to loss_fun
    - log10: bool. default=False
        Whether parameters are given as log10-transformed values
    - loss_fun: callable. default=loss_sf
        Loss function
    - time_start: numeric. default=None
        Start time in seconds
    - **kwargs
        Keyword arguments to loss_fun
    
    Returns: None
    '''
    print('Current parameters:', 10**xk if log10 else xk)
    current_loss = loss_fun(xk, *args, **kwargs)
    print('Current loss:', current_loss)
    if time_start:
        print('Time elapsed (s):', time.time() - time_start)
    print('-----', flush=True)

def callback_gp(res, log10=False):
    '''
    Print out current and best parameters and losses.

    Args
    - res: scipy.optimize.OptimizeResult
    - log10: bool. default=False
        Whether parameters are given as log10-transformed values
    
    Returns: None
    '''
    print('Current parameters:', 10**res.x_iters[-1] if log10 else res.x_iters[-1])
    print('Current loss:', res.func_vals[-1])
    print('Best parameters:', 10**res.x if log10 else res.x)
    print('Best loss:', res.fun)
    print('-----')

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
    kwargs_callback=None,
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
    - kwargs_callback: dict. default=None
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
        All parameter value sets evaluated
    - losses: np.ndarray. shape=(z,)
        Loss for each parameter value set in `xs`
    
    <TODO> Bounds and previous_xs management
    - Do not need to recompute loss if x is in previous_xs
    - Handle case where len(xs) is not product(x0_per_param.shape)
    '''
    kwargs = dict() if kwargs is None else kwargs.copy()
    kwargs_callback = dict() if kwargs_callback is None else kwargs_callback.copy()
    n = bounds.shape[0]
    assert all([bounds[i, 0] <= bounds[i, 1] for i in range(n)])
    assert num >= 1
    assert max_depth >= 1
    if (previous_xs is not None) or (previous_losses is not None):
        assert len(previous_xs) == len(previous_losses)
        assert len(previous_xs) == len(np.unique(previous_xs, axis=0))

    x0_per_param = list(np.linspace(bounds[:, 0], bounds[:, 1], num=num, endpoint=True).T)  # shape: (n, num)
    x0_per_param = [np.sort(np.unique(x0)) for x0 in x0_per_param]
    x0_shapes = tuple(map(len, x0_per_param))
    x0s_all = list(map(list, itertools.product(*x0_per_param))) # list of length-n lists
    xs = [x0 for x0 in x0s_all if x0 not in previous_xs.tolist()] if previous_xs is not None else x0s_all
    xs = np.array(xs)

    status = []
    max_depth -= 1
    if len(xs) == 0:
        print('All parameter sets previous evaluated.', flush=True)
    else:
        if use_pool:
            try:
                n_cpus = len(os.sched_getaffinity(0))
            except:
                n_cpus = os.cpu_count()
            if use_tqdm:
                pbar = tqdm(total=len(xs))
                def update(*a):
                    pbar.update()
        print(f'Evaluating {len(xs)} parameter sets.', flush=True)
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
        if callback:
            callback(params_min, losses_min, bounds, max_depth, **kwargs_callback)

        # cache parameter sets and evalated losses
        if previous_losses is not None:
            if (np.abs(previous_losses.min() - losses_min) <= tol) or (previous_losses.min() < losses_min):
                status.append('convergence tolerance reached')
            previous_losses = np.concatenate((losses, previous_losses))
        else:
            previous_losses = losses
        previous_xs = np.vstack((xs, previous_xs)) if previous_xs is not None else xs

    idx_min_all = np.argmin(previous_losses)
    losses_min_all = previous_losses[idx_min_all]
    params_min_all = previous_xs[idx_min_all]

    # set status
    if max_depth == 0:
        status.append('max_depth reached')
    if num == 1:
        status.append('only 1 parameter sample requested')
    status = '; '.join(status)
    
    result_opt = dict(
        x=params_min_all,
        loss=losses_min_all,
        bounds=bounds,
        depth_remaining=max_depth,
        status=status,
        xs=previous_xs,
        losses=previous_losses)

    # some stopping criteria met
    if status != '':
        return result_opt

    # generate new bounds for next iteration of grid search
    previous_xs_list = previous_xs.tolist()
    losses_long = np.array([previous_losses[previous_xs_list.index(x)] for x in x0s_all])
    losses_wide = losses_long.reshape(x0_shapes)
    losses_pass = losses_wide <= np.quantile(losses_long, greediness)
    label, n_features = scipy.ndimage.label(losses_pass)
    slices = scipy.ndimage.find_objects(label)
    print(f'Generating {len(slices)} new subgrids.', flush=True)
    for slice_set in slices:
        # optional: exclude any slice that does not include the current min
        # if not np.any(losses_pass[slices[i]] <= losses_min):
        #     continue
        bounds_new = np.zeros_like(bounds)
        for i in range(n):
            bounds_new[i, :] = x0_per_param[i][[slice_set[i].start, slice_set[i].stop - 1]]
            if (bounds_new[i, 0] == bounds_new[i, 1]) and (bounds[i, 0] != bounds[i, 1]):
                if slice_set[i].start > 0:
                    bounds_new[i, 0] = (x0_per_param[i][slice_set[i].start - 1] + x0_per_param[i][slice_set[i].start]) / 2
                if slice_set[i].stop < x0_shapes[i]:
                    bounds_new[i, 1] = (x0_per_param[i][slice_set[i].stop - 1] + x0_per_param[i][slice_set[i].stop]) / 2
        result = iterative_grid_search(
            fun,
            bounds_new,
            args=args,
            kwargs=kwargs,
            num=num,
            tol=tol,
            max_depth=max_depth,
            greediness=greediness,
            use_pool=use_pool,
            use_tqdm=use_tqdm,
            callback=callback,
            kwargs_callback=kwargs_callback,
            previous_xs=result_opt['xs'],
            previous_losses=result_opt['losses'])
        result_opt['xs'] = result['xs']
        result_opt['losses'] = result['losses']
        if result['loss'] < losses_min_all:
            result_opt.update(result)
    return result_opt

def main():
    gene_length = 3640
    pos_intron = np.array(
        [[ 828, 1135, 1669, 2363, 2579],
         [ 952, 1229, 2122, 2449, 3537]])
    time_steps = np.array([0, 10]) * 60 + 600 # , 15, 20]) * 60 + 600 # , 25, 30, 45, 60, 75, 90, 120, 240]) * 60 + 600
    mean_fillna0 = np.load(os.path.join(dir_project, 'data_aux', 'sim_spliced_fraction', 'mean_fillna-0.npy'))
    params = np.load(os.path.join(dir_project, 'data_aux', 'sim_spliced_fraction', 'params.npy'))
    print('target parameters:', params[0])
    data = mean_fillna0[time_steps - 1, :]
    data_time = time_steps - 1
    bounds_np = np.array([(0.01, 1), (5e-3, 0.5), (5e-3, 0.5), (10, 50)])

    res = iterative_grid_search(
        loss_sf,
        np.log10(bounds_np),
        num=3,
        max_depth=3,
        args=(data_time, data, pos_intron, gene_length),
        kwargs=dict(n=20, kwargs=dict(use_tqdm=True, log10=True, use_pool=False)),
        use_pool=False, use_tqdm=False,
        callback=callback_iter)
    return res

if __name__ == '__main__':
    print(main())