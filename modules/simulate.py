'''
Simulate transcription and splicing
'''

import itertools
import multiprocessing as mp
import os
import time

import numpy as np
from tqdm.auto import tqdm, trange

def mutually_exclusive_splicing(intervals):
    '''
    Generate square matrix A indicating mutually exclusive splicing events:
      A[i, j] indicates whether splicing of intron i precludes splicing of intron j.
    
    Note that A is not necessarily symmetric. This can occur if an intron is located
    completely within another intron (i.e., start2 > start1 and end2 < end1):
      ----------(intron1)----------------
         ---(intron2)---
    In this case, splicing intron1 precludes splicing of intron2, but not vice versa,
    so A[1, 2] = 1 (True), but A[2, 1] = 0 (False).
  
    This implementation only checks if intron2 is located completely within intron1 but
    is a bit simplistic. A more realistic implementation would check that intron2 is
    located completely within intron1 by a margin on both ends such that the sequences
    denoting the 5' and 3' splice sites of intron1 do not overlap intron2 at all.
  
    Arg(s)
    - itervals: iterable of (start, end), len=n_introns
        Coordinates of introns given by (start, end) where
          - end >= start
          - start and end are inclusive
          - introns are sorted by start position
        Can be a np.ndarray where rows are pairs: i.e., shape is (n_introns, 2)
  
    Returns
    - A: np.ndarray, shape=(n_introns, n_introns), dtype=bool
        Element i, j is True if splicing of intron i precludes splicing of intron j.
        Diagonal is 1. If `np.array_equal(A, np.eye(n_introns))` is True, then none of the introns
        are mutually exclusive.
    '''
    # check that introns are sorted by start position
    start_positions = [interval[0] for interval in intervals]
    assert np.all(start_positions[:-1] <= start_positions[1:])
    # check that start and end positions are appropriately given as end >= start
    assert all((end >= start for start, end in intervals))
  
    n_introns = len(intervals)
    A = np.eye(n_introns, dtype=bool)
    for i, j in itertools.combinations(range(n_introns), 2):
        start1, end1 = intervals[i]
        start2, end2 = intervals[j]
        if start2 <= end1:
            A[i, j] = 1
            if end2 >= end1:
                A[j, i] = 1
        if start1 == start2:
            A[j, i] = 1
    return A

def stats_raw(transcripts, t, stats=None, *, time_steps=None):
    '''
    Return all transcripts at requested time points.

    Arg(s)
    - transcripts: np.ndarray. shape=(<variable>, n_introns + 1). dtype=float
        The current set of transcripts at time `t`.
        - Each row represents a transcript
        - Column i (i = 0, ..., n_introns - 1): whether intron i is
            - np.nan: excluded due to splicing of a mutually exclusive intron
            - -1: spliced
            - between 0 and 1, inclusive: proportion of the intron currently transcribed and present
        - Column n_introns: position of the transcript, 1-indexed
    - t: int
        Current time step
    - stats: dict from int to np.ndarray. default=None
        Stores computed statistics. See Returns.
    - time_steps: list of int. default=None
        Time steps to include in return. If None, all time steps are included.

    Returns
    - stats: dict (int: np.ndarray)
        Keys: time steps
        Values: `transcripts` at time t
    '''
    if stats is None:
        stats = dict()
    if time_steps is None or t in time_steps:
        stats[t] = transcripts
    return stats

def simulate_transcripts(
    params,
    pos_intron, gene_length, n_time_steps, t_wash=600,
    stats_fun=stats_raw, stats_kwargs=None,
    log10=False, alt_splicing=True, rng=None, verbose=True):
    '''
    Simulate transcription, elongation, splicing, and decay of transcripts. Allows for alternative splicing where
    splicing of one intron may be mutually exclusive with splicing of another intron.

    Key assumptions in current implementation:
    - All transcripts share the same transcription start site and end site.
    - Transcription initiation is random (i.e., not bursting).
    - Elongation rate is constant over the entire length of the gene.
    - Splicing rate is constant for all introns.
      - This assumption can be avoided by separately simulating transcripts using 1 intron annotation at a time.
    - Splicing only occurs after transcription of the nucleotide past the 3' splice site.
    - Decay only occurs for fully-elongated transcripts.

    Arg(s)
    - Model parameters
      - params: np.ndarray. shape=(4,). dtype=float
        - k_init: Production rate (transcripts / time)
        - k_decay: Decay rate (1 / time), must be between 0 and 1
        - k_splice: Splicing rate (1 / time), must be between 0 and 1.
            Assumed to be constant for different splice junctions in the same gene
        - k_elong: Elongation velocity (nucleotides / time)
    - Experiment parameters
      - pos_intron: np.ndarray. shape=(2, n_introns)
          Coordinates of introns, 1-indexed. If gene isoform is intronless, pos_intron has shape (2, 0).
      - gene_length: int
          Number of nucleotides from transcription start site to transcription end site
      - t_wash: int. default=600
          Time at which transcription initiation stops
      - n_time_steps: int
          Number of time steps
    - Simulation parameters
      - stats_fun: callable. default=count_per_intron
        - Computes statistics from the simulated transcripts.
        - Called at each time step in the simulation. Must initiate the output variable `stats` the
          first time the function is called.
        - Must have signature `callable(transcripts, t, stats, **kwargs)`:
          - transcripts: np.ndarray. shape=(<variable>, n_introns + 1). dtype=float
              The current set of transcripts at time `t`.
              - Each row represents a transcript
              - Column i (i = 0, ..., n_introns - 1): whether intron i is
                - np.nan: excluded due to splicing of a mutually exclusive intron
                - -1: spliced
                - >= 0: proportion of the intron currently transcribed and present
                  - > 0 indicates that the first intron nucleotide has been incorporated
                  - 1 indicates that the last nucleotide of the intron has been incorporated
                  - > 1 indicates that the intron is present and that elongation has exceeded the 5' splice site
              - Column n_introns: position of the transcript, 1-indexed
          - t: int
              Current time step
          - stats: <variable>. Usually a dict (time step (int): statistics (np.ndarray))
              Stores computed statistics.
      - stats_kwargs: dict. default=None
        - Additional keyword arguments to stats_fun.
      - log10: bool, default=False
          Whether parameters are given as log10-transformed values
      - alt_splicing: bool, or np.ndarray of shape (n_introns, n_introns). default=True
          True: check for overlapping introns; only allow physically possible splicing combinations
          False: do not check for overlapping introns; assume all introns are spliced independently
          np.ndarray: square matrix such that element i, j denotes whether splicing of intron i
            would preclude splicing of intron j. See mutually_exclusive_splicing()
      - rng: np.random.Generator. default=None
          Random number generator. If None, uses np.random.default_rng().
      - verbose: bool. default=True
          Print out status messages every 15 seconds.
    
    Returns
    - stats: <variable>
        Computed statistics of the experiment. Type depends on `stats_fun`.
    '''
    # check argument values
    if log10:
        params = np.power(10, params)
    k_init, k_decay, k_splice, k_elong = params
    assert k_splice >= 0 and k_splice < 1
    assert k_decay >= 0 and k_decay < 1
    assert k_elong >= 0 and k_init >=0
    assert gene_length > 0

    if rng is None:
        rng = np.random.default_rng()
    
    if stats_kwargs is None:
        stats_kwargs = dict()

    # generate intermediate variables
    n_introns = pos_intron.shape[1]
    intron_lengths = pos_intron[1, :] - pos_intron[0, :] + 1
    arange_introns = np.arange(n_introns)

    if alt_splicing is True:
        alt_splicing = mutually_exclusive_splicing(pos_intron.T)
    if isinstance(alt_splicing, np.ndarray):
        assert alt_splicing.dtype == bool
        if np.array_equal(alt_splicing, np.eye(n_introns, dtype=bool)):
            alt_splicing = False
        else:
            # alt_splicing_nan: element i, j is np.nan if splicing of intron i precludes splicing of
            # intron j. Diagonal and all other elements have value 1.
            alt_splicing_nan = alt_splicing.astype(float)
            alt_splicing_nan[alt_splicing_nan == 1] = np.nan
            alt_splicing_nan[np.diag_indices(n_introns)] = 1
            alt_splicing_nan[alt_splicing_nan == 0] = 1

    # -- initialize transcripts --
    # - initial condition: elongating transcripts at the time of labeling initiation
    # - if no transcripts are initialized (i.e., due to chance or a super low transcription rate), `transcripts` remains
    #   a numpy.ndarray of shape (0, n_introns + 1)
    n_transcripts = rng.poisson(gene_length * k_init / k_elong)
    arange_transcripts = np.arange(n_transcripts)
    transcripts = np.zeros((n_transcripts, n_introns + 1), dtype=float)
    transcripts[:, -1] = rng.choice(gene_length, n_transcripts) + 1
    transcripts[:, :-1] = np.maximum((transcripts[:, [-1]] - pos_intron[[0], :] + 1) / intron_lengths[np.newaxis,], 0)
    
    # -- splice initial transcripts --
    log_prob_unspliced = np.log(1 - k_splice) * ((transcripts[:, [-1]] - pos_intron[1, :]) / k_elong)
    # `mask_to_splice`: array of shape (n_transcripts, n_introns)
    #    value of 1 denotes introns to splice
    #    value of np.nan denotes excluded intron
    mask_to_splice = (transcripts[:, :-1] > 1) & (np.log(rng.random(transcripts[:, :-1].shape)) > log_prob_unspliced)
    mask_to_splice = mask_to_splice.astype(float)
    if alt_splicing is not False:
        # Limit mutually exclusive splicing: implementation 1
        # - Pros
        #     - Vectorized over all transcripts
        #     - Order of splicing is randomly chosen for each transcript
        # - Cons
        #     - Loops through all introns
        #     - Difficult to check at the end of each loop whether any mutually exclusive splicing
        #       remains because mutual exclusion depends on the order of splicing, which is random

        # randomly choose order to splice mutually exclusive introns
        splice_order = rng.permuted(np.repeat(arange_introns.reshape(1, -1), n_transcripts, axis=0), axis=1).T
        for idx_introns in splice_order:
            mask_contains_spliced_intron = (transcripts[(arange_transcripts, idx_introns)] > 1) & \
                                           (mask_to_splice[(arange_transcripts, idx_introns)] == 1)
            if not mask_contains_spliced_intron.any():
                continue
            # for transcripts with a full intron to be spliced as given by splice_order, apply the
            # alt_splicing_nan mask to replace excluded introns with np.nan
            excluded_introns = alt_splicing_nan[idx_introns[mask_contains_spliced_intron], :]
            mask_to_splice[mask_contains_spliced_intron, :] *= excluded_introns

    # Update intron states
    transcripts[:, :-1][mask_to_splice == 1] = -1
    transcripts[:, :-1][np.isnan(mask_to_splice)] = np.nan

    # initialize counts of unspliced and spliced introns
    stats = stats_fun(transcripts, t=0, stats=None, **stats_kwargs)
    
    if verbose:
        time_start = time.time()
        # t = 0
        # def interval_status(interval):
        #     timer = threading.Timer(interval, interval_status, (interval,))
        #     timer.daemon = True
        #     timer.start()
        #     print(dict(params=params, t=t, n_transcripts=n_transcripts))
        # interval_status(1)
        # # see https://stackoverflow.com/a/3393759

    # simulation over time
    for t in range(1, n_time_steps):
        if verbose and t % 10 == 0:
            time_now = time.time()
            if time_now > time_start + 15:
                time_start = time_now
                print(dict(params=params, t=t, n_transcripts=n_transcripts))
        # decay
        mask_decay = (transcripts[:, -1] == gene_length) & (rng.random(n_transcripts) < k_decay)
        transcripts = transcripts[~mask_decay, :]
        n_transcripts = transcripts.shape[0]
        arange_transcripts = np.arange(n_transcripts)

        mask_to_splice = (transcripts[:, :-1] > 1) & (rng.random(transcripts[:, :-1].shape) < k_splice)
        mask_to_splice = mask_to_splice.astype(float)
        if alt_splicing is not False:
            splice_order = rng.permuted(np.repeat(arange_introns.reshape(1, -1), n_transcripts, axis=0), axis=1).T
            for idx_introns in splice_order:
                mask_contains_spliced_intron = (transcripts[(arange_transcripts, idx_introns)] > 1) & \
                                               (mask_to_splice[(arange_transcripts, idx_introns)] == 1)
                if not mask_contains_spliced_intron.any():
                    continue
                excluded_introns = alt_splicing_nan[idx_introns[mask_contains_spliced_intron], :]
                mask_to_splice[mask_contains_spliced_intron, :] *= excluded_introns
        transcripts[:, :-1][mask_to_splice == 1] = -1
        transcripts[:, :-1][np.isnan(mask_to_splice)] = np.nan

        # elongate
        v_int = int(k_elong) + (rng.random() > (k_elong - int(k_elong)))
        transcripts[:, -1] = np.minimum(transcripts[:, -1] + v_int, gene_length)

        # initiation
        if t < t_wash:
            n_new_transcripts = rng.poisson(k_init)
            if n_new_transcripts > 0:
                new_transcripts = np.zeros((n_new_transcripts, n_introns + 1), dtype=int)
                if np.minimum(v_int, gene_length) <= 1:
                    new_transcripts[:, -1] = 1
                else:
                    new_transcripts[:, -1] = rng.choice(np.minimum(v_int, gene_length) - 1,
                                                        n_new_transcripts) + 1
                transcripts = np.append(transcripts, new_transcripts, axis=0)
                n_transcripts = transcripts.shape[0]
        transcripts[:, :-1][transcripts[:, :-1] >= 0] = np.maximum(
            (transcripts[:, [-1]] - pos_intron[[0], :] + 1) / intron_lengths[np.newaxis,],
            0)[transcripts[:, :-1] >= 0]

        # compute statistics
        stats = stats_fun(transcripts, t, stats=stats, **stats_kwargs)

    return stats

def mean_nan(stats_all):
    '''
    Replace all NaN values with 1 and return mean taken over axis=0 (presumed to be parallel simulations)
    '''
    return np.mean(np.nan_to_num(stats_all, nan=1), axis=0)

def mean_and_var(stats_all):
    '''
    Return mean and variance taken over axis=0 (presumed to be parallel simulations)
    '''
    return np.mean(stats_all, axis=0), np.var(stats_all, axis=0)

def parallel_simulations(
    n,
    *args,
    multi_stats=False,
    seed=None,
    use_tqdm=True,
    use_pool=True,
    n_cpus=None,
    aggfun=None,
    **kwargs):
    '''
    Aggregate statistics over multiple simulations.

    Args
    - n: int
        Number of simulations
    - *args
        Positional arguments passed to `simulate_transcripts()`
    - multi_stats: bool. default=False
      - True: `stats` returned from each simulation is assumed to be of the form returned by
        stats_transcripts.wrap_wrap_multiple_stats():
          dict (function name (str): dict (time point (int): computed statistics (np.ndarray)))
      - False: `stats` returned from each simulation is assumed to be of the form
          dict (time point (int): computed statistics (np.ndarray))
    - seed: int. default=None
      - If an integer, the random number generator for the ith simulation (0-indexed) is seeded with `seed + i`.
      - Otherwise, the random number generator for each simulation is seeded with `None`.
    - use_tqdm: bool. default=True
        Show tqdm progress bar over the number of simulations
    - use_pool: bool. default=True
        Parallelize simulations using a Process pool.
    - n_cpus: int. default=None
        If None, the number of available CPUs is determined automatically.
    - aggfun: callable. default=None.
        Aggregation function over multiple simulations.
        - If None, returns a list of all simulations.
        - Otherwise, must have definition <aggfun>(stats_all) where `stats_all` is a NumPy array of shape:
          - (n, time_point) if `stats_fun` returns a scalar value at each time point
          - (n, time_point, *(shape)) if `stats_fun` returns a NumPy array of shape `(shape)` at each time point.
          - Example: stats_transcripts.count_per_splice_site returns an array of shape (n_introns, 3) at each time
            point, so the final output over 1 simulation is (time_point, n_introns, 3). Over `n` simulations, the shape
            becomes (n, time_point, n_introns, 3).
    - **kwargs: keyword arguments passed to `simulate_transcripts()`
      - Assumes that `stats_fun` returns a dict (int: np.ndarray) mapping from time point to computed statistics
      - Note that the "rng" key in kwargs is ignored, since it is assumed that each parallel run should have a different
        seed. See `seed` argument.

    Returns
    - time_points: np.ndarray
        Time points
    - <variable>
        The result of calling `aggfun` on a stack of outputs from each simulation.
    '''
    kwargs = kwargs.copy()
    if use_pool:
        if use_tqdm:
            pbar = tqdm(total=n)
            def update(*a):
                pbar.update()
        if n_cpus is None:
            try:
                n_cpus = len(os.sched_getaffinity(0))
            except:
                n_cpus = os.cpu_count()
    if use_pool and n_cpus > 1 and n > 1:
        results = []
        with mp.Pool(n_cpus) as p:
            for i in range(n):
                kwargs['rng'] = np.random.default_rng(seed + i) if isinstance(seed, int) else None
                results.append(
                    p.apply_async(
                        simulate_transcripts,
                        args,
                        kwargs.copy(),
                        callback=update if use_tqdm else None))
            p.close()
            p.join()
        stats_all = [result.get() for result in results]
    else:
        stats_all = [None] * n
        range_fun = trange if use_tqdm else range
        for i in range_fun(n):
            kwargs['rng'] = np.random.default_rng(seed + i) if isinstance(seed, int) else None
            stats_all[i] = simulate_transcripts(*args, **kwargs.copy())

    if multi_stats:
        # check stats function are consistent across simulations
        stats_fun_names = tuple(stats_all[0].keys())
        assert all((set(stats_fun_names) == set(stats.keys()) for stats in stats_all))

        # check time points are consistent across stats functions and across simulations
        time_points = np.array(tuple(stats_all[0][stats_fun_names[0]].keys()))
        assert all((np.array_equal(time_points, np.array(tuple(stats[name].keys()))))
                   for name in stats_fun_names for stats in stats_all)

        # separately aggregate by stats function
        if aggfun is None:
            return time_points, stats_all

        agg_stats = {}
        for name in stats_fun_names:
            if np.isscalar(tuple(stats_all[0][name].values())[0]):
                agg_stats[name] = aggfun(np.vstack([np.array(tuple(stats[name].values())) for stats in stats_all]))
            else:
                agg_stats[name] = aggfun(np.stack([np.stack(tuple(stats[name].values())) for stats in stats_all]))
        return time_points, agg_stats

    time_points = np.array(tuple(stats_all[0].keys()))
    assert all((np.array_equal(np.array(tuple(stats.keys())), time_points)) for stats in stats_all)

    if aggfun is None:
        return time_points, stats_all

    if np.isscalar(tuple(stats_all[0].values())[0]):
        stats_all = np.vstack([np.array(tuple(stats.values())) for stats in stats_all])
    else:
        stats_all = np.stack([np.stack(tuple(stats.values())) for stats in stats_all])
    return time_points, aggfun(stats_all)
