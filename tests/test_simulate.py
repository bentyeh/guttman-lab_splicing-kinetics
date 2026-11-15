'''
Tests for modules/simulate.py and modules/stats_transcripts.py

Run from the project directory with `python -m pytest tests/`, or without pytest via
`python tests/test_simulate.py`.
'''

import os
import sys

import numpy as np

# import simulate module
dir_project = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
sys.path.append(os.path.join(dir_project, 'modules'))
import simulate
import stats_transcripts

def basic_params():
    k_init = 1
    k_decay = 0.05
    k_splice = 0.1
    k_elong = 50

    t_wash = 5
    n_time_steps = 10

    gene_length = 1000
    pos_intron = np.array([[100, 200],
                           [115, 130],
                           [130, 250],
                           [400, 500]]).T

    params = (k_init, k_decay, k_splice, k_elong)
    return params, pos_intron, gene_length, n_time_steps, t_wash

def test_zero_transcripts(verbose=False):
    '''
    Check that transcript simulation function appropriately deals with case when transcription rate is 0, leading to
    zero transcripts.
    '''
    args = list(basic_params())
    args[0] = (0,) + tuple(args[0][1:])
    rng = np.random.default_rng(0)
    stats = simulate.simulate_transcripts(
        *args,
        stats_fun=simulate.stats_raw, stats_kwargs=None,
        log10=False, alt_splicing=True, rng=rng)
    if verbose:
        print(stats)
    assert all(x.shape[0] == 0 for x in stats.values())

def test_parallel_single_exon():
    '''
    Check that parallel simulations are aggregated correctly when transcripts have a single exon.
    '''
    args = list(basic_params())
    args[1] = np.zeros((2, 0), dtype=int)
    n = 2
    time_points, stats_all = simulate.parallel_simulations(
        n,
        *args,
        stats_fun=stats_transcripts.splice_site_counts,
        aggfun=np.stack,
        seed=0,
        use_tqdm=False,
        alt_splicing=False
    )
    assert stats_all.shape == (n, len(time_points))

def test_parallel_multi_exon():
    '''
    Check that parallel simulations are aggregated correctly when transcripts have multiple exons.
    '''
    args = list(basic_params())
    n_introns = args[1].shape[1]
    n = 2
    time_points, stats_all = simulate.parallel_simulations(
        n,
        *args,
        stats_fun=stats_transcripts.splice_site_counts,
        aggfun=np.stack,
        seed=0,
        use_pool=False,
        use_tqdm=False,
        alt_splicing=False
    )
    assert stats_all.shape == (n, len(time_points), n_introns, 3)

def test_reproducible_seeds():
    '''
    Check that simulations run with the same seeds yield the same results
    '''
    # single simulations
    args = list(basic_params())
    rng0 = np.random.default_rng(0)
    stats0 = simulate.simulate_transcripts(*args, rng=rng0)
    rng1 = np.random.default_rng(0)
    stats1 = simulate.simulate_transcripts(*args, rng=rng1)
    assert set(stats0.keys()) == set(stats1.keys())
    assert all((np.array_equal(stats0[t], stats1[t], equal_nan=True) for t in stats0.keys()))

    # parallel simulations
    n = 2
    time_points0, stats_all0 = simulate.parallel_simulations(
        n,
        *args,
        seed=0,
        use_pool=False,
        use_tqdm=False
    )
    time_points1, stats_all1 = simulate.parallel_simulations(
        n,
        *args,
        seed=0,
        use_pool=False,
        use_tqdm=False
    )
    assert len(stats_all0) == n
    assert len(stats_all1) == n
    assert set(stats_all0[0].keys()) == set(stats_all0[1].keys())
    assert set(stats_all1[0].keys()) == set(stats_all1[1].keys())
    assert set(stats_all0[0].keys()) == set(stats_all1[1].keys())
    # different seeds within a run give different simulations
    assert not all((np.array_equal(stats_all0[0][t], stats_all0[1][t], equal_nan=True) for t in stats_all0[0].keys()))
    # the same seed across runs gives the same simulation
    assert all((np.array_equal(stats_all0[0][t], stats_all1[0][t], equal_nan=True) for t in stats_all0[0].keys()))

def test_multi_stats_callable():
    '''
    Check that the same results are obtained with multi_stats_callable as with separately computing the stats.
    '''
    args = list(basic_params())
    rng = np.random.default_rng(0)
    stats_wrapped = simulate.simulate_transcripts(
        *args,
        stats_fun=stats_transcripts.multi_stats_callable(
            stats_transcripts.splice_site_counts,
            stats_transcripts.spliced_fraction),
        rng=rng)
    rng = np.random.default_rng(0)
    stats_count = simulate.simulate_transcripts(
        *args,
        stats_fun=stats_transcripts.splice_site_counts,
        rng=rng)
    rng = np.random.default_rng(0)
    stats_fraction = simulate.simulate_transcripts(
        *args,
        stats_fun=stats_transcripts.spliced_fraction,
        rng=rng)
    assert np.array_equal(
        np.stack(tuple(stats_wrapped['splice_site_counts'].values())),
        np.stack(tuple(stats_count.values())),
        equal_nan=True)
    assert np.array_equal(
        np.stack(tuple(stats_wrapped['spliced_fraction'].values())),
        np.stack(tuple(stats_fraction.values())),
        equal_nan=True)

def test_multi_stats():
    '''
    Check that the pickleable multi_stats() function gives the same results as multi_stats_callable().
    '''
    args = list(basic_params())
    rng = np.random.default_rng(0)
    stats_callable = simulate.simulate_transcripts(
        *args,
        stats_fun=stats_transcripts.multi_stats_callable(
            stats_transcripts.splice_site_counts,
            stats_transcripts.spliced_fraction),
        rng=rng)
    rng = np.random.default_rng(0)
    stats_multi = simulate.simulate_transcripts(
        *args,
        stats_fun=stats_transcripts.multi_stats,
        stats_kwargs={'stats_kwargs': {'splice_site_counts': None, 'spliced_fraction': None}},
        rng=rng)
    assert set(stats_callable.keys()) == set(stats_multi.keys())
    for name in stats_callable.keys():
        assert np.array_equal(
            np.stack(tuple(stats_callable[name].values())),
            np.stack(tuple(stats_multi[name].values())),
            equal_nan=True)

def test_spliced_fraction_bounded():
    '''
    Check that spliced_fraction() returns values in [0, 1] (or NaN when an intron has no spliced or unspliced
    transcripts). Before commit 55093ac, this statistic (then named su_ratio) was spliced / unspliced and unbounded.
    '''
    args = list(basic_params())
    rng = np.random.default_rng(0)
    stats = simulate.simulate_transcripts(
        *args,
        stats_fun=stats_transcripts.spliced_fraction,
        rng=rng)
    values = np.stack(tuple(stats.values()))
    finite = values[~np.isnan(values)]
    assert finite.size > 0
    assert np.all((finite >= 0) & (finite <= 1))

def test_parallel_multi_stats():
    '''
    Check that parallel_simulations() handles multiple stats.
    '''
    args = list(basic_params())
    n = 2
    time_points0, agg_stats = simulate.parallel_simulations(
        n,
        *args,
        multi_stats=True,
        seed=0,
        stats_fun=stats_transcripts.multi_stats_callable(
            stats_transcripts.splice_site_counts,
            stats_transcripts.spliced_fraction),
        aggfun=simulate.mean_and_var,
        use_pool=False,
        use_tqdm=False)
    time_points1, (mean_count, var_count) = simulate.parallel_simulations(
        n,
        *args,
        seed=0,
        stats_fun=stats_transcripts.splice_site_counts,
        aggfun=simulate.mean_and_var,
        use_pool=False,
        use_tqdm=False)
    time_points2, (mean_fraction, var_fraction) = simulate.parallel_simulations(
        n,
        *args,
        seed=0,
        stats_fun=stats_transcripts.spliced_fraction,
        aggfun=simulate.mean_and_var,
        use_pool=False,
        use_tqdm=False)
    assert np.array_equal(time_points0, time_points1) and np.array_equal(time_points0, time_points2)
    assert np.array_equal(agg_stats['splice_site_counts'][0], mean_count)
    assert np.array_equal(agg_stats['splice_site_counts'][1], var_count)
    assert np.array_equal(agg_stats['spliced_fraction'][0], mean_fraction, equal_nan=True)
    assert np.array_equal(agg_stats['spliced_fraction'][1], var_fraction, equal_nan=True)

def main():
    tests = [obj for name, obj in globals().items() if name.startswith('test_') and callable(obj)]
    for test in tests:
        test()
        print(f'{test.__name__}: passed')

if __name__ == '__main__':
    main()
