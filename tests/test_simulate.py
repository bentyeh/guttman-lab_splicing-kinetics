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

def zero_transcripts(verbose=True):
    '''
    Check that transcript simulation function appropriately deals with case when transcription rate is 0, leading to
    zero transcripts.
    '''
    args = list(basic_params())
    args[0][0] = 0
    rng = np.random.default_rng(0)
    stats = simulate.simulate_transcripts(
        *args,
        stats_fun=simulate.stats_raw, stats_kwargs=None,
        log10=False, alt_splicing=True, rng=rng)
    if verbose:
        print(stats)
    assert all(x.shape[0] == 0 for x in stats.values())

def parallel_single_exon(verbose=True):
    '''
    Check that parallel simulations are aggregated correctly when transcripts have a single exon.
    '''
    args = list(basic_params())
    args[1] = np.zeros((2, 0), dtype=int)
    n = 2
    time_points, stats_all = simulate.parallel_simulations(
        n,
        *args,
        stats_fun=stats_transcripts.count_per_splice_site,
        seed=0,
        alt_splicing=False
    )
    assert stats_all.shape == (n, len(time_points))

def parallel_multi_exon(verbose=True):
    '''
    Check that parallel simulations are aggregated correctly when transcripts have a single exon.
    '''
    args = list(basic_params())
    n_introns = args[1].shape[1]
    n = 2
    time_points, stats_all = simulate.parallel_simulations(
        n,
        *args,
        stats_fun=stats_transcripts.count_per_splice_site,
        seed=0,
        use_pool=False,
        alt_splicing=False
    )
    assert stats_all.shape == (n, len(time_points), n_introns, 3)

def reproducible_seeds():
    '''
    Check that simulations run with the same seeds yield the same results
    '''
    # single simulations
    args = list(basic_params())
    n_introns = args[1].shape[1]
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
        use_pool=False
    )
    time_points1, stats_all1 = simulate.parallel_simulations(
        n,
        *args,
        seed=0,
        use_pool=False
    )
    assert len(stats_all0) == n
    assert len(stats_all1) == n
    assert set(stats_all0[0].keys()) == set(stats_all0[1].keys())
    assert set(stats_all1[0].keys()) == set(stats_all1[1].keys())
    assert set(stats_all0[0].keys()) == set(stats_all1[1].keys())
    assert not all((np.array_equal(stats_all0[0][t], stats_all0[1][t], equal_nan=True) for t in stats_all0[0].keys()))
    assert all((np.array_equal(stats_all0[0][t], stats_all1[0][t], equal_nan=True) for t in stats_all0[0].keys()))

def test_wrap_multiple_stats():
    '''
    Check that the same results are obtained with wrap_multiple_stats as with separately computing the stats.
    '''
    args = list(basic_params())
    n_introns = args[1].shape[1]
    rng = np.random.default_rng(0)
    stats_wrapped = simulate.simulate_transcripts(
        *args,
        stats_fun=stats_transcripts.wrap_multiple_stats(
            stats_transcripts.count_per_splice_site,
            stats_transcripts.su_ratio),
        rng=rng)
    rng = np.random.default_rng(0)
    stats_count = simulate.simulate_transcripts(
        *args,
        stats_fun=stats_transcripts.count_per_splice_site,
        rng=rng)
    rng = np.random.default_rng(0)
    stats_ratio = simulate.simulate_transcripts(
        *args,
        stats_fun=stats_transcripts.su_ratio,
        rng=rng)
    assert np.array_equal(
        np.stack(tuple(stats_wrapped['count_per_splice_site'].values())),
        np.stack(tuple(stats_count.values())),
        equal_nan=True)
    assert np.array_equal(
        np.stack(tuple(stats_wrapped['su_ratio'].values())),
        np.stack(tuple(stats_ratio.values())),
        equal_nan=True)

def parallel_multi_stats():
    '''
    Check that parallel_simulations() now handles multiple stats.
    '''
    args = list(basic_params())
    n = 2
    time_points0, agg_stats = simulate.parallel_simulations(
        n,
        *args,
        multi_stats=True,
        seed=0,
        stats_fun=stats_transcripts.wrap_multiple_stats(
            stats_transcripts.count_per_splice_site,
            stats_transcripts.su_ratio),
        aggfun=simulate.mean_and_var,
        use_pool=False)
    time_points1, (mean_count, var_count) = simulate.parallel_simulations(
        n,
        *args,
        seed=0,
        stats_fun=stats_transcripts.count_per_splice_site,
        aggfun=simulate.mean_and_var,
        use_pool=False)
    time_points2, (mean_ratio, var_ratio) = simulate.parallel_simulations(
        n,
        *args,
        seed=0,
        stats_fun=stats_transcripts.su_ratio,
        aggfun=simulate.mean_and_var,
        use_pool=False)
    assert np.array_equal(time_points0, time_points1) and np.array_equal(time_points0, time_points2)
    assert np.array_equal(agg_stats['count_per_splice_site'][0], mean_count)
    assert np.array_equal(agg_stats['count_per_splice_site'][1], var_count)
    assert np.array_equal(agg_stats['su_ratio'][0], mean_ratio)
    assert np.array_equal(agg_stats['su_ratio'][1], var_ratio, equal_nan=True)

def main():
    # zero_transcripts()
    # parallel_single_exon()
    # parallel_multi_exon()
    # reproducible_seeds()
    # test_wrap_multiple_stats()
    parallel_multi_stats()

if __name__ == '__main__':
    main()