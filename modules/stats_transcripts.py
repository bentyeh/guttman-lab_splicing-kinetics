import numpy as np

def multi_stats_callable(*stat_funs):
    '''
    Create callable that wraps multiple stats functions together.
    Note that because the callable is not pickleable, this is not compatible
    with multiprocessing pools.

    Arg(s)
    - *stat_funs: one or more stats functions with definition <stat_fun>(transcripts, t, stats=None, *, ...)

    Returns
    - multiple_stats: callable
      - Wrapper function for multiple stats functions. When called, returns the following `stats` object:
        - stats: dict (str: <variable>)
          - Keys are the names of the stats functions
          - Values are the return values of the stats functions

    Example usage:
        pos_exon = utils_genomics.pos_intron_to_pos_exon(pos_intron, gene_length)
        simulate.simulate_transcripts(
            params, pos_intron, gene_length, n_time_steps, t_wash=600,
            stats_fun=stats_transcripts.wrap_multiple_stats(
                stats_transcripts.splice_site_counts,
                stats_transcripts.spliced_fraction,
                stats_transcripts.junction_counts),
            stats_kwargs={'stat_kwargs': {'junction_counts': {'pos_exon': pos_exon}}})
    '''
    stat_funs_names = [stat_fun.__name__ for stat_fun in stat_funs]
    def multiple_stats(transcripts, t, stats=None, *, stat_kwargs=None):
        if stat_kwargs is None:
            stat_kwargs = dict()
        if stats is None:
            stats = {name: dict() for name in stat_funs_names}
        for stat_fun in stat_funs:
            name = stat_fun.__name__
            if name not in stat_kwargs:
                stat_kwargs[name] = dict()
            stats[name] = stat_fun(transcripts, t, stats=stats[name], **stat_kwargs[name])
        return stats
    return multiple_stats

def multi_stats(transcripts, t, stats=None, stats_kwargs=None):
    '''
    Call multiple stats functions.

    Arg(s)
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
    - stats: np.ndarray. shape=(n_time_points, n_introns, 2) or (n_time_points,). default=None
        Stores computed statistics. See returns.
    - stats_kwargs: dict (<str or callable>: dict (str: <variable>))
        - Keys specify which stats functions to call. Keys can be either be names (str) of stats functions
          or callables themselves.
        - Values are dictionaries of keyword-argument pairs for those stats functions.
          A value of None means to use the default arguments for the corresponding stat function.

    Returns
    - stats: dict (function name (str): dict (time point (int): computed statistics (np.ndarray)))
        Keys are the names of the stats functions.
        Values are the return values of the stats functions.

    Example usage:
        pos_exon = utils_genomics.pos_intron_to_pos_exon(pos_intron, gene_length)
        simulate.simulate_transcripts(
            params, pos_intron, gene_length, n_time_steps, t_wash=600,
            stats_fun=stats_transcripts.multi_stats,
            stats_kwargs={'stats_kwargs': {
                'junction_counts': {'pos_exon': pos_exon},
                'spliced_fraction': None,
                'splice_site_counts': None}})
    '''
    assert stats_kwargs is not None and len(stats_kwargs) > 0
    if stats is None:
        stats = dict()
    for stat_fun, stat_kwargs in stats_kwargs.items():
        stat_fun_callable = globals()[stat_fun] if type(stat_fun) == str else stat_fun
        stat_fun_name = stat_fun_callable.__name__
        if stat_kwargs is None:
            stat_kwargs = dict()
        if stat_fun_name not in stats:
            stats[stat_fun_name] = None
        stats[stat_fun_name] = stat_fun_callable(transcripts, t, stats[stat_fun_name], **stat_kwargs)
    return stats

def splice_site_counts(transcripts, t, stats=None, *, time_points=None):
    '''
    Counts of each splice site.

    Arg(s)
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
    - stats: np.ndarray. shape=(n_time_points, n_introns, 2) or (n_time_points,). default=None
        Stores computed statistics. See returns.
    - time_points: list of int. default=None. len=n_time_points
        Time points to include in return. If None, all time steps are included.
    
    Returns
    - stats: dict (int: np.ndarray)
        Keys: time points
        Values: np.ndarray of shape=(n_introns, 3), or int
        - If an isoform has introns:
          - stats[t][i, 0]: count of transcripts at time point t that contain donor bond of intron i
          - stats[t][i, 1]: count of transcripts at time point t that contain acceptor bond of intron i
          - stats[t][i, 2]: count of transcripts at time point t that have spliced intron i
        - If an isoform has no introns: stats[t] is the count of transcripts at time point t
    '''
    n_introns = transcripts.shape[1] - 1
    if stats is None:
        stats = dict()
    if time_points is None or t in time_points:
        if n_introns == 0:
            stats[t] = transcripts.shape[0]
        else:
            stats[t] = np.vstack((np.nansum(transcripts[:, :-1] >= 0, axis=0),
                                  np.nansum(transcripts[:, :-1] > 1, axis=0),
                                  np.nansum(transcripts[:, :-1] == -1, axis=0))).T
    return stats

def spliced_fraction(transcripts, t, stats=None, *, time_points=None):
    '''
    Ratio of spliced to (spliced + unspliced) transcripts for each intron.

    Arg(s)
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
    - stats: np.ndarray. shape=(n_time_points, n_introns, 2) or (n_time_points,). default=None
        Stores computed statistics. See returns.
    - time_points: list of int. default=None. len=n_time_points
        Time points to include in return. If None, all time steps are included.

    Returns
    - stats: dict (int: np.ndarray)
        Keys: time points
        Values: np.ndarray of shape=(n_introns,), or int(1)
        - For a given intron, a transcript can be considered unspliced only if the 3' splice site has been transcribed.
        - Transcripts that have not been elongated past their 3' splice site, or for which the given intron is not
          "spliceable" (due to splicing of a mutually exclusive intron) are not counted in this statistic.
        - Isoforms with no introns have value `int(1)`.

    Note: spliced_fraction gives strictly equal or less information than splice_site_counts.
      spliced_fraction can be calculated as follows from the output (ssc) of splice_site_counts:
        ssc[:, 2] / ssc[:, [1, 2]].sum(axis=1)
    '''
    n_introns = transcripts.shape[1] - 1
    if stats is None:
        stats = dict()
    if time_points is None or t in time_points:
        if n_introns == 0:
            stats[t] = 1
        else:
            stats[t] = np.nansum(transcripts[:, :-1] == -1, axis=0) / \
                       (np.nansum(transcripts[:, :-1] == -1, axis=0) + np.nansum(transcripts[:, :-1] > 1, axis=0))
    return stats

def junction_counts(
    transcripts,
    t,
    stats=None,
    *,
    time_points=None,
    method='fractional',
    pos_exon=None,
    pos_intron=None,
    scale_by_length=False):
    '''
    Counts of introns, exons, and spliced junctions present in transcripts.
    Assumes no alternative splicing (introns are non-overlapping).

    Arg(s)
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
    - stats: np.ndarray. shape=(n_time_points, n_introns, 2) or (n_time_points,). default=None
        Stores computed statistics. See returns.
    - time_points: list of int. default=None. len=n_time_points
        Time points to include in return. If None, all time steps are included.
    - method: str. default='fractional'
        How to count partially-transcribed introns and exons
            'any': increment count if any part of the feature is transcribed
            'fractional': sum fractional counts
            'full': increment count only if the entire feature has been transcribed
    - pos_exon: np.ndarray. shape=(2, n_introns + 1)=(2, n_exons)
        Required. Coordinates of exons, 1-indexed. If gene isoform is intronless, pos_exon has shape (2, 1).
        See utils_genomics.pos_intron_to_pos_exon().
    - pos_intron: np.ndarray. shape=(2, n_introns)
        Required if scale_by_length is True.
        Coordinates of introns, 1-indexed. If gene isoform is intronless, pos_intron has shape (2, 0).
    - scale_by_length: bool. default=False
        Multiple intron and exon counts by their respective lengths.

    Returns
    - stats: dict (int: np.ndarray)
        Keys: time points
        Values: np.ndarray of shape=(n_introns + 1, 3)=(n_exons, 3), or int
        - stats[t][:, 0]: exon counts
        - stats[t][:-1, 1]: intron counts
        - stats[t][-1, [1, 2]] are always 0
        - stats[t][:, 2]: spliced junction counts
    '''
    assert pos_exon is not None
    assert method in ('any', 'fractional', 'full')
    if scale_by_length:
        assert pos_intron is not None

    n_introns = transcripts.shape[1] - 1

    if stats is None:
        stats = dict()
    if time_points is None or t in time_points:
        intron_counts = transcripts[:, :-1].copy()
        exon_lengths = pos_exon[1, :] - pos_exon[0, :] + 1
        if method == 'fractional':
            intron_counts = np.minimum(1, np.maximum(0, intron_counts))
            intron_counts = np.nansum(intron_counts, axis=0)
            exon_counts = (transcripts[:, [-1]] - pos_exon[[0], :] + 1) / exon_lengths[np.newaxis,]
            exon_counts = np.minimum(1, np.maximum(0, exon_counts))
            exon_counts = np.sum(exon_counts, axis=0)
        elif method == 'any':
            intron_counts = np.nansum(intron_counts > 0, axis=0)
            exon_counts = np.sum(transcripts[:, [-1]] - pos_exon[[0], :] >= 0, axis=0)
        else: # method == 'full'
            intron_counts = np.nansum(intron_counts >= 1, axis=0)
            exon_counts = np.sum(transcripts[:, [-1]] - pos_exon[[1], :] >= 0, axis=0)
        if scale_by_length:
            exon_counts *= exon_lengths
            intron_lengths = pos_intron[1, :] - pos_intron[0, :] + 1

            # Explicit multiplication rather than in-place operation (intron_counts *= intron_lengths)
            # helps avoid potential typecasting error. This can occur with the following inputs:
            # - method == 'any' or 'full' such that intron_counts has dtype int64 at this point
            # - pos_intron has dtype float (or any dtype that is not int64) such that intron_lengths
            #     has dtype float (or any dtype that is not int64)
            # 
            # https://github.com/numpy/numpy/pull/6499/files
            # https://stackoverflow.com/questions/38673531/numpy-cannot-cast-ufunc-multiply-output-from-dtype
            intron_counts = intron_counts * intron_lengths
        junction_counts = np.nansum(transcripts[:, :-1] == -1, axis=0)
        stats[t] = np.zeros((n_introns + 1, 3))
        stats[t][:, 0] = exon_counts
        stats[t][:-1, 1] = intron_counts
        stats[t][:-1, 2] = junction_counts
    return stats
