import warnings
import numpy as np

def wrap_multiple_stats(*stat_funs):
    '''
    Create callable that wraps multiple stats functions together.

    Arg(s)
    - *stat_funs: one or more stats functions with definition <stat_fun>(transcripts, t, stats=None, *, ...)

    Returns
    - multiple_stats: callable
      - Wrapper function for multiple stats functions. When called, returns the following `stats` object:
        - stats: dict (str: <variable>)
          - Keys are the names of the stats functions
          - Values are the return values of the stats functions
    '''
    stat_funs_names = [stat_fun.__name__ for stat_fun in stat_funs]
    def multiple_stats(transcripts, t, stats=None, *, stat_kwargs=None):
        if stat_kwargs is None:
            stat_kwargs = {name: dict() for name in stat_funs_names}
        if stats is None:
            stats = {name: dict() for name in stat_funs_names}
        for stat_fun in stat_funs:
            name = stat_fun.__name__
            stats[name] = stat_fun(transcripts, t, stats=stats[name], **stat_kwargs[name])
        return stats
    return multiple_stats

def count_per_splice_site(transcripts, t, stats=None, *, time_steps=None):
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
            - 0 indicates that the first intron nucleotide has been incorporated
            - 1 indicates that the last nucleotide of the intron has been incorporated
            - > 1 indicates that the intron is present and that elongation has exceeded the 5' splice site
        - Column n_introns: position of the transcript, 1-indexed
    - t: int
        Current time step
    - stats: np.ndarray. shape=(n_time_steps, n_introns, 2) or (n_time_steps,). default=None
        Stores computed statistics. See returns.
    - time_steps: list of int. default=None. len=n_time_steps
        Time steps to include in return. If None, all time steps are included.
    
    Returns
    - stats: dict (int: np.ndarray)
        Keys: time steps
        Values: np.ndarray of shape=(n_introns, 3), or int
        - If an isoform has introns:
          - stats[t][i, 0]: count of transcripts at time step t that contain donor bond of intron i
          - stats[t][i, 1]: count of transcripts at time step t that contain acceptor bond of intron i
          - stats[t][i, 2]: count of transcripts at time step t that have spliced intron i
        - If an isoform has no introns: stats[t] is the count of transcripts at time step t
    '''
    n_introns = transcripts.shape[1] - 1
    if stats is None:
        stats = dict()
    if time_steps is None or t in time_steps:
        if n_introns == 0:
            stats[t] = transcripts.shape[0]
        else:
            stats[t] = np.vstack((np.nansum(transcripts[:, :-1] >= 0, axis=0),
                                  np.nansum(transcripts[:, :-1] > 1, axis=0),
                                  np.nansum(transcripts[:, :-1] == -1, axis=0))).T
    return stats

def spliced_fraction(transcripts, t, stats=None, *, time_steps=None):
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
            - 0 indicates that the first intron nucleotide has been incorporated
            - 1 indicates that the last nucleotide of the intron has been incorporated
            - > 1 indicates that the intron is present and that elongation has exceeded the 5' splice site
        - Column n_introns: position of the transcript, 1-indexed
    - t: int
        Current time step
    - stats: np.ndarray. shape=(n_time_steps, n_introns, 2) or (n_time_steps,). default=None
        Stores computed statistics. See returns.
    - time_steps: list of int. default=None. len=n_time_steps
        Time steps to include in return. If None, all time steps are included.

    Returns
    - stats: dict (int: np.ndarray)
        Keys: time steps
        Values: np.ndarray of shape=(n_introns,), or int(1)
        - For a given intron, a transcript can be considered unspliced only if the 3' splice site has been transcribed.
        - Transcripts that have not been elongated past their 3' splice site, or for which the given intron is not
          "spliceable" (due to splicing of a mutually exclusive intron) are not counted in this statistic.
        - Isoforms with no exons have value `int(1)`.
    '''
    n_introns = transcripts.shape[1] - 1
    if stats is None:
        stats = dict()
    if time_steps is None or t in time_steps:
        if n_introns == 0:
            stats[t] = 1
        else:
            stats[t] = np.nansum(transcripts[:, :-1] == -1, axis=0) / \
                       (np.nansum(transcripts[:, :-1] == -1, axis=0) + np.nansum(transcripts[:, :-1] > 1, axis=0))
    return stats
