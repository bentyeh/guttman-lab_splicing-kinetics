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

def splice_site_counts(transcripts, t, stats=None, *, time_steps=None):
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
            - > 0 indicates that the first intron nucleotide has been incorporated
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
        - Isoforms with no introns have value `int(1)`.
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

def junction_counts(
    transcripts,
    t,
    stats=None,
    *,
    time_steps=None,
    method='fractional',
    pos_exon=None,
    pos_intron=None,
    scale_by_length=False):
    '''
    Counts of introns and exons present in transcripts.
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
    - stats: np.ndarray. shape=(n_time_steps, n_introns, 2) or (n_time_steps,). default=None
        Stores computed statistics. See returns.
    - time_steps: list of int. default=None. len=n_time_steps
        Time steps to include in return. If None, all time steps are included.
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
        Keys: time steps
        Values: np.ndarray of shape=(n_introns + 1, 3)=(n_exons, 3), or int
        - If an isoform has introns:
            - stats[t][:, 0]: exon counts
            - stats[t][:-1, 1]: intron counts
            - stats[t][-1, [1, 2]] are always 0
            - stats[t][:, 2]: spliced junction counts
        - If an isoform has no introns: stats[t] is the count of transcripts at time step t
    '''
    assert pos_exon is not None
    assert method in ('any', 'fractional', 'full')
    if scale_by_length:
        assert pos_intron is not None

    n_introns = transcripts.shape[1] - 1

    if stats is None:
        stats = dict()
    if time_steps is None or t in time_steps:
        if n_introns == 0:
            stats[t] = transcripts.shape[0]
        else:
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
                intron_counts *= intron_lengths
            junction_counts = np.nansum(transcripts[:, :-1] == -1, axis=0)
            stats[t] = np.zeros((n_introns + 1, 3))
            stats[t][:, 0] = exon_counts
            stats[t][:-1, 1] = intron_counts
            stats[t][:-1, 2] = junction_counts
    return stats
