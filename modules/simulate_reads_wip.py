
'''
WORK IN PROGRESS - NOT FUNCTIONAL. Not imported by any other module, script, or notebook.

Sketch (December 2022) of simulating sequencing reads from simulated transcripts, so that
simulated data could be compared to real data at the level of reads rather than transcripts.
Formerly modules/todo.py. Several functions are stubs, and main() does not run. See README.md.
'''

def read_to_junction_counts(reads):
    '''
    Arg(s)
    - reads: np.ndarray. shape=(n_reads, 2, n_introns + 1). dtype=bool or int.
        [[[    0, intron1, intron2, ...]
          [exon1,   exon2,   exon3, ...]],
         [[    0, intron1, intron2, ...]
          [exon1,   exon2,   exon3, ...]],
          ...
        ]]] 
        where each element is a boolean indicator of whether the read overlaps that element.
        Assumes that the underlying transcript is linearly ordered: exon1-intron1-exon2-...

    Returns: np.ndarray. shape=(n_introns, 3). dtype=int
      For each intron, whether the read counts as an exon read, intron read, or spliced read.
    '''
    n_introns = reads.shape[1] - 1
    out = np.zeros((n_introns, 3), dtype=int)

    # introns
    out[:, 1] = reads[:, 0, 1:].sum(axis=0)

    # spliced
    out[:, 2] = np.sum((reads[:, 0, 1:] == 0) & reads[:, 1, 1:], axis=0)

    # exon
    out[:, 0] = np.sum((reads[:, 0, 1:] == 0) & (reads[:, 1, :-1] ^ reads[:, 1, 1:]), axis=0)

    # idx = np.flatnonzero(read.any(axis=0))
    # for i in np.union1d(np.minimum(idx, n_introns - 1), np.maximum(idx - 1, 0)):
    #     # exon
    #     out[i, 0] += (read[0, i+1] == 0) & (read[1, i] ^ read[1, i+1])
    #     # intron
    #     out[i, 1] += read[0, i+1] == 1
    #     # spliced
    #     out[i, 2] += (read[0, i+1] == 0) & (read[1, i+1] == 1)
    return out

def count_weighted_reads(
    transcripts, t, stats=None, *,
    n_time_steps=None, pos_intron=None, gene_length,
    mean_fragment_length, read_length, min_length=50, paired_end=False, fpk=0.1):
    '''
    ??

    Consider the following linear gene structure:

    |-----exon1-----|-----intron1-----|-----exon2-----|-----intron2-----|-----exon3-----|
      --read1--
                --read2--
                     --read3--
                 --                    read4-- 

    - The exon1-intron1-exon2 intron unit would have a count vector of [1, 2, 1].
      - read1 is an exon read (reads located exclusively in exon i only or exon i+1 only).
      - read2 and read3 are intron reads (reads overlapping intron i).
      - read4 is a spliced read (reads overlapping both exon i and exon i+1, but not intron i).
    - The exon2-intron2-exon3 intron unit would have a count vector of [1, 0, 0]? <TODO>

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
    - stats: np.ndarray. shape=(n_time_steps, n_introns, 3) or (n_time_steps,). dtype=int. default=None
        Stores computed statistics. See returns.
    - time_steps: list of int. default=None. len=n_time_steps
        Time steps to include in return. If None, all time steps are included.
    - pos_intron: np.ndarray. shape=(2, n_introns). default=None
        Coordinates of introns, 1-indexed. Same as in `simulate_transcripts()`
        Only optional if an isoform has no introns.
    - gene_length: int
        Number of nucleotides from transcription start site to transcription end site. Same as in
        `simulate_transcripts()`
    - mean_fragment_length: int
        Mean fragment length in number of nucleotides
    - read_length: int
        Read length
    - min_length: int. default=50
        Minimum fragment length for sequencing (e.g., from gel cut)
    - paired_end: bool. default=False
        Was the library prepared with paired end sequencing
    - fpk: int. default=0.1
        Fragments per kilobase

    Returns
    - stats: np.ndarray. shape=(n_time_steps, n_introns, 3) or (n_time_steps,)
        stats[t, i, [0, 1, 2]] gives the counts of exon reads, intron reads, and spliced reads
        relative to intron i at time step t. If an isoform has no introns, then `stats` is a vector
        of length n_time_steps representing counts of exon reads at each time step.
    '''
    n_introns = transcripts.shape[1] - 1
    if stats is None:
        assert n_time_steps is not None
        if n_introns == 0:
            stats = np.zeros(n_time_steps, dtype=int)
        else:
            stats = np.zeros((n_time_steps, n_introns, 3), dtype=int)

    if n_introns == 0:
        stats[t] = transcripts[transcripts[:, -1] >= min_length, -1].sum() / 1000 * fpk
        return stats
    
    n_transcripts = transcripts.shape[0]
    # transcript: intron1, intron2, pos
    # pos_intron: [intron1, intron2]
    #             [intron1, intron2]
    intron_lengths = pos_intron[1, :] - pos_intron[0, :] + 1
    exon_lengths = np.append(pos_intron[0, :], gene_length) - np.concatenate(([0], pos_intron[1, :])) - 1
    exon_lengths_cum = np.cumsum(exon_lengths)

    # transcript length = transcript position - total length of spliced introns
    transcript_lengths = transcripts[:, -1] - np.dot(transcripts[:, :-1] == -1, intron_lengths)

    # intron_contrib, exon_contrib: length of each intron or exon present in the transcript
    intron_contrib = np.rint(np.maximum(np.nan_to_num(transcripts[:, :-1], nan=0), 0) * intron_lengths).astype(int)
    exon_contrib = \
        np.maximum(
            np.minimum(
                (transcript_lengths - intron_contrib.sum(axis=1)).reshape(-1, 1) - np.concatenate(([0], exon_lengths_cum[:-1])).reshape(1, -1),
                exon_lengths.reshape(1, -1)),
            0)
    

    # mask_in_intron = (transcripts[:, :-1] > 0) & (transcripts[:, :-1] < 1)
    # intron_contrib[mask_in_intron.any(axis=1)] += transcripts[mask_in_intron.any(axis=1), -1] - pos_intron[0, np.nonzero(mask_in_intron)[1]]
    
    # convert representation of a transcript as described in the arguments to
    # [[0,       exon1],
    #  [intron1, exon2],
    #  [intron2, exon3],
    #  ...]]
    # where each element denotes the contribution of that element to the transcript length
    # shape=(n_transcripts, n_introns + 1, 2)
    transcripts_contrib = np.stack((np.insert(intron_contrib, 0, 0, 1), exon_contrib), axis=2)
    transcripts_cum = np.cumsum(transcript_contrib.reshape(n_transcripts, -1), axis=1).reshape((n_transcripts, -1, 2))
    # transcript_cum = np.cumsum(transcript_contrib).reshape((n_introns + 1, 2)) # <TODO> extend to multiple transcripts

    # representation of a fragment or a read
    # - [0, intron1, intron2]
    # - [exon1, exon2, exon3]

    # representation of a transcript
    # [10, 0, 50]

    # transcript_lengths - ~(transcripts[:, :-1].astype(bool)) @ intron_lengths.reshape(-1, 1)


    # transcript[:-1] * intron_lengths
    # np.flatnonzero(transcript[:-1])[-1]


    # For each transcript, for each fragment:
    # 1. find start position of fragment
    # 2. start position of read = start position of fragment
    # 3. generate read matrix
    # 4. read --> counts
    # 5. weight each read by 1 / # fragments
    # Weight total counts by transcript_length / 1000 * fpk
    reads = []
    for i in range(transcripts.shape[0]):
        transcript_length = transcript_lengths[i]
        transcript_contrib = transcripts_contrib[i]
        transcript_cum = transcripts_cum[i]
        if transcript_length >= min_length:
            fragment_length = min(mean_fragment_length, transcript_length)
            n_fragments = transcript_length - fragment_length + 1
            for start in range(1, n_fragments + 1):
                # fragment shape=(n_introns + 1, 2)
                fragment = transcript_contrib > 0
                fragment[start > transcript_cum] = 0
                fragment[(x[1:] for x in np.nonzero(start + fragment_length - 1 <= transcript_cum))] = 0
                # 5' read
                # fragment[np.unravel_index(np.flatnonzero(start + read_length <= transcript_cum)[1:], transcript_cum.shape)] = 0
                read1 = fragment.copy()
                read1[(x[1:] for x in np.nonzero(start + read_length - 1 <= transcript_cum))] = 0
                # 3' read
                read2 = fragment.copy()
                read2[start + fragment_length - read_length > transcript_cum] = 0
                reads.extend((read1, read2))
            out = read_to_junction_counts(np.stack(reads, axis=-1)) / n_fragments * transcript_length / 1000 * fpk
            out = out if paired_end else out / 2
            stats[t] += out

    return stats

def sample_reads(
    transcripts, t, stats=None, *,
    n_time_steps=None, pos_intron, gene_length,
    mean_fragment_length, read_length, min_length=50, paired_end=False, fpk=0.1):
    # <TODO> Similar as count_weighted_reads(), but sample reads instead of taking weighted average over
    # possible reads
    pass



# dir_scripts = os.path.split(os.path.realpath(__file__))[0]
# sys.path.append(dir_scripts)
# import simulate_reads

# possible representations of transcripts with n introns
# 1. transcript: [intron_1, intron_2, ..., intron_n, pos]
#    - intron_i takes on one of the following values:
#        - np.nan: excluded due to splicing of a mutually exclusive intron
#        - -1: spliced
#        - 0 to 1: fraction present (0 = not yet transcribed, 1 = fully present)
#    - pos is the position of the last nucleotide added to the transcript
# 2. transcript_contrib:
# #  [[0,       exon1],
#     [intron1, exon2],
#     [intron2, exon3],
#     ...]]
#    where each element denotes the contribution of that element to the transcript length
# 3. transcript_cum:
# #  [[0,       exon1],
#     [intron1, exon2],
#     [intron2, exon3],
#     ...]]
#    where each element denotes the position of the last transcribed nucleotide of that element;
#    spliced introns have the same value as the preceding exon; untranscribed regions have value 0
# 4. BED format: a list of continuous intervals, and separately a boolean array of whether each
#    intron has been spliced


        # splice
        # mask_contains_intron = (transcripts[:, [-1]] > pos_intron[1, :]) & (transcripts[:, :-1] == 0)
        # spliced = (mask_contains_intron & (rng.random(transcripts[:, :-1].shape) < k_splice))
        # if alt_splicing is not False:
        #     for j in rng.permutation(n_introns):
        #         set_to_nan = alt_splicing[[j], :].astype(float)
        #         set_to_nan[set_to_nan == 1] = np.nan
        #         set_to_nan[0, j] = 1
        #         mask_collisions = transcripts[:, j] == 1
        #         spliced[mask_collisions, :] *= set_to_nan
        # spliced[transcripts[:, :-1] == 1] = 1 # any intron already spliced remains spliced
        # transcripts[:, :-1] = spliced
    


def main():
    k_init = 0
    k_decay = 0.05
    k_splice = 0.1
    k_elong = 50

    gene_length = 1000
    pos_intron = np.array([[100, 200],
                        [115, 130],
                        [130, 250],
                        [400, 500]]).T
    
    rng = np.random.default_rng(0)
    n_time_steps = 10

    # stats_kwargs = dict(
    #     n_time_steps=n_time_steps,
    #     pos_intron=pos_intron,
    #     gene_length=gene_length,
    #     mean_fragment_length=500,
    #     read_length=120,
    #     min_length=50,
    #     paired_end=False,
    #     fpk=0.1
    # )

    stats = simulate_transcripts(
        (k_init, k_decay, k_splice, k_elong),
        pos_intron, gene_length, n_time_steps=n_time_steps, t_wash=600,
        stats_fun=count_weighted_reads, stats_kwargs=stats_kwargs,
        log10=False, alt_splicing=True, rng=rng)
    
    print(stats)

if __name__ == '__main__':
    main()



def transcripts_to_bam(transcripts, pos_intron, annot):
    pass