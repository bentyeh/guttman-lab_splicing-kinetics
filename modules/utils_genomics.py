import numpy as np
import pandas as pd
import scipy.sparse

def parse_transcript_gtf_to_introns(
    gtf,
    out_format='gtf',
    relative_coords=False,
    relative_strand=False,
    return_gene_length=True,
    col_feature_type=None,
    col_start=None,
    col_end=None,
    col_strand=None):
    '''
    Parse GTF annotation (which only specifies exons) describing a single transcript into intron coordinates,
    either as a GTF file or `pos_intron` NumPy array.

    Args
    - gtf: pandas.DataFrame
        Either follows the GTF file format (see https://www.gencodegenes.org/pages/data_format.html) or
        has column names given by the col_feature_type, col_start, col_end, and col_strand arguments.
    - out_format: str. default='gtf'
        See Returns.
    - relative_coords: bool. default=False
        Subtract the start coordinate (then add back 1 so that coordinates are 1-indexed) of the transcript
        from all features
    - relative_strand: bool. default=False
        Requires relative_coords to be True.
        If an intron is on the '-' strand, recompute the start and end coordinates relative to
        the orientation of transcription.
    - return_gene_length: bool. default=True
        Return (gene_length, introns). See Returns.
    - col_feature_type, col_start, col_end, col_strand: str. default=None
        Column names for corresponding GTF columns. Specifying these arguments allows this function to be
        used with pandas DataFrames whose columns are not ordered according to the GTF file format.

    Returns
    - gene_length: int
        Only returned if return_gene_length is True.
        Number of nucleotides of the transcript.
    - introns: pd.DataFrame or np.ndarray
        If out_format == 'gtf': output introns as a pandas DataFrame following the GTF format
            Keeps the same additional annotations as the corresponding exons where appropriate
        If out_format == 'pos_intron': coordinates (1-indexed) of introns as a NumPy array of shape=(2, n_introns)
            If gene isoform is intronless, the output has shape (2, 0).
    '''
    assert out_format in ('gtf', 'pos_intron')
    col_feature_type = gtf.columns[2] if col_feature_type is None else col_feature_type
    col_start = gtf.columns[3] if col_start is None else col_start
    col_end = gtf.columns[4] if col_end is None else col_end
    idx_transcript = np.flatnonzero(gtf[col_feature_type] == 'transcript')
    assert len(idx_transcript) == 1
    idx_transcript = gtf.index[idx_transcript[0]]

    start = int(gtf.at[idx_transcript, col_start])
    end = int(gtf.at[idx_transcript, col_end])
    gene_length = end - start + 1

    gtf_exons = gtf.loc[gtf[col_feature_type] == 'exon'].sort_values([col_start, col_end])
    assert gtf_exons.iloc[0][col_start] == start and gtf_exons.iloc[-1][col_end] == end

    if len(gtf_exons) == 1:
        if out_format == 'gtf':
            introns = pd.DataFrame(columns=gtf.columns)
        else:
            introns = np.empty((2, 0), dtype=int)
    else:
        col_strand = gtf.columns[6] if col_strand is None else col_strand
        strand = gtf[col_strand].unique()
        assert len(strand) == 1
        strand = strand[0]

        introns_start = gtf_exons.iloc[:-1, 4].values + 1
        introns_end = gtf_exons.iloc[1:, 3].values - 1

        if out_format == 'pos_intron':
            relative_coords = True
            relative_strand = True

        if relative_coords:
            introns_start = introns_start - start + 1
            introns_end = introns_end - start + 1
        if relative_strand and strand == '-':
            assert relative_coords is True
            introns_start_new = gene_length - introns_end + 1
            introns_end = gene_length - introns_start + 1
            introns_start = introns_start_new

        if out_format == 'pos_intron':
            introns = np.vstack((introns_start, introns_end))
        else:
            col_chr, col_annot, col_score, col_phase, col_additional = gtf.columns[[0, 1, 5, 7, 8]]
            tmp = gtf[[col_chr, col_annot]].drop_duplicates()
            assert len(tmp) == 1
            chrom, annot = tmp.values.squeeze()
            additional = gtf[col_additional].str.replace('exon_number \d+; exon_id "[^"]+"; ', '', regex=True).unique()
            assert len(additional) == 1
            additional = additional[0] + \
                f' gene_length "{gene_length}"; transcript_start "{start}"; transcript_end "{end}"; transcript_strand "{strand}";'
            introns = pd.DataFrame({
                col_chr: chrom,
                col_annot: annot,
                col_feature_type: 'intron',
                col_start: introns_start,
                col_end: introns_end,
                col_score: '.',
                col_strand: '+' if relative_strand else strand,
                col_phase: '.',
                col_additional: additional})
    if return_gene_length:
        return gene_length, introns
    return introns

def parse_gtf_to_introns(
    gtf,
    regex_transcript_id=r'((?:ENSMUST|ENST)\d+\.?\d*)',
    out_format='gtf',
    col_feature_type=None,
    col_additional=None,
    **kwargs):
    '''
    Wrapper around parse_transcript_gtf_to_introns() to parse a GTF file describing multiple
    transcripts into their intron coordinates.

    Args
    - gtf
        Coordinates are 1-indexed, inclusive.
        Note that all exons for a given transcript_id share identical annotation source,
        score, strand, phase, and additional information (except .str.replace('exon_number \d+; exon_id "[^"]+"; ', '', regex=True))
    - regex_transcript_id: str. default=r'((?:ENSMUST|ENST)\d+\.?\d*)'
        Regular expression to match a transcript id in the additional information column of a GTF file
    - out_format: str. default='gtf'
        'gtf' or 'pos_intron'. See parse_transcript_gtf_to_introns()
    - col_feature_type, col_additional: str. default=None
        Column names for corresponding GTF columns. See parse_transcript_gtf_to_introns().
    - **kwargs
        Keyword arguments passed to parse_transcript_gtf_to_introns()

    Returns
      If out_format == 'gtf': pd.DataFrame
        Table of introns in GTF format, keeping the same additional annotations as the corresponding exons where appropriate
      If out_format == 'pos_intron':
        - gene_length: dict(str -> int)
            Map transcript_id to gene length
        - pos_introns: dict(str -> np.ndarray)
            Map transcript_id to a pos_intron NumPy array describing intron coordinates
    '''
    col_feature_type = gtf.columns[2] if col_feature_type is None else col_feature_type
    col_additional = gtf.columns[8] if col_additional is None else col_additional
    introns = gtf.copy()
    introns['transcript_id'] = introns[col_additional].str.extract(f'transcript_id "{regex_transcript_id}"')
    introns = introns.loc[
        (~introns['transcript_id'].isna())
        & (introns[col_feature_type].isin(('transcript', 'exon')))
    ].copy()
    if out_format == 'gtf':
        return introns.groupby('transcript_id', group_keys=False) \
            .apply(lambda df: parse_transcript_gtf_to_introns(
                df,
                col_feature_type=col_feature_type,
                return_gene_length=False,
                **kwargs)) \
            .drop(columns='transcript_id') \
            .reset_index(drop=True)
    else:
        gene_lengths = {}
        pos_introns = {}
        for transcript_id, gtf_transcript in introns.groupby('transcript_id', group_keys=False):
            gene_lengths[transcript_id], pos_introns[transcript_id] = parse_transcript_gtf_to_introns(
                gtf_transcript,
                out_format='pos_intron',
                col_feature_type=col_feature_type,
                return_gene_length=True,
                **kwargs)
        return gene_lengths, pos_introns

def pos_intron_to_pos_exon(pos_intron, gene_length):
    '''
    Return exon coordinates corresponding to intron coordinates, assuming no alternative splicing.

    Args
    - pos_intron: np.ndarray. shape=(2, n_introns)
        Coordinates of introns, 1-indexed. If gene isoform is intronless, pos_intron has shape (2, 0).
        Introns are assumed to be in order and non-overlapping (i.e., no alternative splicing).
    - gene_length: int
        Number of nucleotides from transcription start site to transcription end site

    Returns
    - pos_exon: np.ndarray. shape=(2, n_introns + 1)=(2, n_exons)
        Coordinates of exons, 1-indexed. If gene isoform is intronless, pos_exon has shape (2, 1).
    '''
    n_introns = pos_intron.shape[1]
    if n_introns == 0:
        return np.array([[1], [gene_length]], dtype=int)
    if n_introns > 1:
        assert np.all(pos_intron[:, 1:] - pos_intron[:, :-1] >= 0)
        assert np.all(pos_intron[1, :-1] < pos_intron[0, 1:])
    pos_exon = np.ones((2, n_introns + 1), dtype=int)
    pos_exon[0, 1:] = pos_intron[1, :] + 1
    pos_exon[1, :-1] = pos_intron[0, :] - 1
    pos_exon[1, -1] = gene_length
    return pos_exon

def mutually_exclusive_splicing(intervals):
    '''
    Generate sparse square matrix A indicating mutually exclusive splicing events:
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
    - A: scipy.sparse.dok_array, shape=(n_introns, n_introns), dtype=bool
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
    A = scipy.sparse.dok_array(scipy.sparse.eye(n_introns), dtype=bool)
    for i in range(n_introns - 1):
        for j in range(i + 1, n_introns):
            start1, end1 = intervals[i]
            start2, end2 = intervals[j]
            if start2 > end1:
                break
            A[i, j] = 1
            if (end2 >= end1) or (start1 == start2):
                A[j, i] = 1
    return A
