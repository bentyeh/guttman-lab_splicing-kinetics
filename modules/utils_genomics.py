import numpy as np

def parse_annotation_to_pos_intron(df):
    '''
    Parse GTF annotation for a transcript into a `pos_intron` Numpy array denoting intron coordinates.

    Args
    - df: pd.DataFrame
        GTF format with all entries corresponding to a particular transcript. Must contain columns 'feature type',
        'start', and 'end'.
    
    Returns
    - pos_intron: np.ndarray. shape=(2, n_introns)
        Coordinates of introns, 1-indexed. If gene isoform is intronless, pos_intron has shape (2, 0).
    - gene_length: int
        Number of nucleotides from transcription start site to transcription end site
    '''
    start = int(df.loc[df['feature type'] == 'transcript', 'start'])
    end = int(df.loc[df['feature type'] == 'transcript', 'end'])
    gene_length = end - start + 1
    df_exons = df.loc[df['feature type'] == 'exon'].sort_values(['start', 'end'])
    assert df_exons.iloc[0]['start'] == start
    introns = []
    for i in range(1, len(df_exons)):
        introns.append((df_exons.iloc[i-1]['end'] + 1, df_exons.iloc[i]['start'] - 1))
    pos_intron = np.array(introns).T - start + 1
    assert np.all(pos_intron[1, :] - pos_intron[0, :] > 0)
    return gene_length, pos_intron