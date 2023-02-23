'''
Simulate transcription and splicing

Assumes the following project directory structure:
- project directory/
  - modules/
    - simulate.py
    - stats_transcripts.py
    - utils_genomics.py
  - scripts/
    - simulate_main.py (this file)
'''

import argparse
import json
import os
import sys
import textwrap
import numpy as np
import pandas as pd

dir_project = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
sys.path.append(os.path.join(dir_project, 'modules'))
import simulate
import stats_transcripts
import utils_genomics

def move_column_inplace(df, col_name, pos):
    '''
    Move column col_name to position pos in the DataFrame df.
    Operation is performed in-place.
    Source: https://stackoverflow.com/a/58686641
    '''
    col = df.pop(col_name)
    df.insert(pos, col.name, col)

def agg_stats_to_df(agg_stats, stats_fun, time_points):
    '''
    Args
    - agg_stats: dict(int: np.ndarray) or np.ndarray
        Format is described by the table below where
        - dict[t] = dictionary with t=n_time_points keys
        - ? = a variable value (i.e., cannot be determined a priori)

        stats_fun          | n_introns | agg_stats format                         |
        -------------------|-----------|------------------------------------------|
        stats_raw          | n_introns | dict[t](int: np.ndarray[?, n_introns+1]) |
        spliced_fraction   | 1+        | np.ndarray[t, n_introns]                 |
        splice_site_counts | 0         | np.ndarray[t]                            |
        splice_site_counts | 1+        | np.ndarray[t, n_introns, 3]              |
        junction_counts    | n_introns | np.ndarray[t, n_introns+1, 3]            |

    - stats_fun: str or callable
    - time_points: list-like

    Returns: pd.DataFrame
      Columns depend on stats_fun
      - stats_raw: <time_point> <intron_1> ... <intron_N> <polymerase position>
      - spliced_fraction: <time_point> <intron_1> .., <intron_N>
      - splice_site_counts: <time_point> <donor_1> ... <donor_N> <acceptor_1> ... <acceptor_N> <spliced_1> ... <spliced_N>
          If there are no introns, the columns are <time_point> <n_transcripts>.
      - junction_counts: <time_point> <exon_1> ... <exon_N+1> <intron_1> ... <intron_N> <spliced_1> ... <spliced_N>
    '''
    if callable(stats_fun):
        stats_fun = stats_fun.__name__
    assert stats_fun in ('stats_raw', 'spliced_fraction', 'splice_site_counts', 'junction_counts')
    if stats_fun == 'stats_raw':
        results = []
        for t, transcripts in agg_stats.items():
            results.append(pd.DataFrame(transcripts).assign(time_point=time_points[t]))
        n_introns = transcripts.shape[1] - 1
        assert all((df.shape[1] - 1 == n_introns for df in results))
        colnames = [f'intron_{i}' for i in range(1, n_introns+1)] + ['polymerase_position']
        df = pd.concat(results, axis=0, ignore_index=True) \
            .rename(columns=dict(zip(range(n_introns + 1), colnames)))
    elif stats_fun == 'spliced_fraction':
        colnames = [f'intron_{i}' for i in range(1, n_introns+1)]
        df = pd.DataFrame(agg_stats, columns=colnames).assign(time_point=time_points)
    elif stats_fun == 'splice_site_counts':
        if len(agg_stats.shape) == 1:
            df = pd.DataFrame(dict(time_point=time_points, n_transcripts=agg_stats))
        else:
            n_introns = agg_stats.shape[1]
            n_time_points = agg_stats.shape[0]
            df = pd.DataFrame(
                    agg_stats.reshape(-1, 3, order='C'),
                    columns=['donor', 'acceptor', 'spliced']) \
                .assign(
                    index=np.tile(np.arange(1, n_introns + 1), n_time_points),
                    time_point=np.repeat(time_points, n_introns)) \
                .melt(id_vars=['time_point', 'index'], var_name='feature') \
                .pipe(lambda df: df.assign(feature=df['feature'] + '_' + df['index'].astype(str))) \
                .drop(columns='index') \
                .pivot(index='time_point', columns='feature', values='value') \
                .reset_index()
    else: # stats_fun == 'junction_counts'
        n_introns = agg_stats.shape[1] - 1
        n_time_points = agg_stats.shape[0]
        df = pd.DataFrame(
                agg_stats.reshape(-1, 3, order='C'),
                columns=['exon', 'intron', 'spliced']) \
            .assign(
                index=np.tile(np.arange(1, n_introns + 2), n_time_points),
                time_point=np.repeat(time_points, n_introns + 1)) \
            .melt(id_vars=['time_point', 'index'], var_name='feature') \
            .pipe(lambda df: df.loc[~(df['feature'].isin(('introns', 'spliced')) & (df['index'] == n_introns + 1))]) \
            .pipe(lambda df: df.assign(feature=df['feature'] + '_' + df['index'].astype(str))) \
            .drop(columns='index') \
            .pivot(index='time_point', columns='feature', values='value') \
            .reset_index()
            # the first .pipe() drops useless intron and spliced rows
            # (there is 1 less intron and spliced feature than the number of exons)
    move_column_inplace(df, 'time_point', 0)
    return df

def main(
    n,
    params,
    pos_intron,
    gene_length,
    n_time_steps,
    stats_fun=None,
    aggfun=None,
    output_time_points=None,
    file_out=None,
    **kwargs):
    '''
    Wrapper for simulate.parallel_simulations() that reformats simulation results
    as a table.

    Args
    - n, params, pos_intron, gene_length, n_time_steps, **kwargs
        Passed to simulate.simulate_transcripts() via simulate.parallel_simulations().
    - stats_fun: str
        Stats function. See simulate.simulate_transcripts() docstring and the
        --stats_fun argparse argument.
        One of 'stats_raw', 'spliced_fraction', 'splice_site_counts', 'junction_counts'
    - aggfun: str
        One of 'mean', 'mean_nan'. See simulate.mean(), simulate.mean_nan(), and the
        --aggfun argparse argument.
    - output_time_points: list-like of int
        See the time_points argument of various stats functions.
    - file_out: str
        Path to save simulation results. See the --file_out argparse argument.

    Returns: pd.DataFrame
      See the stats_fun argparse argument.
    '''

    stats_kwargs=dict()
    if output_time_points:
        stats_kwargs.update(dict(time_points=output_time_points))
    if stats_fun == 'junction_counts':
        stats_kwargs.update(dict(pos_exon=utils_genomics.pos_intron_to_pos_exon(pos_intron, gene_length)))

    stats_fun_callable = {
        'stats_raw': simulate.stats_raw,
        'spliced_fraction': stats_transcripts.spliced_fraction,
        'splice_site_counts': stats_transcripts.splice_site_counts,
        'junction_counts': stats_transcripts.junction_counts
    }[stats_fun]

    aggfun_callable = {
        'mean': simulate.mean,
        'mean_nan': simulate.mean_nan
    }[aggfun]

    time_points, agg_stats = simulate.parallel_simulations(
        n,
        params,
        pos_intron,
        gene_length,
        n_time_steps,
        stats_fun=stats_fun_callable,
        stats_kwargs=stats_kwargs,
        aggfun=aggfun_callable,
        **kwargs)

    '''
    Possible structures of agg_stats

    | aggfun | stats_fun          | n_introns | output data type                                  |
    |--------|--------------------|-----------|---------------------------------------------------|
    | None   | stats_raw          | n_introns | list[n](dict[t](int: np.ndarray[?, n_introns+1])) |
    | None   | junction_counts    | n_introns | list[n](dict[t](int: np.ndarray[n_introns+1, 3])) |
    | None   | spliced_fraction   | 1+        | list[n](dict[t](int: np.ndarray[n_introns,]))     |
    | None   | spliced_fraction   | 0         | list[n](dict[t](int: 1))                          |
    | None   | splice_site_counts | 1+        | list[n](dict[t](int: np.ndarray[n_introns, 3]))   |
    | None   | splice_site_counts | 0         | list[n](dict[t](int: int))                        |
    | mean   | stats_raw          | n/a       | n/a                                               |
    | mean   | junction_counts    | n_introns | np.ndarray[t, n_introns+1, 3]                     |
    | mean   | spliced_fraction   | 1+        | np.ndarray[t, n_introns]                          |
    | mean   | splice_site_counts | 1+        | np.ndarray[t, n_introns, 3]                       |
    | mean   | splice_site_counts | 0         | np.ndarray[t]                                     |
    '''

    if aggfun is None:
        if stats_fun == 'stats_raw':
            df = pd.concat(
                [agg_stats_to_df(result, stats_fun, time_points).assign(simulation=i)
                 for i, result in enumerate(agg_stats)],
                axis=0,
                ignore_index=True)
        else:
            if np.isscalar(tuple(agg_stats[0].values())[0]):
                df = pd.concat(
                    [agg_stats_to_df(
                        np.array(tuple(result.values())),
                        stats_fun,
                        time_points).assign(simulation=i)
                     for i, result in enumerate(agg_stats)],
                    axis=0,
                    ignore_index=True)
            else:
                df = pd.concat(
                    [agg_stats_to_df(
                        np.stack(tuple(result.values())),
                        stats_fun,
                        time_points).assign(simulation=i)
                     for i, result in enumerate(agg_stats)],
                    axis=0,
                    ignore_index=True)
        move_column_inplace(df, 'simulation', 0)
    else:
        df = agg_stats_to_df(agg_stats, stats_fun, time_points)

    if file_out is None:
        file_out = sys.stdout
    df.to_csv(file_out, sep='\t', index=False)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=textwrap.fill((
            'Simulate transcription. '
            'Specify the simulation parameters either with a JSON file (using the --file_spec argument) '
            'or with individual arguments (--params and --file_annot are required).'
            '\n'
            'The output is a tab-delimited table with a header row whose columns are given as described by the '
            '--stats_fun argument.')),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--file_out',
        type=str,
        help=('Path to save simulation results. '
              'Compression inferred from extension (see pandas.DataFrame.to_csv()).'
              'If not specified, print to standard out.'))
    parser.add_argument(
        '--file_spec',
        type=str,
        help=textwrap.dedent('''
            Path to JSON file giving the specifications of the simulation.
            If provided, all other arguments (except --file_out) are ignored.

            Example JSON file:
                {
                    "params": [1, 0.0005, 0.001, 50],
                    "log10": false,
                    "pos_intron": [[ 828, 1135, 1669, 2363, 2579],
                                   [ 952, 1229, 2122, 2449, 3537]],
                    "gene_length": 3640,
                    "n_time_steps": 15000,
                    "t_wash": 600,
                    "stats_fun": "junction_counts",
                    "aggfun": "mean",
                    "output_time_points": [600, 1200, 1500, 1800, 2100, 2400, 3300, 4200, 5100, 6000, 7800, 15000],
                    "n": 100,
                    "seed": null,
                    "use_pool": true,
                    "use_tqdm": true
                }'''))
    parser.add_argument(
        '--params',
        type=float,
        nargs=4,
        help='Parameter values: k_init, k_decay, k_splice, k_elong')
    parser.add_argument(
        '-log10',
        action='store_true',
        help='Parameter values are given as log10 of their actual value.')
    parser.add_argument(
        '--file_annot',
        type=str,
        help=textwrap.dedent('''
            Path to transcript annotation file, a whitespace-delimited file:

                # ignored comments
                gene_length
                intron1_start   intron1_end
                intron2_start   intron2_end
                ...
                intronN_start   intronN_end

            where all coordinates are 1-indexed (inclusive) relative to
            an assumed TSS at coordinate 1.'''))
    parser.add_argument(
        '--n_time_steps',
        type=int,
        default=15000,
        help='Number of time steps in seconds. (default: 15000)')
    parser.add_argument(
        '--t_wash',
        type=int,
        default=600,
        help='Time (seconds) at which (labeled) transcription initiation stops. (default: 600)')
    parser.add_argument(
        '--stats_fun',
        type=str,
        choices=['stats_raw', 'spliced_fraction', 'splice_site_counts', 'junction_counts'],
        default='junction_counts',
        help=textwrap.dedent('''
            What statistics to compute from the simulated transcripts. (default: junction_counts)

            Brackets <column_name> denote the column names found in the header.
            If --aggfun is not provided, then the first column gives the simulation
            number, followed by the columns specified below.

            stats_raw: <time_point> <intron_1> ... <intron_N> <polymerase position>
              Raw transcripts at each time step. Must concurrently set --aggfun none.

            spliced_fraction: <time_point> <intron_1> .., <intron_N>
              Ratio of spliced to (spliced + unspliced) transcripts for each intron.
              For a given intron, a transcript can be considered unspliced only if
              the 3' splice site has been transcribed. The number of introns is assumed
              to be >0.

            splice_site_counts: <time_point> <donor_1> ... <donor_N> <acceptor_1> ... <acceptor_N> <spliced_1> ... <spliced_N>
              Counts of each splice site (donor, acceptor, or spliced).
              If there are no introns, the columns are <time_point> <n_transcripts>.

            junction_counts: <time_point> <exon_1> ... <exon_N+1> <intron_1> ... <intron_N> <spliced_1> ... <spliced_N>
              Counts of exons, introns, and spliced junctions
              - Assumes no alternative splicing (introns are non-overlapping).
              - Current implementation assumes default options:
                - Count partially-transcribed introns and exons as the fraction-transcribed
                - Do not scale counts by intron/exon lengths
            '''))
    parser.add_argument(
        '--aggfun',
        type=str,
        choices=['mean', 'mean_nan', 'none'],
        default='mean',
        help=textwrap.dedent('''
            Aggregation function over multiple simulations. (default: mean)

            mean: Average output over all simulations.
              Note that NaN values (if present) may be inadvertently propagated.

            mean_nan: Replace all NaN values with 1, then average output over all simulations.
              Use if --stats_fun is spliced_fraction.

            none: The results of each simulation is included in the output,
                  and a <simulation> column is added as the first output column.
            '''))
    parser.add_argument(
        '--output_time_points',
        type=int,
        nargs='*',
        help='Time steps of the simulation to output.')
    parser.add_argument(
        '-n',
        type=int,
        default=100,
        help='number of simulations to aggregate over (default: 100)')
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed to use for random number generation')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose')
    parser.add_argument('-no_pool', dest='use_pool', action='store_false',
                        help='Do not use multiprocessing pool')
    parser.add_argument('-no_progress', dest='use_tqdm', action='store_false',
                        help='Do not show progress bars')
    args = parser.parse_args()

    if args.verbose:
        print(args, file=sys.stderr)
    if args.stats_fun == 'stats_raw':
        assert args.aggfun == 'none'
    if args.aggfun == 'none':
        args.aggfun = None
    if args.file_spec:
        assert os.path.exists(args.file_spec)
        with open(args.file_spec, 'rt') as f:
            # limit size of data to be parsed; check for potentially malicious JSON string;
            # see https://docs.python.org/3/library/json.html
            assert os.path.getsize(args.file_spec) < 1e6
            simulation_args = json.load(f)
        simulation_args['pos_intron'] = np.array(simulation_args['pos_intron'])
    else:
        assert os.path.exists(args.file_annot)
        assert args.params is not None
        gene_length = None
        with open(args.file_annot, 'rt') as f:
            coordinates = []
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                if line == '':
                    break
                if gene_length is None:
                    gene_length = int(line)
                    continue
                start, end = line.split()
                coordinates.append((int(start), int(end)))
        pos_intron = np.array(coordinates).T
        simulation_args = vars(args)
        simulation_args['pos_intron'] = pos_intron
        simulation_args['gene_length'] = gene_length
        simulation_args.pop('file_annot', None)
        simulation_args.pop('file_spec', None)
    main(**simulation_args)
