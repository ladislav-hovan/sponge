### Imports ###
import bioframe
import time

import pandas as pd

from math import ceil, sqrt
from multiprocessing import Pool
from pathlib import Path
from typing import List, Iterable, Tuple

from sponge.config import MOTIF_URL
from sponge.data_retrieval import download_with_progress

FILTER_INPUT = Tuple[str, pd.DataFrame, Iterable[str], str, int, int, float]

### Functions ###
def filter_edges(
    bb_ref: Path,
    bed_df: pd.DataFrame,
    motif_list: Iterable[str],
    chrom: str,
    start_ind: int,
    final_ind: int,
    score_threshold: float = 400,
) -> pd.DataFrame:
    """
    Filters possible binding site matches for the provided transcription
    factors from a bigbed file. This is done for a number of continuous
    regions of a single chromosome subject to a score threshold.

    Parameters
    ----------
    bb_ref : Path
        Path to a bigbed file that stores all possible matches
    bed_df : pd.DataFrame
        Pandas DataFrame containing the regions of interest in the
        chromosome, typically promoters
    motif_list : Iterable[str]
        Iterable containing the matrix IDs of transcription factors of
        interest
    chrom : str
        Name of the chromosome of interest
    start_ind : int
        Starting index of the region DataFrame (bed_df)
    final_ind : int
        Final index of the region DataFrame (bed_df)
    score_threshold : float, optional
        Score required for selection, by default 400

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing the filtered edges from the
        regions of interest
    """

    df = pd.DataFrame()

    for transcript in bed_df.index[start_ind:final_ind]:
        # Retrieve the start and end points
        start,end = bed_df.loc[transcript][['start', 'end']]
        # Load all matches in that region from the bigbed file
        motifs = bioframe.read_bigbed(bb_ref, chrom, start=start, end=end)
        # Ensure the entire motif fits within range (not default behaviour)
        motifs = motifs[(motifs['start'] >= start) & (motifs['end'] <= end)]
        # Filter only the transcription factors in the list
        max_scores = motifs[motifs['name'].isin(motif_list)].groupby(
            'name')[['score', 'TFName']].max()
        # Filter only high enough scores
        max_scores = max_scores[max_scores['score'] >= score_threshold]
        max_scores.reset_index(inplace=True)
        # Add the transcript (region) name for identification
        max_scores['transcript'] = transcript
        # Append the results to the dataframe
        df = pd.concat((df, max_scores), ignore_index=True, copy=False)

    return df


def iterate_chromosomes(
    promoter_df: pd.DataFrame,
    bigbed_file: Path,
    chromosomes: List[str],
    matrix_ids: List[str],
    n_processes: int = 1,
    score_threshold: float = 400,
) -> List[pd.DataFrame]:
    """
    Iterates over the chromosomes in a bigbed file, intersecting the
    TF binding sites with a list of defined regions and returning the
    result.

    Parameters
    ----------
    promoter_df : pd.DataFrame
        Pandas DataFrame containing the regions of interest
    bigbed_file : Path
        Path to a JASPAR bigbed file
    chromosomes : List[str]
        List of chromosomes to use
    matrix_ids : List[str]
        List of matrix IDs of transcription factors to use
    n_processes : int, optional
        Number of processes to run in parallel, by default 1
    score_threshold : float, optional
        Score required for selection, by default 400

    Returns
    -------
    List[pd.DataFrame]
        List of pandas DataFrames containing the edges between TFs and
        regions of interest
    """

    results_list = []
    p = Pool(n_processes)

    print ()
    print ('Iterating over the chromosomes...')
    for chrom in chromosomes:
        st_chr = time.time()
        df_chrom = promoter_df[promoter_df['chrom'] == chrom]
        if len(df_chrom) == 0:
            suffix = 'no transcripts'
        elif len(df_chrom) == 1:
            suffix = '1 transcript'
        else:
            suffix = f'{len(df_chrom)} transcripts'
        print (f'Chromosome {chrom[3:]} with ' + suffix)
        if len(df_chrom) == 0:
            continue
        # This is a heuristic approximation of the ideal chunk size
        # Based off of performance benchmarking
        chunk_size = ceil(sqrt(len(df_chrom) / n_processes))
        chunk_divisions = [i for i in range(0, len(df_chrom), chunk_size)]
        input_tuples = [(bigbed_file, df_chrom, matrix_ids, chrom, i,
            i+chunk_size, score_threshold) for i in chunk_divisions]
        # Run the calculations in parallel
        result = p.starmap_async(filter_edges, input_tuples,
            chunksize=n_processes)
        edges_chrom_list = result.get()
        results_list += edges_chrom_list
        elapsed_chr = time.time() - st_chr
        print (f'Done in: {elapsed_chr // 60:n} m {elapsed_chr % 60:.2f} s')

    return results_list


def process_chromosome(
    motifs_chrom: pd.DataFrame,
    transcript_df: pd.DataFrame,
    score_threshold: float = 400,
) -> pd.DataFrame:
    """
    Finds the overlap between TF binding sites and regions of interest
    within a single chromosome.

    Parameters
    ----------
    motifs_chrom : pd.DataFrame
        Pandas DataFrame containing the detected TF binding sites
        in a single chromosome
    transcript_df : pd.DataFrame
        Pandas DataFrame containing the regions of interest in a single
        chromosome
    score_threshold : float, optional
        Score required for selection, by default 400

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing the filtered edges from the
        regions of interest in a single chromosome
    """

    df = pd.DataFrame()

    for t in transcript_df.index:
        # Retrieve the start and end points
        start,end = transcript_df.loc[t][['start', 'end']]
        # Find the corresponding TF binding sites using binary search
        id_start = motifs_chrom['start'].searchsorted(start, 'left')
        id_end = motifs_chrom['end'].searchsorted(end, 'right')
        motifs = motifs_chrom.iloc[id_start:id_end]
        if len(motifs) == 0:
            continue
        # Filter only high enough scores
        to_add = {}
        to_add['score'] = motifs['score'].max()
        if to_add['score'] < score_threshold:
            continue
        # Add the transcript (region) name for identification
        to_add['transcript'] = t
        # Append the results to the dataframe
        df = pd.concat((df, pd.DataFrame(to_add, index=[0])),
            ignore_index=True, copy=False)

    return df


def process_motif(
    motif_df: pd.DataFrame,
    bed_df: pd.DataFrame,
    score_threshold: float = 400,
    n_processes: int = 1,
) -> pd.DataFrame:
    """
    Finds all the binding sites within regions of interest over all
    chromosomes for a single TF.

    Parameters
    ----------
    motif_df : pd.DataFrame
        Pandas DataFrame containing the detected TF binding sites in
        the genome
    bed_df : pd.DataFrame
        Pandas DataFrame containing the regions of interest
    score_threshold : float, optional
        Score required for selection, by default 400
    n_processes : int, optional
        Number of processes to run in parallel, by default 1

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing the filtered edges from the
        regions of interest
    """

    p = Pool(n_processes)
    # Find out chromosome boundaries, relying on continuity
    chrom_info = motif_df.reset_index().groupby('chrom')
    chrom_mins = chrom_info['index'].min()
    chrom_maxs = chrom_info['index'].max()
    # Setup input for parallelisation across chromosomes
    input_tuples = [(motif_df.loc[chrom_mins[chr]:chrom_maxs[chr]],
        bed_df[bed_df['chrom'] == chr], score_threshold)
        for chr in chrom_mins.index]
    # Parallelise the filtering across chromosomes
    result = p.starmap_async(process_chromosome, input_tuples,
        chunksize=n_processes)
    result_list = result.get()
    # Merge the results
    df = pd.concat(result_list, ignore_index=True, copy=False)

    return df


def iterate_motifs(
    promoter_df: pd.DataFrame,
    chromosomes: List[str],
    tf_names: List[str],
    matrix_ids: List[str],
    jaspar_release: str,
    assembly: str,
    n_processes: int = 1,
    score_threshold: float = 400,
) -> List[pd.DataFrame]:
    """
    Iterates over the TFs to filter all binding sites within regions of
    interest, downloading the TF tracks as they are required.

    Parameters
    ----------
    promoter_df : pd.DataFrame
        Pandas DataFrame containing the regions of interest
    chromosomes : List[str]
        List of chromosomes to use
    tf_names : List[str]
        List of names of transcription factors to use, ordered the same
        way as matrix_ids
    matrix_ids : List[str]
        List of matrix IDs of transcription factors to use, ordered the
        same way as tf_names
    jaspar_release : str
        JASPAR release used
    assembly : str
        Assembly of the genome used
    n_processes : int, optional
        Number of processes to run in parallel, by default 1
    score_threshold : float, optional
        Score required for selection, by default 400

    Returns
    -------
    List[pd.DataFrame]
        List of pandas DataFrames containing the edges between TFs and
        regions of interest
    """

    results_list = []

    print ()
    print ('Iterating over the transcription factors...')
    for tf,m_id in zip(tf_names, matrix_ids):
        print (f'Processing the TF {tf} with matrix ID {m_id}')
        file_name = f'{m_id}.tsv.gz'
        to_request = [tr.format(
            year=jaspar_release[-4:],
            genome_assembly=assembly) + file_name for tr in MOTIF_URL]
        # Attempt to download the TF track
        try:
            bytes = download_with_progress(to_request)
        except Exception:
            print ('Unable to download', file_name)
            print (f'The TF {tf} will be skipped')
            print ()
            continue
        # Load the downloaded TF track
        MOTIF_COLS = ['chrom', 'start', 'end', 'TFName', 'p-val', 'score',
            'strand']
        motif_df = pd.read_csv(bytes, sep='\t', names=MOTIF_COLS,
            compression='gzip')
        motif_df.drop(columns=['p-val', 'TFName', 'strand'], inplace=True)
        motif_df = motif_df[motif_df['chrom'].isin(chromosomes)]
        # Adjust the scores to match the bigbed ones (max 1000)
        BIGBED_MAX_SCORE = 1000
        motif_df['score'] = motif_df['score'].astype(int).apply(
            lambda x: min(x, BIGBED_MAX_SCORE))
        # Process the individual TF track
        result = process_motif(motif_df, promoter_df, score_threshold,
            n_processes)
        result['name'] = m_id
        result['TFName'] = tf
        # Append the resulting pandas DataFrame to the list
        results_list.append(result)

    return results_list