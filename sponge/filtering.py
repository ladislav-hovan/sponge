### Imports ###
import bioframe
import os
import time

import pandas as pd

from math import ceil, sqrt
from multiprocessing import Pool
from pathlib import Path
from typing import List, Iterable, Tuple

from sponge.config import MOTIF_URL
from sponge.file_retrieval import download_with_progress

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
        Iterable containing the names of transcription factors of
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
        A pandas DataFrame containing the filtered edges from the
        regions of interest
    """

    df = pd.DataFrame()

    for transcript in bed_df.index[start_ind:final_ind]:
        # Retrieve the start and end points
        start,end = bed_df.loc[transcript][['start', 'end']]
        # Load all matches in that region from the bigbed file
        motifs = bioframe.read_bigbed(bb_ref, chrom, start=start, end=end)
        # Filter only the transcription factors in the list
        max_scores = motifs[motifs['TFName'].isin(motif_list)].groupby(
            'TFName')[['score', 'name']].max()
        # Filter only high enough scores
        max_scores = max_scores[max_scores['score'] >= score_threshold]
        max_scores.reset_index(inplace=True)
        # Add the transcript (region) name for identification
        max_scores['transcript'] = transcript
        # Append the results to the dataframe
        df = pd.concat((df, max_scores), ignore_index=True, copy=False)

    return df


def iterate_chromosomes(
    df_full: pd.DataFrame,
    bigbed_file: Path,
    chromosomes: List[str],
    tf_names: List[str],
    n_processes: int = 1,
    score_threshold: float = 400,
) -> List[pd.DataFrame]:


    results_list = []
    p = Pool(n_processes)

    print ()
    print ('Iterating over the chromosomes...')
    for chrom in chromosomes:
        st_chr = time.time()
        df_chrom = df_full[df_full['chrom'] == chrom]
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
        input_tuples = [(bigbed_file, df_chrom, tf_names, chrom, i,
            i+chunk_size, score_threshold) for i in chunk_divisions]
        # Run the calculations in parallel
        result = p.starmap_async(filter_edges, input_tuples,
            chunksize=n_processes)
        edges_chrom_list = result.get()
        results_list += edges_chrom_list
        elapsed_chr = time.time() - st_chr
        print (f'Done in: {elapsed_chr // 60:n} m {elapsed_chr % 60:.2f} s')

    return results_list


def iterate_motifs(
    df_full: pd.DataFrame,
    chromosomes: List[str],
    tf_names: List[str],
    matrix_ids: List[str],
    temp_folder: Path,
    jaspar_release: str,
    assembly: str,
    n_processes: int = 1,
    score_threshold: float = 400,
) -> List[pd.DataFrame]:


    for tf,m_id in zip(tf_names, matrix_ids):
        file_name = f'{m_id}.tsv.gz'
        to_request = [tr.format(
            year=jaspar_release[-4:],
            genome_assembly=assembly) + file_name for tr in MOTIF_URL]
        save_path = os.path.join(temp_folder, file_name)

        try:
            download_with_progress(to_request, save_path)
        except Exception as e:
            print (f'Unable to download {file_name}')